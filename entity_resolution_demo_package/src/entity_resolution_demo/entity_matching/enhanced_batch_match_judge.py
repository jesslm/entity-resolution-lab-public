"""
Enhanced Batch Match Judge

Processes multiple name matches in a single LLM call with improved confidence scoring
and more natural language explanations. This version ensures proper handling
of LLM unavailability and never generates fallback explanations programmatically.

(Previously known as enhanced_batch_match_judge_CORRECTED.py)
"""

import logging
import json
import os
import openai
from typing import Dict, List, Any, Tuple, Optional
import time
import hashlib
import requests

import sys
from pathlib import Path

from entity_resolution_demo.entity_matching.entity_match import PotentialMatch, EntityMatch


class EnhancedBatchMatchJudge:
    """
    High-performance batch processing for name match judgments with improved explanations
    
    Features:
    - Batch LLM calls with enhanced prompt
    - Better confidence scoring based on name commonality
    - More natural language explanations
    - Improved handling of multilingual and role-based matches
    - Proper handling of LLM unavailability
    """
    
    def __init__(self, config: Dict[str, Any], batch_size: int = None, es_client=None):
        """Initialize the enhanced batch match judge"""
        # Initialize basic properties
        self.config = config
        self.batch_size = batch_size or config.get('entity_matching', {}).get('llm', {}).get('batch_size', 10)
        self.logger = logging.getLogger(__name__)
        self.es_client = es_client
        
        # Extract LLM config from nested structure if needed
        llm_config = config.get('entity_matching', {}).get('llm', {})
        if not llm_config and 'llm' in config:
            llm_config = config['llm']
        
        # Make sure we have a valid LLM config
        if not llm_config:
            self.logger.warning("No LLM configuration found. Using default OpenAI configuration.")
            llm_config = {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'temperature': 0.1,
                'max_tokens': 2000,
                'enabled': True
            }
        
        # Set up LLM client
        self.provider = llm_config.get('provider', 'openai')
        self.model = llm_config.get('model', 'gpt-3.5-turbo')
        self.temperature = llm_config.get('temperature', 0.1)
        self.max_tokens = llm_config.get('max_tokens', 2000)
        self.llm_enabled = llm_config.get('enabled', True)
        
        # Initialize OpenAI client if using OpenAI
        if self.provider == 'openai':
            # Get API key from config if provided, otherwise from environment
            api_key = llm_config.get('api_key') or os.environ.get('OPENAI_API_KEY')
            if api_key:
                self.logger.info(f"Using API key from {'config' if llm_config.get('api_key') else 'environment'}: {api_key[:5]}...")
                # Force set the API key in both the openai module and the environment
                openai.api_key = api_key
                os.environ['OPENAI_API_KEY'] = api_key
            else:
                self.logger.warning("No OpenAI API key found in config or environment")
                
            # Check if LiteLLM proxy URL is configured
            proxy_url = os.getenv('LITELLM_PROXY_URL')
            if proxy_url:
                # Fix the URL by removing any trailing paths
                if '/chat/completions' in proxy_url:
                    proxy_url = proxy_url.split('/chat/completions')[0]
                
                self.logger.info(f"Using LiteLLM proxy URL: {proxy_url}")
                
                # Use the base URL directly without appending /chat/completions
                # The OpenAI client will handle the correct endpoint
                self.client = openai.OpenAI(base_url=proxy_url)
            else:
                self.client = openai.OpenAI()
        else:
            self.logger.warning(f"Unsupported LLM provider: {self.provider}. Only 'openai' is supported.")
            self.llm_enabled = False
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_llm_calls = 0
        self.total_matches_processed = 0
        
        # Simple in-memory cache for repeated comparisons
        self.match_cache = {}
        
        self.logger.info(f"EnhancedBatchMatchJudge initialized with batch_size={self.batch_size}")
    
    def judge_potential_matches(self, potential_matches: List[PotentialMatch]) -> List[EntityMatch]:
        """
        Judge a list of potential matches using the LLM.
        If LLM is unavailable, return the potential matches with basic explanations.
        """
        if not potential_matches:
            return []
        
        # If LLM is disabled, convert potential matches to entity matches without LLM judgment
        if not self.llm_enabled:
            self.logger.warning("LLM is disabled. Returning potential matches without LLM judgment.")
            return [pm.to_entity_match() for pm in potential_matches]
        
        # Prepare batch input for LLM
        batch_input = []
        for i, pm in enumerate(potential_matches):
            batch_input.append({
                'pair_index': i,
                'query_name': pm.extracted_entity.name,
                'candidate_name': pm.watched_entity.name,
                'context': pm.extracted_entity.context or ""
            })
        
        # Process batch through LLM
        try:
            # Log the batch input for debugging
            for i, pair in enumerate(batch_input):
                self.logger.debug(f"Batch input {i}: {pair['query_name']} -> {pair['candidate_name']}")
            
            results = self.batch_judge_matches(batch_input)
            
            # Log the results for debugging
            for i, result in enumerate(results):
                self.logger.debug(f"Batch result {i}: {result['query_name']} -> {result['candidate_name']}")
                self.logger.debug(f"  Match type: {result.get('match_type', 'unknown')}, Confidence: {result.get('confidence', 0.0)}")
                self.logger.debug(f"  Reasoning: {result.get('reasoning', '')[:100]}...")
            
            # Convert results to entity matches
            entity_matches = []
            for i, result in enumerate(results):
                pm = potential_matches[i]
                
                # Verify that the result matches the potential match
                if result['query_name'] != pm.extracted_entity.name or result['candidate_name'] != pm.watched_entity.name:
                    self.logger.warning(f"Mismatch between result and potential match: "
                                      f"{result['query_name']} -> {result['candidate_name']} vs "
                                      f"{pm.extracted_entity.name} -> {pm.watched_entity.name}")
                
                entity_matches.append(pm.to_entity_match(
                    confidence=result.get('confidence', pm.es_score),
                    llm_explanation=result
                ))
            
            return entity_matches
        
        except Exception as e:
            self.logger.error(f"Error judging potential matches: {e}")
            # Return potential matches without LLM judgment
            return [pm.to_entity_match() for pm in potential_matches]
    
    def batch_judge_matches(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of match pairs through the LLM.
        Returns a list of judgment results with explanations.
        """
        start_time = time.time()
        self.total_matches_processed += len(batch)
        
        # Check cache first
        cached_results = []
        uncached_pairs = []
        
        for pair in batch:
            query_name = pair.get('query_name', '')
            candidate_name = pair.get('candidate_name', '')
            context = pair.get('context', '')
            
            cache_key = self._create_cache_key(query_name, candidate_name, context)
            if cache_key in self.match_cache:
                cached_result = self.match_cache[cache_key].copy()
                cached_result['pair_index'] = pair.get('pair_index', 0)
                cached_result['query_name'] = query_name
                cached_result['candidate_name'] = candidate_name
                cached_result['candidate_metadata'] = pair.get('candidate_metadata', {})
                cached_result['cache_hit'] = True
                cached_results.append(cached_result)
                self.cache_hits += 1
            else:
                pair['cache_key'] = cache_key
                uncached_pairs.append(pair)
                self.cache_misses += 1
        
        # Process uncached pairs
        uncached_results = []
        if uncached_pairs:
            try:
                # Process in batches
                for i in range(0, len(uncached_pairs), self.batch_size):
                    batch_slice = uncached_pairs[i:i + self.batch_size]
                    batch_results = self._process_batch(batch_slice)
                    uncached_results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                # Create error results for uncached pairs
                for pair in uncached_pairs:
                    uncached_results.append(self._create_error_result(pair, str(e)))
        
        # Combine cached and uncached results
        all_results = cached_results + uncached_results
        
        # Sort by original pair index
        all_results.sort(key=lambda x: x.get('pair_index', 0))
        
        elapsed_time = time.time() - start_time
        matches_per_second = len(batch) / elapsed_time if elapsed_time > 0 else 0
        self.logger.info(f"Batch processed {len(batch)} matches in {elapsed_time:.2f}s ({matches_per_second:.1f} matches/sec)")
        self.logger.info(f"Cache hits: {self.cache_hits}, misses: {self.cache_misses}, LLM calls: {self.total_llm_calls}")
        
        return all_results
    
    def _create_cache_key(self, query_name: str, candidate_name: str, context_snippet: str = "") -> str:
        """Create a cache key for name pair comparison"""
        # Ensure all inputs are strings to prevent slice errors
        query_str = str(query_name) if query_name is not None else ""
        candidate_str = str(candidate_name) if candidate_name is not None else ""
        context_str = str(context_snippet) if context_snippet is not None else ""
        
        # Use first 100 chars of context to balance caching with context sensitivity
        context_key = context_str[:100] if context_str else ""
        combined = f"{query_str.lower()}|{candidate_str.lower()}|{context_key}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of match pairs through the LLM"""
        results = []
        
        try:
            # Create batch prompt
            prompt = self._create_batch_prompt(batch)
            
            # Call LLM API
            self.total_llm_calls += 1
            response_text = ""
            
            # Try using the configured client first
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at judging name matches for named entity resolution."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                self.logger.info("Successfully used OpenAI client")
                
                # Log token usage for performance analysis
                if hasattr(response, 'usage') and response.usage:
                    self.logger.info(f"Original batch processing - Input tokens: {response.usage.prompt_tokens}, Output tokens: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
                else:
                    self.logger.warning("No token usage information available in response")
                
            except Exception as e:
                self.logger.warning(f"Error using OpenAI client: {e}. Falling back to direct OpenAI API.")
                
                # Fall back to direct OpenAI API
                # Get the API key directly from the environment to ensure we have the latest value
                api_key = os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                
                self.logger.info(f"Using API key from environment for direct API call: {api_key[:5]}...")
                
                # Use direct OpenAI API with the new client format
                from openai import OpenAI
                direct_client = OpenAI(api_key=api_key)
                
                response = direct_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at judging name matches for named entity resolution."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                self.logger.info("Successfully used direct OpenAI API")
            
            # Parse the response
            parsed_results = self._parse_batch_response(response_text, batch)
            
            # Update cache and add metadata
            for i, result in enumerate(parsed_results):
                pair = batch[i]
                cache_key = pair.get('cache_key')
                
                # Add metadata
                result['query_name'] = pair.get('query_name', '')
                result['candidate_name'] = pair.get('candidate_name', '')
                result['candidate_metadata'] = pair.get('candidate_metadata', {})
                result['pair_index'] = pair.get('pair_index', i)
                result['processing_method'] = 'batch_llm'
                result['cache_hit'] = False
                
                # Cache result
                if cache_key:
                    self.match_cache[cache_key] = result.copy()
                
                results.append(result)
        
        except Exception as e:
            self.logger.error(f"Batch LLM call failed: {e}")
            self.logger.warning(f"Falling back to individual processing for batch of {len(batch)} items")
            
            # Process each pair individually
            for pair in batch:
                try:
                    # Create individual prompt
                    individual_prompt = self._create_individual_prompt(pair)
                    
                    # Call LLM API
                    self.total_llm_calls += 1
                    
                    # Use direct OpenAI API
                    # Get the API key directly from the environment to ensure we have the latest value
                    api_key = os.environ.get('OPENAI_API_KEY')
                    if not api_key:
                        raise ValueError("OpenAI API key not found in environment variables")
                    
                    self.logger.info(f"Using API key from environment for individual API call: {api_key[:5]}...")
                    
                    # Use direct OpenAI API with the new client format
                    from openai import OpenAI
                    direct_client = OpenAI(api_key=api_key)
                    
                    response = direct_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert at judging name matches for named entity resolution."},
                            {"role": "user", "content": individual_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    # Parse response
                    individual_response_text = response.choices[0].message.content.strip()
                    
                    # Parse individual response
                    result = self._parse_individual_response(individual_response_text, pair)
                    
                    # Add metadata
                    result['query_name'] = pair.get('query_name', '')
                    result['candidate_name'] = pair.get('candidate_name', '')
                    result['candidate_metadata'] = pair.get('candidate_metadata', {})
                    result['pair_index'] = pair.get('pair_index', 0)
                    result['processing_method'] = 'individual_llm'
                    result['cache_hit'] = False
                    
                    # Cache result
                    cache_key = pair.get('cache_key')
                    if cache_key:
                        self.match_cache[cache_key] = result.copy()
                    
                    results.append(result)
                
                except Exception as e:
                    self.logger.error(f"Error calling LLM for judgment: {e}")
                    results.append(self._create_error_result(pair, str(e)))
        
        return results
    
    def _create_error_result(self, pair: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create an error result when LLM call fails"""
        return {
            'confidence': 0.0,
            'is_match': False,
            'reasoning': "Error occurred during match evaluation",
            'match_type': 'error',
            'factors': {
                'name_similarity': 0.0,
                'context_support': 0.0
            },
            'original_confidence': 0.0,
            'rarity_multiplier': 1.0,
            'rarity_explanation': "Error occurred",
            'query_name': pair.get('query_name', ''),
            'candidate_name': pair.get('candidate_name', ''),
            'candidate_metadata': pair.get('candidate_metadata', {}),
            'pair_index': pair.get('pair_index', 0),
            'processing_method': 'error_fallback',
            'cache_hit': False,
            'explanation': {
                'confidence_factors': {
                    'name_similarity': 0.0,
                    'context_support': 0.0,
                    'cultural_variation': 0.0,
                    'partial_match_strength': 0.0
                },
                'key_evidence': ["Pre-filtered as non-match"],
                'risk_factors': ["No significant name overlap detected"],
                'detailed_reasoning': "Error occurred during match evaluation"
            }
        }
    
    def _parse_batch_response(self, response_text: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse the LLM response for a batch of match pairs"""
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            # Try to find JSON array in the response
            if not response_text.startswith('['):
                # Look for JSON array in the response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx+1]
                else:
                    raise ValueError("No JSON array found in response")
            
            parsed_results = json.loads(response_text)
            
            if not isinstance(parsed_results, list):
                raise ValueError("Response is not a JSON array")
                
            # Log the parsed results for debugging
            self.logger.debug(f"Parsed {len(parsed_results)} results from LLM response")
            for i, pr in enumerate(parsed_results):
                self.logger.debug(f"Parsed result {i}: {pr.get('match_type', 'unknown')}, "
                               f"Confidence: {pr.get('confidence', 0.0)}, "
                               f"Reasoning: {pr.get('reasoning', '')[:100]}...")
            
            # Check if we have the same number of results as batch items
            if len(parsed_results) != len(batch):
                self.logger.warning(f"Mismatch between parsed results ({len(parsed_results)}) and batch items ({len(batch)})")
            
            results = []
            for i, (parsed_result, pair) in enumerate(zip(parsed_results, batch)):
                # Log the pairing for debugging
                self.logger.debug(f"Pairing result {i} with batch item: {pair.get('query_name', '')} -> {pair.get('candidate_name', '')}")
                self.logger.debug(f"  Result match_type: {parsed_result.get('match_type', 'unknown')}, "
                               f"Confidence: {parsed_result.get('confidence', 0.0)}")
                self.logger.debug(f"  Result reasoning: {parsed_result.get('reasoning', '')[:100]}...")
                
                confidence_factors = parsed_result.get('confidence_factors', {})
                
                if 'name_uniqueness' not in confidence_factors:
                    confidence_factors['name_uniqueness'] = 0.5
                
                confidence_factors['cultural_variation'] = confidence_factors.get('cultural_variation', 0.0)
                confidence_factors['partial_match_strength'] = confidence_factors.get('partial_match_strength', 0.0)
                
                result = {
                    'pair_index': pair.get('pair_index', i),
                    'query_name': pair.get('query_name', ''),
                    'candidate_name': pair.get('candidate_name', ''),
                    'confidence': float(parsed_result.get('confidence', 0.0)),
                    'is_match': bool(parsed_result.get('is_match', False)),
                    'reasoning': str(parsed_result.get('reasoning', 'Batch processed')),
                    'match_type': str(parsed_result.get('match_type', 'unknown')),
                    
                    # Top-level explanation fields
                    'explanation_full': str(parsed_result.get('explanation_full', 'No explanation provided')),
                    'confidence_factors': confidence_factors,
                    'key_evidence': parsed_result.get('key_evidence', []),
                    'risk_factors': parsed_result.get('risk_factors', []),
                    
                    # Also keep the nested explanation for backward compatibility
                    'explanation': {
                        'confidence_factors': confidence_factors,
                        'key_evidence': parsed_result.get('key_evidence', []),
                        'risk_factors': parsed_result.get('risk_factors', [])
                    }
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing batch response: {e}")
            # Return error results
            return [self._create_error_result(pair, str(e)) for pair in batch]
    
    def _parse_individual_response(self, response_text: str, pair: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM response for an individual match pair"""
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            parsed_result = json.loads(response_text)
            
            confidence_factors = parsed_result.get('confidence_factors', {})
            
            if 'name_uniqueness' not in confidence_factors:
                confidence_factors['name_uniqueness'] = 0.5
            
            confidence_factors['cultural_variation'] = confidence_factors.get('cultural_variation', 0.0)
            confidence_factors['partial_match_strength'] = confidence_factors.get('partial_match_strength', 0.0)
            
            result = {
                'confidence': float(parsed_result.get('confidence', 0.0)),
                'is_match': bool(parsed_result.get('is_match', False)),
                'reasoning': str(parsed_result.get('reasoning', 'Individual processed')),
                'match_type': str(parsed_result.get('match_type', 'unknown')),
                
                # Top-level explanation fields
                'explanation_full': str(parsed_result.get('explanation_full', 'No explanation provided')),
                'confidence_factors': confidence_factors,
                'key_evidence': parsed_result.get('key_evidence', []),
                'risk_factors': parsed_result.get('risk_factors', []),
                
                # Also keep the nested explanation for backward compatibility
                'explanation': {
                    'confidence_factors': confidence_factors,
                    'key_evidence': parsed_result.get('key_evidence', []),
                    'risk_factors': parsed_result.get('risk_factors', [])
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error parsing individual response: {e}")
            return self._create_error_result(pair, str(e))
    
    def _create_batch_prompt(self, batch: List[Dict[str, Any]]) -> str:
        """Create an enhanced prompt for batch processing with improved confidence scoring"""
        
        prompt = """### **Revised Prompt: Entity Resolution Match Judge**

### **Role & Task**

You are an expert at named entity resolution and name matching. Your task is to analyze multiple name pairs and, using the provided guidelines, determine with high accuracy whether each pair refers to the same person. 

**CRITICAL INSTRUCTION**: Each name pair MUST be evaluated INDEPENDENTLY. Never refer to other pairs in your reasoning or explanations. Each pair should be treated as if it were the only pair you are evaluating.

You MUST provide a structured JSON output for each pair.

### **Thinking Process (Chain-of-Thought)**

For each name pair, follow these specific steps. This process is crucial for reliable decisions.

1.  **Initial Analysis**: Begin by comparing the two names on a basic level. Identify any immediate similarities such as identical names, nicknames, last-name matches, or initials.
2.  **Contextual Evaluation**: Analyze any provided contextual information (e.g., job titles, affiliations, locations) that supports or contradicts the match. How strong are these context clues?
3.  **Uniqueness Assessment**: Evaluate the commonality and uniqueness of each name component. Acknowledge that common names (e.g., "John Smith") increase ambiguity, while unique names reduce it.
4.  **Pattern Identification**: Based on your analysis, explicitly identify any specific match patterns present (e.g., a nickname match, a missing middle name, a last-name-only match).
5.  **Final Conclusion**: Synthesize all of your findings to form a final conclusion. Determine if a match is likely and assign a final confidence score from 0.0 to 1.0. The reasoning for this score must be tied directly to the evidence you've evaluated.

### **Guidelines for Evaluation**

-----

  * **Match Patterns**: Explicitly check for and identify the following patterns:

      * `exact`: Direct matches or minor formatting differences (e.g., "Tesla" vs "Tesla, Inc.", "Google" vs "Google LLC").
      * `nickname`: Common or known nicknames (e.g., Bob for Robert).
      * `partial_last`: Match on the last name only.
      * `initial`: Match on initials (e.g., J. Smith).
      * `missing_components`: One name is a subset of the other (e.g., Phil Carr vs Phillip Charles Carr).
      * `out_of_order`: Components appear in a different order.
      * `phonetic`: Names that sound similar but are spelled differently.
      * `cultural`: Transliterations or different naming conventions.
      * `semantic`: One name is a descriptive phrase that refers to the other entity (e.g., "electric car manufacturer" → "Tesla, Inc.", "social media platform" → "Facebook").
      * `title_role`: One name is a specific title or role (e.g., "POTUS", "CEO of Apple") and the other is the specific person who holds that title/role (e.g., "Joe Biden", "Tim Cook"). Only use this when matching an official title or position to the person who holds it.
      * `unlikely`: No clear connection, likely a false match.

  * **Confidence Scoring**:

      * **HIGH CONFIDENCE (0.8-1.0)**: Strong evidence like unique names, exact matches, or compelling contextual support. If you assign a high confidence score, your reasoning MUST support this high confidence.
      * **MEDIUM CONFIDENCE (0.5-0.7)**: Ambiguous cases like common names with a missing middle name or a last-name-only match.
      * **LOW CONFIDENCE (0.0-0.4)**: Very weak or contradictory evidence, partial matches on common names, or a likely false match. If your reasoning suggests the entities are different or unlikely to match, you MUST assign a low confidence score.

  * **Risk Factors**: Always list potential concerns that lower confidence. For example: "common last name," "initial-only match," "no contextual support."

### **Important Considerations**

1. **CRITICAL: Independent Evaluation**: Evaluate each name pair INDEPENDENTLY. NEVER refer to other pairs in your reasoning or explanations. Each pair should be treated as a completely separate evaluation with no reference to "previous pairs" or "repeat matches".

2. **Name Uniqueness**: Treat unique names (e.g., "Vladimir Putin") with higher confidence than common names (e.g., "John Smith"). Rare surnames should increase confidence, while common surnames should decrease confidence.

3. **Context Importance**: When context is provided, it should heavily influence your decision, especially for common names. If context supports a match (e.g., same profession, organization, or specific details), this should significantly increase confidence.

4. **Consistency**: Apply the same criteria to all name pairs. Don't be stricter or more lenient with some pairs than others.

5. **Ambiguity**: When in doubt, provide a lower confidence score and list the specific risk factors.

6. **Contextual Clues**: Look for specific contextual clues like professions, organizations, locations, or achievements that can help confirm or refute a match.

7. **Surname Rarity**: Consider how common or rare the surnames are. For example, "Smith" and "Johnson" are very common surnames and should be treated with lower confidence than rare surnames like "Karapetyan" or "Zuckerberg".

8. **Titles and Roles**: When one name is a title or role (e.g., "POTUS", "CEO") and the other is a person's name, consider this a match if the person currently holds that title/role. For example, "POTUS" and "Joe Biden" should be considered a match with high confidence if the context supports it.

9. **Non-Latin Scripts**: When encountering names in non-Latin scripts (e.g., Japanese, Arabic, Cyrillic), ALWAYS provide a translation or transliteration to English in your reasoning. For example, if analyzing "テイラー・スウィフト", note that this is "Taylor Swift" in English. This is CRITICAL for proper cross-language entity matching.

### **Output Format**

-----

For each pair, respond with a JSON object containing:
- "confidence": float between 0.0-1.0
- "is_match": boolean
- "match_type": one of the match patterns defined above
- "reasoning": detailed explanation of your decision
- "explanation_full": concise summary suitable for display to users
- "confidence_factors": object with specific factors that influenced confidence:
  - "name_similarity": 0.0-1.0 (how similar the names are)
  - "context_support": 0.0-1.0 (how much context supports the match)
  - "name_uniqueness": 0.0-1.0 (how unique/rare the names are)
- "key_evidence": list of 2-4 key pieces of evidence that support or contradict the match
- "risk_factors": list of potential concerns or ambiguities

Respond with a JSON array of results in the same order as the input pairs."""
        
        prompt += "\n\nName pairs to analyze (remember to evaluate each pair INDEPENDENTLY, never referring to other pairs in your reasoning):"
        
        for i, pair in enumerate(batch):
            prompt += f"\n{i+1}. Query: '{pair['query_name']}' vs Candidate: '{pair['candidate_name']}'"
            if pair.get('context'):
                context_snippet = pair['context'][:300] + ('...' if len(pair['context']) > 300 else '')
                prompt += f"\n   Context: {context_snippet}"
                prompt += f"\n   IMPORTANT: Use this context to inform your decision. Context is especially important for common names."
        
        prompt += "\n\n**CRITICAL: Respond with ONLY a valid JSON array. No additional text, explanations, or formatting. Each object must have all required fields.**"
        prompt += "\n\nExample format:\n[\n  {\n    \"confidence\": 0.85,\n    \"is_match\": true,\n    \"match_type\": \"nickname\",\n    \"reasoning\": \"Will is a common nickname for William\",\n    \"explanation_full\": \"Names match with nickname variation\",\n    \"confidence_factors\": {\n      \"name_similarity\": 0.9,\n      \"context_support\": 0.8,\n      \"name_uniqueness\": 0.7\n    },\n    \"key_evidence\": [\"Will is nickname for William\", \"Same last name Johnson\"],\n    \"risk_factors\": []\n  }\n]"
        
        return prompt
    
    def _create_individual_prompt(self, pair: Dict[str, Any]) -> str:
        """Create a prompt for an individual match pair"""
        prompt = """### **Entity Resolution Match Judge**

Your task is to analyze this name pair and determine if they refer to the same entity.

**CRITICAL INSTRUCTION**: Evaluate this name pair INDEPENDENTLY, without reference to any other pairs you may have seen. Treat this as a standalone evaluation.

Follow these steps:
1. Compare the names for similarities
2. Analyze any context provided
3. Consider name uniqueness
4. Identify match patterns
5. Make a final determination with confidence score

IMPORTANT: When encountering names in non-Latin scripts (e.g., Japanese, Arabic, Cyrillic), ALWAYS provide a translation or transliteration to English in your reasoning. For example, if analyzing "テイラー・スウィフト", note that this is "Taylor Swift" in English.

Respond with a JSON object containing:
- "confidence": float between 0.0-1.0
- "is_match": boolean
- "match_type": one of: exact, nickname, partial_last, initial, missing_components, out_of_order, phonetic, cultural, unlikely
- "reasoning": detailed explanation of your decision
- "explanation_full": concise summary suitable for display
- "confidence_factors": object with specific factors that influenced confidence
- "key_evidence": list of 2-4 key pieces of evidence
- "risk_factors": list of potential concerns or ambiguities

Respond with JSON only."""
        
        prompt += f"\n\nName pair to analyze:\nQuery: '{pair['query_name']}' vs Candidate: '{pair['candidate_name']}'"
        
        if pair.get('context'):
            context_snippet = pair['context'][:300] + ('...' if len(pair['context']) > 300 else '')
            prompt += f"\nContext: {context_snippet}"
            prompt += f"\nIMPORTANT: Use this context to inform your decision."
        
        return prompt