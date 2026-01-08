"""
Improved Batch Function Calling Judge

This implementation supports true batch processing with function calling,
allowing for larger batches and better scalability than the current approach.
"""

import logging
import json
import os
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, validator
import time
import hashlib

from entity_resolution_demo.entity_matching.entity_match import PotentialMatch, EntityMatch


# Pydantic models for structured output validation
class NameMatchResult(BaseModel):
    """Structured result for name matching with validation"""
    confidence: float = Field(ge=0.0, le=1.0, description="Match confidence score (0-1)")
    is_match: bool = Field(description="Whether the names refer to the same entity")
    match_type: str = Field(description="Type of match: exact, nickname, partial_last, initial, missing_components, out_of_order, phonetic, cultural, semantic, title_role, unlikely")
    reasoning: str = Field(description="Detailed explanation of the decision process")
    explanation_full: str = Field(description="Concise summary suitable for display to users")
    confidence_factors: Dict[str, float] = Field(default_factory=dict, description="Specific factors that influenced confidence")
    key_evidence: List[str] = Field(default_factory=list, description="Key pieces of evidence (2-4 items)")
    risk_factors: List[str] = Field(default_factory=list, description="Potential concerns or ambiguities")
    
    @validator('match_type')
    def validate_match_type(cls, v):
        if v not in SHARED_MATCH_TYPES:
            raise ValueError(f"match_type must be one of {SHARED_MATCH_TYPES}")
        return v


class BatchMatchResult(BaseModel):
    """Structured result for batch processing"""
    results: List[NameMatchResult] = Field(description="Individual match results")
    processing_time: float = Field(description="Total processing time in seconds")
    success_rate: float = Field(ge=0.0, le=1.0, description="Success rate of the batch")


# Import shared function definitions for consistency
from entity_resolution_demo.entity_matching.shared_function_definitions import (
    BATCH_FUNCTION_DEFINITIONS,
    SHARED_MATCH_TYPES
)


class ImprovedBatchFunctionCallingJudge:
    """
    Improved batch processing with function calling for entity resolution
    
    Features:
    - True batch processing (single LLM call for multiple pairs)
    - Function calling for structured output
    - Better scalability for larger batches
    - Fallback to individual processing if batch fails
    """
    
    def __init__(self, config: Dict[str, Any], batch_size: int = None, es_client=None):
        """Initialize the improved batch function calling judge"""
        self.config = config
        self.batch_size = batch_size or config.get('entity_matching', {}).get('llm', {}).get('batch_size', 20)  # Larger default batch size
        self.logger = logging.getLogger(__name__)
        self.es_client = es_client
        
        # Extract LLM config
        llm_config = config.get('entity_matching', {}).get('llm', {})
        if not llm_config and 'llm' in config:
            llm_config = config['llm']
        
        if not llm_config:
            self.logger.warning("No LLM configuration found. Using default OpenAI configuration.")
            llm_config = {
                'provider': 'openai',
                'model': 'gpt-4',
                'temperature': 0.1,
                'max_tokens': 2000,  # Larger for batch processing
                'enabled': True
            }
        
        # Set up LLM client
        self.provider = llm_config.get('provider', 'openai')
        self.model = llm_config.get('model', 'gpt-4')
        self.temperature = llm_config.get('temperature', 0.1)
        self.max_tokens = llm_config.get('max_tokens', 2000)
        self.llm_enabled = llm_config.get('enabled', True)
        
        # Initialize OpenAI client
        if self.provider == 'openai':
            api_key = llm_config.get('api_key') or os.environ.get('OPENAI_API_KEY')
            if api_key:
                self.logger.info(f"Using API key: {api_key[:5]}...")
                
                # Check if LiteLLM proxy URL is configured
                proxy_url = os.getenv('LITELLM_PROXY_URL')
                if proxy_url:
                    # Fix the URL by removing any trailing paths
                    if '/chat/completions' in proxy_url:
                        proxy_url = proxy_url.split('/chat/completions')[0]
                    
                    self.logger.info(f"Using LiteLLM proxy URL: {proxy_url}")
                    
                    # Use the base URL directly without appending /chat/completions
                    # The OpenAI client will handle the correct endpoint
                    self.client = AsyncOpenAI(api_key=api_key, base_url=proxy_url)
                else:
                    self.client = AsyncOpenAI(api_key=api_key)
            else:
                self.logger.warning("No OpenAI API key found")
                self.llm_enabled = False
        else:
            self.logger.warning(f"Unsupported LLM provider: {self.provider}")
            self.llm_enabled = False
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_calls = 0
        self.successful_calls = 0
        self.fallback_calls = 0
        self.cache = {}
        
        # Fallback strategy configuration
        self.use_fallback = llm_config.get('use_fallback', True)
        self.fallback_judge = None  # Will be set if fallback is needed
        
    def set_fallback_judge(self, fallback_judge):
        """Set the fallback judge for when batch processing fails"""
        self.fallback_judge = fallback_judge
        
    async def judge_batch(self, 
                         pairs: List[Dict[str, Any]], 
                         use_cache: bool = True) -> BatchMatchResult:
        """
        Judge multiple name matches using true batch processing
        
        Args:
            pairs: List of name pairs to match
            use_cache: Whether to use caching for performance
            
        Returns:
            BatchMatchResult: Structured batch results
        """
        if not self.llm_enabled:
            return self._create_fallback_batch_result(pairs, "LLM disabled")
        
        start_time = time.time()
        
        try:
            # Primary strategy: True batch processing with function calling
            results = await self._batch_function_calling_strategy(pairs, use_cache)
            
            processing_time = time.time() - start_time
            success_rate = sum(1 for r in results if r.confidence > 0) / len(results) if results else 0
            
            return BatchMatchResult(
                results=results,
                processing_time=processing_time,
                success_rate=success_rate
            )
            
        except Exception as e:
            self.logger.warning(f"Batch function calling failed: {e}")
            
            # Fallback strategy: Individual processing
            if self.use_fallback and self.fallback_judge:
                self.fallback_calls += 1
                self.logger.info("Using fallback individual processing")
                return await self._fallback_batch_strategy(pairs, use_cache)
            else:
                return self._create_fallback_batch_result(pairs, f"Batch processing failed: {e}")
    
    async def _batch_function_calling_strategy(self, 
                                            pairs: List[Dict[str, Any]], 
                                            use_cache: bool) -> List[NameMatchResult]:
        """Primary strategy: True batch processing with function calling"""
        
        # Process in batches
        all_results = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_results = await self._process_batch_with_function_calling(batch, use_cache)
            all_results.extend(batch_results)
        
        return all_results
    
    async def _process_batch_with_function_calling(self, 
                                                 batch: List[Dict[str, Any]], 
                                                 use_cache: bool) -> List[NameMatchResult]:
        """Process a batch using function calling"""
        
        # Create batch prompt
        prompt = self._create_batch_prompt(batch)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert at entity resolution and name matching. Follow this systematic approach:

THINKING PROCESS (Chain-of-Thought):
1. Initial Analysis: Compare names for immediate similarities (identical, nicknames, initials)
2. Contextual Evaluation: Analyze provided context (job titles, organizations, locations)
3. Uniqueness Assessment: Evaluate name commonality (rare names increase confidence)
4. Pattern Identification: Identify specific match patterns present
5. Final Conclusion: Synthesize findings with confidence score

MATCH PATTERNS WITH EXAMPLES:
- EXACT: "John Smith" vs "John Smith", "Tesla" vs "Tesla, Inc."
- NICKNAME: "Bob" vs "Robert", "Bill" vs "William", "Liz" vs "Elizabeth"
- PARTIAL_LAST: "Smith" vs "John Smith", "Johnson" vs "Robert Johnson"
- INITIAL: "J. Smith" vs "John Smith", "A. Johnson" vs "Alice Johnson"
- MISSING_COMPONENTS: "Phil Carr" vs "Phillip Charles Carr", "Mary" vs "Mary Elizabeth"
- OUT_OF_ORDER: "Smith, John" vs "John Smith", "Johnson, Robert" vs "Robert Johnson"
- PHONETIC: "Smith" vs "Smyth", "Johnson" vs "Jonson", "Taylor" vs "Tayler"
- CULTURAL: "José" vs "Jose", "François" vs "Francois", "Müller" vs "Mueller"
- SEMANTIC: "electric car manufacturer" vs "Tesla", "social media platform" vs "Facebook"
- TITLE_ROLE: "POTUS" vs "Joe Biden", "CEO of Apple" vs "Tim Cook"
- UNLIKELY: "John Smith" vs "Jane Smith", "Apple Inc." vs "Microsoft Corp."

CONFIDENCE SCORING:
- HIGH (0.8-1.0): Unique names with strong evidence, exact matches
- MEDIUM (0.5-0.7): Ambiguous cases, common names with missing components
- LOW (0.0-0.4): Weak evidence, partial matches on common names without context

IMPORTANT CONSIDERATIONS:
- Name Uniqueness: Rare surnames (Karapetyan, Zuckerberg) increase confidence, common surnames (Smith, Johnson) decrease confidence
- Context Importance: Job titles, organizations, locations significantly influence decisions
- Risk Factors: Always identify concerns like "common last name", "initial-only match", "no contextual support"
- Non-Latin Scripts: Provide English translation/transliteration in reasoning (e.g., "テイラー・スウィフト" = "Taylor Swift")
- Independent Evaluation: Evaluate each name pair INDEPENDENTLY. Never refer to other pairs in your reasoning.

Provide structured results with detailed reasoning, confidence factors, key evidence, and risk factors."""
                    },
                    {"role": "user", "content": prompt}
                ],
                tools=[{"type": "function", "function": BATCH_FUNCTION_DEFINITIONS[0]}],
                tool_choice={"type": "function", "function": {"name": "analyze_batch_matches"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract function call arguments
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                function_call = tool_calls[0].function
                if function_call.name == "analyze_batch_matches":
                    arguments = json.loads(function_call.arguments)
                    results = arguments.get('results', [])
                    
                    # Log token usage for performance analysis
                    if hasattr(response, 'usage') and response.usage:
                        self.logger.info(f"Batch function calling - Input tokens: {response.usage.prompt_tokens}, Output tokens: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
                    else:
                        self.logger.warning("No token usage information available in response")
                    
                    # Validate and convert results
                    validated_results = []
                    for result_data in results:
                        try:
                            result = NameMatchResult(**result_data)
                            validated_results.append(result)
                        except Exception as e:
                            self.logger.warning(f"Invalid result in batch: {e}")
                            validated_results.append(self._create_fallback_result("", "", f"Invalid result: {e}"))
                    
                    return validated_results
            
            raise ValueError("No valid function call in response")
            
        except Exception as e:
            self.logger.error(f"Batch function calling strategy failed: {e}")
            raise
    
    def _create_batch_prompt(self, batch: List[Dict[str, Any]]) -> str:
        """Create a batch prompt for multiple name pairs"""
        prompt = "Analyze the following name pairs and determine if they refer to the same entity:\n\n"
        
        for i, pair in enumerate(batch):
            query_name = pair.get('query_name', '')
            candidate_name = pair.get('candidate_name', '')
            context = pair.get('context', '')
            
            prompt += f"{i+1}. Query: '{query_name}' vs Candidate: '{candidate_name}'"
            if context:
                context_snippet = context[:200] + ('...' if len(context) > 200 else '')
                prompt += f"\n   Context: {context_snippet}"
            prompt += "\n"
        
        prompt += "\nAnalyze each pair independently and provide your assessment for each."
        
        return prompt
    
    def _create_fallback_result(self, query_name: str, candidate_name: str, reason: str) -> NameMatchResult:
        """Create a fallback result when all strategies fail"""
        return NameMatchResult(
            confidence=0.0,
            is_match=False,
            match_type="unlikely",
            reasoning=f"Fallback result: {reason}",
            explanation_full=f"Unable to analyze match: {reason}",
            confidence_factors={
                "name_similarity": 0.0,
                "context_support": 0.0,
                "name_uniqueness": 0.5
            },
            key_evidence=["Analysis failed"],
            risk_factors=[reason]
        )
    
    def _create_fallback_batch_result(self, pairs: List[Dict[str, Any]], reason: str) -> BatchMatchResult:
        """Create a fallback batch result"""
        results = []
        for pair in pairs:
            query_name = pair.get('query_name', '')
            candidate_name = pair.get('candidate_name', '')
            results.append(self._create_fallback_result(query_name, candidate_name, reason))
        
        return BatchMatchResult(
            results=results,
            processing_time=0.0,
            success_rate=0.0
        )
    
    async def _fallback_batch_strategy(self, pairs: List[Dict[str, Any]], use_cache: bool) -> BatchMatchResult:
        """Fallback to individual processing if batch processing fails"""
        if not self.fallback_judge:
            return self._create_fallback_batch_result(pairs, "No fallback judge available")
        
        try:
            # Use the fallback judge for individual processing
            results = []
            for pair in pairs:
                query_name = pair.get('query_name', '')
                candidate_name = pair.get('candidate_name', '')
                context = pair.get('context', '')
                
                try:
                    result = await self.fallback_judge.judge_match(query_name, candidate_name, context, use_cache)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Fallback processing failed for {query_name} vs {candidate_name}: {e}")
                    results.append(self._create_fallback_result(query_name, candidate_name, f"Fallback failed: {e}"))
            
            return BatchMatchResult(
                results=results,
                processing_time=0.0,
                success_rate=sum(1 for r in results if r.confidence > 0) / len(results) if results else 0
            )
        except Exception as e:
            self.logger.error(f"Fallback batch strategy failed: {e}")
            return self._create_fallback_batch_result(pairs, f"Fallback failed: {e}")
    
    async def _async_batch_judge_matches(self, batch_input: List[Dict[str, Any]]) -> List[NameMatchResult]:
        """
        Async version of batch processing for compatibility with async contexts.
        This method processes matches in batches of the configured batch_size.
        """
        if not batch_input:
            return []
        
        if not self.llm_enabled:
            self.logger.warning("LLM is disabled. Returning fallback results.")
            return [self._create_fallback_result(
                pair['query_name'], 
                pair['candidate_name'], 
                "LLM disabled"
            ) for pair in batch_input]
        
        start_time = time.time()
        all_results = []
        
        # Process in batches of the configured batch_size
        for i in range(0, len(batch_input), self.batch_size):
            batch = batch_input[i:i + self.batch_size]
            try:
                batch_results = await self._process_batch_with_function_calling(batch, use_cache=True)
                all_results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Error processing batch {i//self.batch_size + 1}: {e}")
                # Add fallback results for this batch
                for pair in batch:
                    all_results.append(self._create_fallback_result(
                        pair['query_name'], 
                        pair['candidate_name'], 
                        f"Batch processing failed: {e}"
                    ))
        
        elapsed_time = time.time() - start_time
        matches_per_second = len(batch_input) / elapsed_time if elapsed_time > 0 else 0
        self.logger.info(f"Batch processed {len(batch_input)} matches in {elapsed_time:.2f}s ({matches_per_second:.1f} matches/sec)")
        
        return all_results
    
    def judge_potential_matches(self, potential_matches: List[PotentialMatch]) -> List[EntityMatch]:
        """
        Judge a list of potential matches using batch function calling.
        This method is required for compatibility with RealTimeEntityMatcher.
        
        Args:
            potential_matches: List of potential matches to judge
            
        Returns:
            List[EntityMatch]: Judged entity matches
        """
        if not potential_matches:
            return []
        
        # If LLM is disabled, convert potential matches to entity matches without LLM judgment
        if not self.llm_enabled:
            self.logger.warning("LLM is disabled. Returning potential matches without LLM judgment.")
            return [pm.to_entity_match() for pm in potential_matches]
        
        # Prepare batch input for function calling
        batch_input = []
        for i, pm in enumerate(potential_matches):
            batch_input.append({
                'pair_index': i,
                'query_name': pm.extracted_entity.name,
                'candidate_name': pm.watched_entity.name,
                'context': pm.extracted_entity.context or ""
            })
        
        # Process batch through function calling
        try:
            # Log the batch input for debugging
            for i, pair in enumerate(batch_input):
                self.logger.debug(f"Batch input {i}: {pair['query_name']} -> {pair['candidate_name']}")
            
            # Use the async batch processing method with proper async handling
            import asyncio
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in a running loop, we need to use a different approach
                    # Create a new event loop in a thread
                    import concurrent.futures
                    import threading
                    
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self._async_batch_judge_matches(batch_input))
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        results = future.result()
                else:
                    # If no loop is running, use asyncio.run
                    results = asyncio.run(self._async_batch_judge_matches(batch_input))
            except Exception as async_error:
                self.logger.warning(f"Async batch call failed: {async_error}, using fallback strategy")
                # Fallback to individual processing
                results = []
                for pair in batch_input:
                    fallback_result = self._create_fallback_result(
                        pair['query_name'],
                        pair['candidate_name'],
                        f"Batch processing failed: {async_error}"
                    )
                    results.append(fallback_result)
            
            # Log the results for debugging
            for i, result in enumerate(results):
                self.logger.debug(f"Batch result {i}: {batch_input[i]['query_name']} -> {batch_input[i]['candidate_name']}")
                self.logger.debug(f"  Match type: {result.match_type}, Confidence: {result.confidence}")
                self.logger.debug(f"  Reasoning: {result.reasoning[:100]}...")
            
            # Convert results to entity matches
            entity_matches = []
            for i, result in enumerate(results):
                pm = potential_matches[i]
                
                # Verify that the result matches the potential match
                if batch_input[i]['query_name'] != pm.extracted_entity.name or batch_input[i]['candidate_name'] != pm.watched_entity.name:
                    self.logger.warning(f"Mismatch between result and potential match: "
                                      f"{batch_input[i]['query_name']} -> {batch_input[i]['candidate_name']} vs "
                                      f"{pm.extracted_entity.name} -> {pm.watched_entity.name}")
                
                # Convert NameMatchResult to the format expected by to_entity_match
                llm_explanation = {
                    'confidence': result.confidence,
                    'match_type': result.match_type,
                    'reasoning': result.reasoning,
                    'explanation_full': result.explanation_full,
                    'confidence_factors': result.confidence_factors,
                    'key_evidence': result.key_evidence,
                    'risk_factors': result.risk_factors
                }
                
                entity_matches.append(pm.to_entity_match(
                    confidence=result.confidence,
                    llm_explanation=llm_explanation
                ))
            
            return entity_matches
        
        except Exception as e:
            self.logger.error(f"Error judging potential matches with batch function calling: {e}")
            # Return potential matches without LLM judgment
            return [pm.to_entity_match() for pm in potential_matches]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_calls = self.total_calls
        success_rate = (self.successful_calls / total_calls) if total_calls > 0 else 0
        fallback_rate = (self.fallback_calls / total_calls) if total_calls > 0 else 0
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'total_calls': total_calls,
            'successful_calls': self.successful_calls,
            'fallback_calls': self.fallback_calls,
            'success_rate': success_rate,
            'fallback_rate': fallback_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate
        }
    
    def clear_cache(self):
        """Clear the performance cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
