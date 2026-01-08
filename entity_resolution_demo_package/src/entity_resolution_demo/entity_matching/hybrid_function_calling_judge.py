"""
Hybrid Function Calling Judge - Combines original detailed prompts with optimized output schema.
Maintains quality while reducing JSON output size for better performance.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .hybrid_schemas import HybridNameMatchResult, HYBRID_INDIVIDUAL_FUNCTION_DEFINITIONS, HYBRID_BATCH_FUNCTION_DEFINITIONS


class HybridFunctionCallingJudge:
    """
    Hybrid function calling judge using original detailed prompts with optimized output schema.
    Maintains quality while reducing JSON output size for better performance.
    """
    
    def __init__(self, config: Dict[str, Any], batch_size: int = None, es_client=None):
        """Initialize the hybrid function calling judge"""
        self.config = config
        self.batch_size = batch_size or config.get('entity_matching', {}).get('llm', {}).get('batch_size', 8)
        self.logger = logging.getLogger(__name__)
        self.es_client = es_client
        
        # Token usage tracking
        self.token_usage = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'estimated_cost': 0.0
        }
        self.total_judgment_time = 0.0
        
        # Cost calculation (GPT-4 Turbo pricing as of 2024)
        self.input_token_cost = 0.00001  # $0.01 per 1K input tokens
        self.output_token_cost = 0.00003  # $0.03 per 1K output tokens
        
        # Extract LLM config
        llm_config = config.get('entity_matching', {}).get('llm', {})
        if not llm_config and 'llm' in config:
            llm_config = config['llm']
        
        # Initialize LLM client
        self.model = llm_config.get('model', 'gpt-4')
        self.temperature = llm_config.get('temperature', 0.0)  # Use 0 for deterministic function calling
        self.max_tokens = llm_config.get('max_tokens', 2000)
        self.llm_enabled = llm_config.get('enabled', True)
        
        # Initialize OpenAI client with LiteLLM proxy support
        import openai
        import os
        
        api_key = llm_config.get('api_key') or config.get('api_key')
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            self.logger.warning("No OpenAI API key found")
            self.llm_enabled = False
            self.client = None
        else:
            # Set API key in environment for consistency
            os.environ['OPENAI_API_KEY'] = api_key
            
            # Check if LiteLLM proxy URL is configured (same as original approach)
            proxy_url = os.getenv('LITELLM_PROXY_URL')
            if proxy_url:
                # Fix the URL by removing any trailing paths
                if '/chat/completions' in proxy_url:
                    proxy_url = proxy_url.split('/chat/completions')[0]
                
                self.logger.info(f"Using LiteLLM proxy URL: {proxy_url}")
                self.client = openai.AsyncOpenAI(base_url=proxy_url)
            else:
                self.client = openai.AsyncOpenAI()
        
        self.logger.info(f"HybridFunctionCallingJudge initialized with batch_size={self.batch_size}")
    
    def _create_detailed_prompt(self, query_name: str, candidate_name: str, context: str) -> str:
        """Create a detailed prompt using original approach guidance."""
        prompt = f"""Analyze this name pair for entity resolution:

Query Name: {query_name}
Candidate Name: {candidate_name}
Context: {context}

Consider all possible match patterns and provide a structured analysis."""
        
        return prompt
    
    def _create_detailed_batch_prompt(self, batch: List[Dict[str, Any]]) -> str:
        """Create a detailed batch prompt using original approach guidance."""
        prompt = "Analyze these name pairs for entity resolution:\n\n"
        
        for i, pair in enumerate(batch, 1):
            prompt += f"Match {i}:\n"
            prompt += f"  Query Name: {pair['query_name']}\n"
            prompt += f"  Candidate Name: {pair['candidate_name']}\n"
            prompt += f"  Context: {pair['context']}\n\n"
        
        return prompt
    
    def _calculate_cost(self):
        """Calculate estimated cost based on token usage."""
        input_cost = self.token_usage['total_input_tokens'] * self.input_token_cost
        output_cost = self.token_usage['total_output_tokens'] * self.output_token_cost
        self.token_usage['estimated_cost'] = round(input_cost + output_cost, 4)
    
    def _create_fallback_result(self, query_name: str, candidate_name: str, error_msg: str) -> HybridNameMatchResult:
        """Create a fallback result for error cases."""
        return HybridNameMatchResult(
            confidence=0.0,
            is_match=False,
            match_type="unlikely",
            reasoning=f"Error: {error_msg}",
            key_evidence=[]
        )
    
    async def judge_match(self, query_name: str, candidate_name: str, context: str = "") -> HybridNameMatchResult:
        """Judge a single name match using hybrid function calling."""
        if not self.llm_enabled:
            self.logger.warning("LLM is disabled. Returning fallback result.")
            return self._create_fallback_result(query_name, candidate_name, "LLM disabled")
        
        try:
            return await self._hybrid_function_calling_strategy(query_name, candidate_name, context)
        except Exception as e:
            self.logger.error(f"Hybrid function calling strategy failed: {e}")
            return self._create_fallback_result(query_name, candidate_name, str(e))
    
    async def _hybrid_function_calling_strategy(self, query_name: str, candidate_name: str, context: str) -> HybridNameMatchResult:
        """Primary strategy: Use hybrid function calling for structured output."""
        prompt = self._create_detailed_prompt(query_name, candidate_name, context)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at entity resolution. Analyze name pairs and determine if they refer to the same entity. Use the function to return structured results."
                    },
                    {"role": "user", "content": prompt}
                ],
                tools=[{"type": "function", "function": HYBRID_INDIVIDUAL_FUNCTION_DEFINITIONS[0]}],
                tool_choice={"type": "function", "function": {"name": "analyze_name_match"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Log token usage
            if hasattr(response, 'usage') and response.usage:
                self.logger.info(f"Hybrid individual function calling - Input tokens: {response.usage.prompt_tokens}, Output tokens: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
            
            # Extract function call arguments
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                function_call = tool_calls[0].function
                if function_call.name == "analyze_name_match":
                    arguments = json.loads(function_call.arguments)
                    return HybridNameMatchResult(**arguments)
            
            raise ValueError("No valid function call in response")
            
        except Exception as e:
            self.logger.error(f"Hybrid function calling strategy failed: {e}")
            raise
    
    async def judge_batch(self, pairs: List[Dict[str, Any]], use_cache: bool = True) -> List[HybridNameMatchResult]:
        """Judge multiple name matches using hybrid batch function calling."""
        if not self.llm_enabled:
            self.logger.warning("LLM is disabled. Returning fallback results.")
            return [self._create_fallback_result(pair['query_name'], pair['candidate_name'], "LLM disabled") for pair in pairs]
        
        start_time = time.time()
        results = []
        
        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            try:
                batch_results = await self._process_batch_with_hybrid_function_calling(batch, use_cache)
                results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Error processing batch {i//self.batch_size + 1}: {e}")
                # Add fallback results for this batch
                for pair in batch:
                    results.append(self._create_fallback_result(
                        pair['query_name'], 
                        pair['candidate_name'], 
                        f"Batch processing failed: {e}"
                    ))
        
        elapsed_time = time.time() - start_time
        matches_per_second = len(pairs) / elapsed_time if elapsed_time > 0 else 0
        self.logger.info(f"Hybrid batch processed {len(pairs)} matches in {elapsed_time:.2f}s ({matches_per_second:.1f} matches/sec)")
        
        return results
    
    async def _process_batch_with_hybrid_function_calling(self, batch: List[Dict[str, Any]], use_cache: bool) -> List[HybridNameMatchResult]:
        """Process a batch using hybrid function calling."""
        start_time = time.time()
        prompt = self._create_detailed_batch_prompt(batch)
        
        try:
            self.logger.info(f"Using temperature: {self.temperature} for batch processing")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at entity resolution. Analyze multiple name pairs and determine if they refer to the same entity. Use the function to return structured results."
                    },
                    {"role": "user", "content": prompt}
                ],
                tools=[{"type": "function", "function": HYBRID_BATCH_FUNCTION_DEFINITIONS[0]}],
                tool_choice={"type": "function", "function": {"name": "analyze_batch_matches"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.token_usage['total_input_tokens'] += response.usage.prompt_tokens
                self.token_usage['total_output_tokens'] += response.usage.completion_tokens
                self.token_usage['total_tokens'] += response.usage.total_tokens
                self._calculate_cost()  # Update cost after each API call
                self.logger.info(f"Hybrid batch function calling - Input tokens: {response.usage.prompt_tokens}, Output tokens: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
            
            # Extract function call arguments
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                function_call = tool_calls[0].function
                if function_call.name == "analyze_batch_matches":
                    arguments = json.loads(function_call.arguments)
                    results = arguments.get('results', [])
                    
                    # Validate and convert results
                    validated_results = []
                    for result_data in results:
                        try:
                            result = HybridNameMatchResult(**result_data)
                            validated_results.append(result)
                        except Exception as e:
                            self.logger.warning(f"Invalid result in batch: {e}")
                            validated_results.append(self._create_fallback_result("", "", f"Invalid result: {e}"))
                    
                    # Track timing
                    batch_time = time.time() - start_time
                    self.total_judgment_time += batch_time
                    
                    return validated_results
            
            raise ValueError("No valid function call in response")
            
        except Exception as e:
            self.logger.error(f"Hybrid batch function calling strategy failed: {e}")
            raise
    
    def judge_potential_matches(self, potential_matches: List[Any]) -> List[Any]:
        """Synchronous wrapper for compatibility with RealTimeEntityMatcher."""
        # Use ThreadPoolExecutor to run async method from sync context
        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, self._async_judge_potential_matches(potential_matches))
            return future.result()
    
    async def _async_judge_potential_matches(self, potential_matches: List[Any]) -> List[Any]:
        """Async version of judge_potential_matches."""
        if not potential_matches:
            return []
        
        # Convert to batch format - handle both PotentialMatch objects and dictionaries
        batch_input = []
        for match in potential_matches:
            if hasattr(match, 'extracted_entity') and hasattr(match, 'watched_entity'):
                # It's a PotentialMatch object
                batch_input.append({
                    'query_name': match.watched_entity.name,
                    'candidate_name': match.extracted_entity.name,
                    'context': getattr(match, 'context', '')
                })
            else:
                # It's a dictionary
                batch_input.append({
                    'query_name': match.get('query_name', ''),
                    'candidate_name': match.get('candidate_name', ''),
                    'context': match.get('context', '')
                })
        
        # Get hybrid function calling results
        hybrid_results = await self.judge_batch(batch_input)
        
        # Convert hybrid results back to EntityMatch objects
        entity_matches = []
        for i, result in enumerate(hybrid_results):
            pm = potential_matches[i]
            
            # Create LLM explanation dictionary
            llm_explanation = {
                'match_type': result.match_type,
                'is_match': result.is_match,
                'reasoning': result.reasoning,
                'key_evidence': result.key_evidence
            }
            
            # Create EntityMatch from PotentialMatch and hybrid result
            entity_match = pm.to_entity_match(
                confidence=result.confidence,
                llm_explanation=llm_explanation
            )
            entity_matches.append(entity_match)
        
        return entity_matches
