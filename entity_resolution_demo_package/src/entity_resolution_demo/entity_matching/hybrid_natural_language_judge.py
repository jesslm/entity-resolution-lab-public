#!/usr/bin/env python3
"""
Hybrid Natural Language Judge

Uses the same detailed prompts as the original approach but with an optimized output schema
that only includes essential fields. This allows us to test if the performance gain comes from
function calling vs natural language output method.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from entity_resolution_demo.entity_matching.hybrid_schemas import HybridNameMatchResult
from entity_resolution_demo.pipeline_runner.config import load_config

logger = logging.getLogger(__name__)

class HybridNaturalLanguageJudge:
    """Judge that uses natural language prompts with optimized output schema."""
    
    def __init__(self, config: Dict[str, Any], batch_size: int = 4, es_client=None):
        """Initialize the hybrid natural language judge."""
        self.config = config
        self.batch_size = batch_size
        self.es_client = es_client
        self.llm_enabled = True
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client with LiteLLM proxy support
        import openai
        import os
        
        api_key = config.get('api_key')
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            self.logger.warning("No OpenAI API key found")
            self.llm_enabled = False
            self.client = None
        else:
            # Set API key in environment for consistency
            os.environ['OPENAI_API_KEY'] = api_key
            
            # Check if LiteLLM proxy URL is configured
            proxy_url = os.getenv('LITELLM_PROXY_URL')
            if proxy_url:
                # Fix the URL by removing any trailing paths
                if '/chat/completions' in proxy_url:
                    proxy_url = proxy_url.split('/chat/completions')[0]
                
                self.logger.info(f"Using LiteLLM proxy URL: {proxy_url}")
                self.client = openai.AsyncOpenAI(base_url=proxy_url)
            else:
                self.client = openai.AsyncOpenAI()
        
        self.logger.info(f"HybridNaturalLanguageJudge initialized with batch_size={self.batch_size}")
    
    async def judge_batch(self, potential_matches: List[Dict]) -> List[HybridNameMatchResult]:
        """Judge a batch of potential matches using natural language prompts."""
        if not self.llm_enabled or not potential_matches:
            return self._create_fallback_results(potential_matches)
        
        try:
            return await self._process_batch_with_natural_language(potential_matches)
        except Exception as e:
            self.logger.error(f"Error in natural language batch processing: {e}")
            return self._create_fallback_results(potential_matches)
    
    async def _process_batch_with_natural_language(self, potential_matches: List[Dict]) -> List[HybridNameMatchResult]:
        """Process batch using natural language prompts with optimized schema."""
        start_time = time.time()
        
        # Create the natural language prompt
        prompt = self._create_natural_language_prompt(potential_matches)
        
        # Make the LLM call
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        # Log token usage
        if hasattr(response, 'usage') and response.usage:
            self.logger.info(f"Hybrid natural language - Input tokens: {response.usage.prompt_tokens}, "
                           f"Output tokens: {response.usage.completion_tokens}, "
                           f"Total: {response.usage.total_tokens}")
        
        # Parse the response
        content = response.choices[0].message.content
        results = self._parse_natural_language_response(content, potential_matches)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Hybrid natural language processed {len(potential_matches)} matches in {processing_time:.2f}s")
        
        return results
    
    def _get_system_prompt(self) -> str:
        """Get the detailed system prompt (same as original approach)."""
        return """You are an expert entity resolution specialist. Your task is to determine if two names refer to the same entity and provide detailed analysis.

## Core Task
Analyze name pairs to determine if they refer to the same entity. Consider various match types and provide confidence scores.

## Match Types
- **EXACT**: Identical names (e.g., "John Smith" vs "John Smith")
- **NICKNAME**: Common nicknames (e.g., "Bob" vs "Robert", "Bill" vs "William")
- **PARTIAL_LAST**: Last name matches, first name differs (e.g., "Smith" vs "John Smith")
- **INITIAL**: Initials vs full names (e.g., "J. Smith" vs "John Smith")
- **MISSING_COMPONENTS**: Missing name parts (e.g., "Phil Carr" vs "Phillip Charles Carr")
- **OUT_OF_ORDER**: Different name order (e.g., "Smith, John" vs "John Smith")
- **PHONETIC**: Similar sounding names (e.g., "Smith" vs "Smyth")
- **CULTURAL**: Different cultural representations (e.g., "José" vs "Jose")
- **SEMANTIC**: Semantic relationships (e.g., "electric car manufacturer" vs "Tesla")
- **TITLE_ROLE**: Titles and roles (e.g., "POTUS" vs "Joe Biden")
- **UNLIKELY**: No clear relationship

## Critical Rules
- Tesla Inc vs Tesla, Inc. = EXACT match (same company, different formatting)
- Tesla CEO vs Tesla Inc = TITLE_ROLE match (CEO refers to the company)
- John Smith vs Jane Smith = UNLIKELY (different first names, same last name)
- President Biden vs Joe Biden = TITLE_ROLE match (President is title for Joe Biden)
- Dr. Smith vs John Smith = TITLE_ROLE match (Dr. is title, Smith is last name)

## Output Format
For each name pair, provide a JSON object with these fields:
- **confidence**: Number between 0.0 and 1.0
- **is_match**: Boolean (true if confidence >= 0.5)
- **match_type**: One of the match types above
- **reasoning**: Brief explanation of your decision
- **key_evidence**: Array of 2-3 key pieces of evidence

Use high confidence (0.8-1.0) for unique names and strong evidence, medium confidence (0.5-0.7) for ambiguous cases, and low confidence (0.0-0.4) for weak evidence."""
    
    def _create_natural_language_prompt(self, potential_matches: List[Dict]) -> str:
        """Create natural language prompt for the batch."""
        prompt = f"Analyze the following {len(potential_matches)} name pairs for entity resolution:\n\n"
        
        for i, match in enumerate(potential_matches, 1):
            query_name = match.get('query_name', 'Unknown')
            candidate_name = match.get('candidate_name', 'Unknown')
            context = match.get('context', 'No additional context')
            
            prompt += f"{i}. Query: '{query_name}' vs Candidate: '{candidate_name}'\n"
            prompt += f"   Context: {context}\n\n"
        
        prompt += "Provide your analysis as a JSON array, with one object per name pair in the same order. Each object should have: confidence, is_match, match_type, reasoning, key_evidence."
        
        return prompt
    
    def _parse_natural_language_response(self, content: str, potential_matches: List[Dict]) -> List[HybridNameMatchResult]:
        """Parse natural language response into structured results."""
        try:
            # Try to extract JSON from the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = content[json_start:json_end]
            parsed_results = json.loads(json_str)
            
            # Convert to HybridNameMatchResult objects
            results = []
            for i, result_data in enumerate(parsed_results):
                if i < len(potential_matches):
                    result = HybridNameMatchResult(
                        confidence=float(result_data.get('confidence', 0.0)),
                        is_match=bool(result_data.get('is_match', False)),
                        match_type=str(result_data.get('match_type', 'unlikely')),
                        reasoning=str(result_data.get('reasoning', '')),
                        key_evidence=list(result_data.get('key_evidence', []))
                    )
                    results.append(result)
                else:
                    # Create fallback result for extra entries
                    results.append(self._create_fallback_result(potential_matches[i] if i < len(potential_matches) else {}))
            
            # Ensure we have results for all potential matches
            while len(results) < len(potential_matches):
                results.append(self._create_fallback_result(potential_matches[len(results)]))
            
            return results[:len(potential_matches)]
            
        except Exception as e:
            self.logger.error(f"Error parsing natural language response: {e}")
            return self._create_fallback_results(potential_matches)
    
    def _create_fallback_results(self, potential_matches: List[Dict]) -> List[HybridNameMatchResult]:
        """Create fallback results when LLM processing fails."""
        return [self._create_fallback_result(match) for match in potential_matches]
    
    def _create_fallback_result(self, match: Dict) -> HybridNameMatchResult:
        """Create a single fallback result."""
        return HybridNameMatchResult(
            confidence=0.1,
            is_match=False,
            match_type="unlikely",
            reasoning="Fallback result due to processing error",
            key_evidence=["Processing error occurred"]
        )
    
    def judge_potential_matches(self, potential_matches: List[Dict]) -> List[HybridNameMatchResult]:
        """Synchronous wrapper for compatibility with RealTimeEntityMatcher."""
        if not potential_matches:
            return []
        
        # Use ThreadPoolExecutor to run async method from sync context
        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, self.judge_batch(potential_matches))
            return future.result()
