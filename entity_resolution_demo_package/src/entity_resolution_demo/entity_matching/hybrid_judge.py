"""
Hybrid Judge - Non-disruptive implementation switching

Allows switching between the original EnhancedBatchMatchJudge and the new
FunctionCallingJudge without disrupting current functionality.

This provides a safe migration path and allows A/B testing of both approaches.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from entity_resolution_demo.entity_matching.enhanced_batch_match_judge import EnhancedBatchMatchJudge
from entity_resolution_demo.entity_matching.function_calling_judge import FunctionCallingJudge, NameMatchResult, BatchMatchResult
from entity_resolution_demo.entity_matching.entity_match import PotentialMatch, EntityMatch


class HybridJudge:
    """
    Hybrid judge that can switch between original and function calling implementations
    
    Features:
    - Non-disruptive implementation switching
    - A/B testing capabilities
    - Fallback strategies
    - Performance comparison
    - Gradual migration support
    """
    
    def __init__(self, config: Dict[str, Any], batch_size: int = None, es_client=None):
        """Initialize the hybrid judge"""
        self.config = config
        self.batch_size = batch_size
        self.es_client = es_client
        self.logger = logging.getLogger(__name__)
        
        # Determine which implementation to use
        self.implementation = config.get('entity_matching', {}).get('llm', {}).get('implementation', 'function_calling')
        self.use_fallback = config.get('entity_matching', {}).get('llm', {}).get('use_fallback', True)
        self.enable_comparison = config.get('entity_matching', {}).get('llm', {}).get('enable_comparison', False)
        
        # Initialize both judges
        self.original_judge = None
        self.function_calling_judge = None
        
        # Initialize the primary judge
        if self.implementation == 'function_calling':
            self.logger.info("Using Function Calling Judge as primary implementation")
            self.function_calling_judge = FunctionCallingJudge(config, batch_size, es_client)
            
            if self.use_fallback:
                self.original_judge = EnhancedBatchMatchJudge(config, batch_size, es_client)
                self.function_calling_judge.set_fallback_judge(self.original_judge)
        else:
            self.logger.info("Using Original Enhanced Batch Match Judge as primary implementation")
            self.original_judge = EnhancedBatchMatchJudge(config, batch_size, es_client)
            
            if self.use_fallback:
                self.function_calling_judge = FunctionCallingJudge(config, batch_size, es_client)
        
        # Performance tracking
        self.comparison_results = []
        self.performance_stats = {
            'original_calls': 0,
            'function_calling_calls': 0,
            'fallback_calls': 0,
            'comparison_matches': 0,
            'comparison_mismatches': 0
        }
    
    async def judge_match(self, 
                         query_name: str, 
                         candidate_name: str, 
                         context: str = "",
                         use_cache: bool = True) -> Union[NameMatchResult, Any]:
        """
        Judge a single name match using the configured implementation
        
        Args:
            query_name: The query name to match
            candidate_name: The candidate name to match against
            context: Additional context information
            use_cache: Whether to use caching for performance
            
        Returns:
            Match result from the configured implementation
        """
        if self.implementation == 'function_calling':
            return await self._judge_with_function_calling(query_name, candidate_name, context, use_cache)
        else:
            return await self._judge_with_original(query_name, candidate_name, context, use_cache)
    
    async def judge_batch(self, 
                         pairs: List[Dict[str, Any]], 
                         use_cache: bool = True) -> Union[BatchMatchResult, Any]:
        """
        Judge multiple name matches using the configured implementation
        
        Args:
            pairs: List of name pairs to match
            use_cache: Whether to use caching for performance
            
        Returns:
            Batch results from the configured implementation
        """
        if self.implementation == 'function_calling':
            return await self._judge_batch_with_function_calling(pairs, use_cache)
        else:
            return await self._judge_batch_with_original(pairs, use_cache)
    
    async def _judge_with_function_calling(self, query_name: str, candidate_name: str, context: str, use_cache: bool):
        """Judge using function calling implementation"""
        if not self.function_calling_judge:
            raise ValueError("Function calling judge not initialized")
        
        self.performance_stats['function_calling_calls'] += 1
        return await self.function_calling_judge.judge_match(query_name, candidate_name, context, use_cache)
    
    async def _judge_with_original(self, query_name: str, candidate_name: str, context: str, use_cache: bool):
        """Judge using original implementation"""
        if not self.original_judge:
            raise ValueError("Original judge not initialized")
        
        self.performance_stats['original_calls'] += 1
        return await self.original_judge.judge_match(query_name, candidate_name, context, use_cache)
    
    async def _judge_batch_with_function_calling(self, pairs: List[Dict[str, Any]], use_cache: bool):
        """Judge batch using function calling implementation"""
        if not self.function_calling_judge:
            raise ValueError("Function calling judge not initialized")
        
        return await self.function_calling_judge.judge_batch(pairs, use_cache)
    
    async def _judge_batch_with_original(self, pairs: List[Dict[str, Any]], use_cache: bool):
        """Judge batch using original implementation"""
        if not self.original_judge:
            raise ValueError("Original judge not initialized")
        
        return await self.original_judge.judge_batch(pairs, use_cache)
    
    async def compare_implementations(self, 
                                    query_name: str, 
                                    candidate_name: str, 
                                    context: str = "") -> Dict[str, Any]:
        """
        Compare both implementations on the same input
        
        This is useful for A/B testing and validation
        """
        if not self.enable_comparison:
            raise ValueError("Comparison mode not enabled")
        
        if not self.original_judge or not self.function_calling_judge:
            raise ValueError("Both judges must be initialized for comparison")
        
        self.logger.info(f"Comparing implementations for: {query_name} vs {candidate_name}")
        
        # Get results from both implementations
        original_result = await self.original_judge.judge_match(query_name, candidate_name, context)
        function_calling_result = await self.function_calling_judge.judge_match(query_name, candidate_name, context)
        
        # Compare results
        comparison = {
            'query_name': query_name,
            'candidate_name': candidate_name,
            'context': context,
            'original_result': original_result,
            'function_calling_result': function_calling_result,
            'confidence_diff': abs(original_result.confidence - function_calling_result.confidence),
            'match_agreement': original_result.is_match == function_calling_result.is_match,
            'match_type_agreement': original_result.match_type == function_calling_result.match_type,
            'timestamp': time.time()
        }
        
        # Track performance
        self.comparison_results.append(comparison)
        self.performance_stats['comparison_matches'] += 1
        
        if not comparison['match_agreement']:
            self.performance_stats['comparison_mismatches'] += 1
            self.logger.warning(f"Match disagreement: {query_name} vs {candidate_name}")
        
        return comparison
    
    def switch_implementation(self, new_implementation: str):
        """
        Switch between implementations at runtime
        
        Args:
            new_implementation: 'function_calling' or 'original'
        """
        if new_implementation not in ['function_calling', 'original']:
            raise ValueError("Implementation must be 'function_calling' or 'original'")
        
        old_implementation = self.implementation
        self.implementation = new_implementation
        
        self.logger.info(f"Switched implementation from {old_implementation} to {new_implementation}")
        
        # Update config
        if 'entity_matching' not in self.config:
            self.config['entity_matching'] = {}
        if 'llm' not in self.config['entity_matching']:
            self.config['entity_matching']['llm'] = {}
        
        self.config['entity_matching']['llm']['implementation'] = new_implementation
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between implementations"""
        if not self.comparison_results:
            return {"message": "No comparison data available"}
        
        # Calculate statistics
        total_comparisons = len(self.comparison_results)
        match_agreements = sum(1 for r in self.comparison_results if r['match_agreement'])
        match_type_agreements = sum(1 for r in self.comparison_results if r['match_type_agreement'])
        
        avg_confidence_diff = sum(r['confidence_diff'] for r in self.comparison_results) / total_comparisons
        
        return {
            'total_comparisons': total_comparisons,
            'match_agreement_rate': match_agreements / total_comparisons,
            'match_type_agreement_rate': match_type_agreements / total_comparisons,
            'average_confidence_difference': avg_confidence_diff,
            'performance_stats': self.performance_stats,
            'recent_comparisons': self.comparison_results[-10:]  # Last 10 comparisons
        }
    
    def get_implementation_stats(self) -> Dict[str, Any]:
        """Get statistics for the current implementation"""
        if self.implementation == 'function_calling' and self.function_calling_judge:
            return self.function_calling_judge.get_performance_stats()
        elif self.original_judge:
            # Get stats from original judge if available
            return {
                'implementation': 'original',
                'cache_hits': getattr(self.original_judge, 'cache_hits', 0),
                'cache_misses': getattr(self.original_judge, 'cache_misses', 0),
                'total_calls': getattr(self.original_judge, 'total_calls', 0)
            }
        else:
            return {"message": "No implementation stats available"}
    
    def clear_comparison_data(self):
        """Clear comparison data"""
        self.comparison_results.clear()
        self.performance_stats = {
            'original_calls': 0,
            'function_calling_calls': 0,
            'fallback_calls': 0,
            'comparison_matches': 0,
            'comparison_mismatches': 0
        }
        self.logger.info("Comparison data cleared")
    
    def enable_comparison_mode(self):
        """Enable comparison mode (requires both judges)"""
        if not self.original_judge or not self.function_calling_judge:
            raise ValueError("Both judges must be initialized for comparison mode")
        
        self.enable_comparison = True
        self.logger.info("Comparison mode enabled")
    
    def disable_comparison_mode(self):
        """Disable comparison mode"""
        self.enable_comparison = False
        self.logger.info("Comparison mode disabled")
