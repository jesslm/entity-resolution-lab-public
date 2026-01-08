#!/usr/bin/env python3
"""
Real-Time Entity Matcher

This module orchestrates the entity matching process with a clear separation between:
1. Potential match identification using Elasticsearch
2. LLM-based judgment for rich explanations

It ensures that LLM-generated explanations are properly preserved and never
generated programmatically as fallbacks.

(Previously known as real_time_entity_matcher_CORRECTED.py)
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import time
import json
from pathlib import Path

# Import entity matching components
from entity_resolution_demo.search.elastic_client import ElasticClient
from entity_resolution_demo.entity_preparation.entity_watch_list import EntityWatchList, WatchedEntity
from entity_resolution_demo.article_processing.article_processor import ProcessedArticle, ExtractedEntity
from entity_resolution_demo.entity_matching.elasticsearch_entity_matcher import ElasticsearchEntityMatcher
from entity_resolution_demo.entity_matching.entity_match import EntityMatch, MatchingResult, PotentialMatch
# Import EnhancedBatchMatchJudge inside methods to avoid circular imports


class RealTimeEntityMatcher:
    """
    Orchestrates the entity matching process with a clear separation between
    potential match identification and LLM judgment.
    
    This implementation ensures that LLM-generated explanations are
    properly preserved and never generated programmatically as fallbacks.
    """
    
    def __init__(self, 
                 watch_list: EntityWatchList,
                 elastic_client: ElasticClient,
                 config: Optional[Dict[str, Any]] = None,
                 batch_size: Optional[int] = None):
        """
        Initialize the real-time entity matcher
        
        Args:
            watch_list: Entity watch list
            elastic_client: Elasticsearch client
            config: Optional configuration dictionary
            batch_size: Optional batch size for LLM processing. If provided, overrides config.
        """
        self.watch_list = watch_list
        self.elastic_client = elastic_client
        self.config = config or {}
        self.batch_size = batch_size  # Store for passing to EnhancedBatchMatchJudge
        self.logger = logging.getLogger(__name__)
        
        # Initialize Elasticsearch entity matcher for potential match identification
        self.es_matcher = ElasticsearchEntityMatcher(
            watch_list=watch_list,
            elastic_client=elastic_client,
            config=config
        )
        
        # Configuration for LLM explanations
        self.use_llm_explanations = self.config.get('entity_matching', {}).get('matching', {}).get('use_llm_explanations', True)
        
        # Initialize LLM batch match judge if LLM explanations are enabled
        if self.use_llm_explanations:
            try:
                # Import here to avoid circular imports
                from entity_resolution_demo.entity_matching.enhanced_batch_match_judge import EnhancedBatchMatchJudge
                # Pass batch_size explicitly if provided, so it overrides config
                self.match_judge = EnhancedBatchMatchJudge(
                    config=config, 
                    batch_size=batch_size,  # Explicit batch_size overrides config
                    es_client=elastic_client.es
                )
                self.logger.info(f"Initialized EnhancedBatchMatchJudge for LLM-based explanations with batch_size={self.match_judge.batch_size}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EnhancedBatchMatchJudge: {e}")
                self.use_llm_explanations = False
        
        # Initialize statistics
        self.stats = {
            'articles_processed': 0,
            'entities_matched': 0,
            'potential_matches_found': 0,
            'llm_explanations_generated': 0,
            'high_confidence_matches': 0
        }
        
        self.logger.info("RealTimeEntityMatcher initialized")
    
    def match_article(self, processed_article: ProcessedArticle) -> MatchingResult:
        """
        Match entities from a processed article against the watch list
        
        This method implements the two-phase entity matching process:
        1. Find potential matches using Elasticsearch
        2. Judge potential matches using LLM (if enabled)
        
        Returns a MatchingResult with the matches found.
        """
        start_time = time.time()
        self.logger.info(f"Matching article {processed_article.article.id} against watch list")
        
        # Initialize result
        result = MatchingResult(
            article_id=processed_article.article.id,
            article_title=processed_article.article.title,
            total_entities_extracted=len(processed_article.extracted_entities),
            matches_found=[],
            processing_time=0.0
        )
        
        # Update statistics
        self.stats['articles_processed'] += 1
        
        # OPTIMIZATION: Collect all potential matches first, then batch process them
        all_potential_matches = []
        
        # Phase 1: Find all potential matches for all extracted entities
        for extracted_entity in processed_article.extracted_entities:
            # Find potential matches using Elasticsearch
            potential_matches = self.es_matcher.find_potential_matches(
                extracted_entity=extracted_entity,
                article=processed_article.article
            )
            
            # Update statistics
            self.stats['potential_matches_found'] += len(potential_matches)
            
            # Store matches for batch processing
            if potential_matches:
                all_potential_matches.extend(potential_matches)
        
        # Phase 2: Batch process all potential matches through LLM (if enabled)
        if self.use_llm_explanations and all_potential_matches:
            try:
                # Process all potential matches through LLM in batches
                # The EnhancedBatchMatchJudge will handle batching internally
                all_entity_matches = self.match_judge.judge_potential_matches(all_potential_matches)
                
                # Update statistics
                self.stats['llm_explanations_generated'] += len(all_entity_matches)
                
                # Add all matches to result
                result.matches_found.extend(all_entity_matches)
                
                # Count high confidence matches
                high_confidence_matches = [m for m in all_entity_matches if m.confidence >= 0.8]
                self.stats['high_confidence_matches'] += len(high_confidence_matches)
                
            except Exception as e:
                self.logger.error(f"Error judging potential matches with LLM: {e}")
                # Fall back to potential matches without LLM judgment
                entity_matches = [pm.to_entity_match() for pm in all_potential_matches]
                result.matches_found.extend(entity_matches)
        else:
            # Convert potential matches to entity matches without LLM judgment
            entity_matches = [pm.to_entity_match() for pm in all_potential_matches]
            result.matches_found.extend(entity_matches)
        
        # Update statistics
        self.stats['entities_matched'] += len(result.matches_found)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        
        self.logger.info(f"Matching complete for article {processed_article.article.id}: "
                        f"{len(result.matches_found)} matches found in {processing_time:.2f}s")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about entity matching operations"""
        # Combine stats from Elasticsearch matcher
        combined_stats = {
            **self.stats,
            'elasticsearch': self.es_matcher.get_stats()
        }
        return combined_stats


# For testing purposes
if __name__ == "__main__":
    import json
    from entity_resolution_demo.article_processing.article_processor import Article
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = {
        'entity_matching': {
            'matching': {
                'use_llm_explanations': True
            },
            'llm': {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'temperature': 0.1,
                'max_tokens': 1000,
                'enabled': True
            }
        },
        'hybrid_search_enabled': True,
        'match_threshold': 0.1,
        'semantic_similarity_threshold': 0.1
    }
    
    # Create Elasticsearch client
    elastic_client = ElasticClient(config)
    
    # Create watch list
    watch_list = EntityWatchList(elastic_client=elastic_client)
    
    # Create entity matcher
    matcher = RealTimeEntityMatcher(
        watch_list=watch_list,
        elastic_client=elastic_client,
        config=config
    )
    
    # Create test article
    article = Article(
        id="test_article",
        title="Test Article",
        content="Elon Musk announced that Tesla will be releasing a new model next year.",
        source="test",
        language="en"
    )
    
    # Create test extracted entities
    extracted_entities = [
        ExtractedEntity(
            name="Elon Musk",
            entity_type="PERSON",
            confidence=1.0,
            context="Elon Musk announced that Tesla will be releasing a new model next year.",
            position=0,
            extraction_method="test"
        ),
        ExtractedEntity(
            name="Tesla",
            entity_type="ORGANIZATION",
            confidence=1.0,
            context="Elon Musk announced that Tesla will be releasing a new model next year.",
            position=27,
            extraction_method="test"
        )
    ]
    
    # Create processed article
    processed_article = ProcessedArticle(
        article=article,
        extracted_entities=extracted_entities
    )
    
    # Match article
    result = matcher.match_article(processed_article)
    
    # Print results
    print(f"Found {len(result.matches_found)} matches:")
    for i, match in enumerate(result.matches_found):
        print(f"Match {i+1}: {match.extracted_entity.name} -> {match.watched_entity.name} "
              f"(Confidence: {match.confidence:.2f}, Type: {match.match_type})")
        if match.has_llm_explanation():
            print(f"  Reasoning: {match.reasoning[:100]}...")
        else:
            print("  No LLM explanation available")
