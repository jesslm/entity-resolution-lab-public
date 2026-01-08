#!/usr/bin/env python3
"""
Elasticsearch-Powered Entity Matcher

Leverages Elasticsearch AI/semantic search capabilities for advanced entity matching:
- Direct keyword matching for exact matches
- Hybrid search combining lexical + semantic search using RRF
- Clear separation between potential match identification and LLM judgment
- No fallback explanations generated programmatically

(Previously known as elasticsearch_entity_matcher_CORRECTED.py)
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from pathlib import Path

# Import entity matching components
from entity_resolution_demo.search.elastic_client import ElasticClient
from entity_resolution_demo.entity_preparation.entity_watch_list import EntityWatchList, WatchedEntity
from entity_resolution_demo.article_processing.article_processor import Article, ProcessedArticle, ExtractedEntity
from entity_resolution_demo.entity_matching.entity_match import PotentialMatch


class ElasticsearchEntityMatcher:
    """
    Advanced entity matcher using Elasticsearch AI capabilities for direct keyword matching
    and hybrid search combining lexical and semantic approaches with RRF.
    
    This implementation ensures clear separation between potential match identification
    and LLM judgment, with no fallback explanations generated programmatically.
    """
    
    def __init__(self, 
                 watch_list: EntityWatchList,
                 elastic_client: ElasticClient,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the entity matcher
        
        Args:
            watch_list: Entity watch list
            elastic_client: Elasticsearch client
            config: Optional configuration dictionary
        """
        self.watch_list = watch_list
        self.elastic_client = elastic_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.match_threshold = self.config.get('match_threshold', 0.1)
        self.semantic_similarity_threshold = self.config.get('semantic_similarity_threshold', 0.1)
        self.hybrid_search_enabled = self.config.get('hybrid_search_enabled', True)
        self.max_hybrid_matches = self.config.get('max_hybrid_matches', 2)
        
        # Initialize statistics
        self.stats = {
            'entity_queries': 0,
            'direct_matches_found': 0,
            'hybrid_searches_performed': 0,
            'hybrid_matches_found': 0,
            'total_potential_matches': 0,
            'duplicates_removed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize caching
        self.match_cache = {}
        self.cache_enabled = self.config.get('cache_enabled', True)
        
        # Set up file logging for queries if enabled
        self.log_queries = self.config.get('log_queries', False)
        if self.log_queries:
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            self.query_log_file = log_dir / f"elasticsearch_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            self.logger.info(f"Logging queries to {self.query_log_file}")
        
        self.logger.info("ElasticsearchEntityMatcher initialized")
    
    def _create_cache_key(self, entity_name: str, entity_type: str, context: str = "") -> str:
        """Create a cache key for entity matching"""
        import hashlib
        # Use first 100 chars of context to balance caching with context sensitivity
        context_key = context[:100] if context else ""
        combined = f"{entity_name.lower()}|{entity_type.lower()}|{context_key}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def find_potential_matches(self, extracted_entity: ExtractedEntity, article: Article) -> List[PotentialMatch]:
        """
        Find potential matches for an extracted entity in the watch list.
        
        This method implements a three-step matching process:
        1. Direct exact name matching (match_type='exact')
        2. Alias matching (match_type='alias')
        3. Hybrid search combining lexical and semantic search (match_type='hybrid')
        
        Args:
            extracted_entity: The extracted entity to match
            article: The article containing the entity
            
        Returns:
            List of potential matches
        """
        # Check cache first if caching is enabled
        if self.cache_enabled:
            cache_key = self._create_cache_key(
                extracted_entity.name, 
                extracted_entity.entity_type, 
                extracted_entity.context
            )
            
            if cache_key in self.match_cache:
                self.stats['cache_hits'] += 1
                self.logger.debug(f"Cache hit for entity '{extracted_entity.name}'")
                # Return cached results, but update article-specific fields
                cached_matches = self.match_cache[cache_key]
                for match in cached_matches:
                    match.article_id = article.id
                    match.article_title = article.title
                    match.article_source = article.source
                return cached_matches
            else:
                self.stats['cache_misses'] += 1
        
        potential_matches = []
        
        try:
            # Step 1: Try exact name matching first
            exact_matches = self._find_exact_matches(extracted_entity)
            
            if exact_matches:
                self.logger.info(f"Found {len(exact_matches)} exact matches for '{extracted_entity.name}'")
                self.stats['exact_matches_found'] = self.stats.get('exact_matches_found', 0) + len(exact_matches)
                
                # Convert exact matches to potential matches
                for match_data in exact_matches:
                    entity_name = match_data.get('entity_name', '')
                    watched_entity = self.watch_list.find_entity_by_name_or_alias(entity_name)
                    
                    if watched_entity:
                        potential_match = PotentialMatch(
                            extracted_entity=extracted_entity,
                            watched_entity=watched_entity,
                            match_type='exact',
                            es_score=match_data.get('confidence', 1.0),
                            article_id=article.id,
                            article_title=article.title,
                            article_source=article.source
                        )
                        potential_matches.append(potential_match)
                        
                # If we found exact matches, return them immediately
                if potential_matches:
                    return potential_matches
            
            # Step 2: If no exact matches, try alias matching
            alias_matches = self._find_alias_matches(extracted_entity)
            
            if alias_matches:
                self.logger.info(f"Found {len(alias_matches)} alias matches for '{extracted_entity.name}'")
                self.stats['alias_matches_found'] = self.stats.get('alias_matches_found', 0) + len(alias_matches)
                
                # Convert alias matches to potential matches
                for match_data in alias_matches:
                    entity_name = match_data.get('entity_name', '')
                    watched_entity = self.watch_list.find_entity_by_name_or_alias(entity_name)
                    
                    if watched_entity:
                        potential_match = PotentialMatch(
                            extracted_entity=extracted_entity,
                            watched_entity=watched_entity,
                            match_type='alias',
                            es_score=match_data.get('confidence', 0.9),  # High confidence for alias matches
                            article_id=article.id,
                            article_title=article.title,
                            article_source=article.source
                        )
                        potential_matches.append(potential_match)
                        
                # If we found alias matches, return them immediately
                if potential_matches:
                    return potential_matches
            
            # Step 3: If no exact or alias matches, try hybrid search
            self.logger.info(f"No exact or alias matches found for '{extracted_entity.name}', trying hybrid search")
            hybrid_matches = self._find_hybrid_matches(extracted_entity)
            
            if hybrid_matches:
                self.logger.info(f"Found {len(hybrid_matches)} hybrid matches for '{extracted_entity.name}'")
                self.stats['hybrid_matches_found'] += len(hybrid_matches)
                
                # Convert hybrid matches to potential matches (limit to top N)
                for match_data in hybrid_matches[:self.max_hybrid_matches]:
                    entity_name = match_data.get('entity_name', '')

                    watched_entity = self.watch_list.find_entity_by_name_or_alias(entity_name)
                    
                    if watched_entity:

                        potential_match = PotentialMatch(
                            extracted_entity=extracted_entity,
                            watched_entity=watched_entity,
                            match_type='hybrid',
                            es_score=match_data.get('confidence', 0.0),
                            article_id=article.id,
                            article_title=article.title,
                            article_source=article.source
                        )
                        potential_matches.append(potential_match)
                    else:
                        pass  # No action needed for this case
            else:
                self.logger.info(f"No hybrid matches found for '{extracted_entity.name}'")
            
            # Deduplicate potential matches before returning
            if potential_matches:
                deduplicated_matches = self._deduplicate_potential_matches(potential_matches)
                duplicates_removed = len(potential_matches) - len(deduplicated_matches)
                if duplicates_removed > 0:
                    self.stats['duplicates_removed'] += duplicates_removed
                    self.logger.info(f"Removed {duplicates_removed} duplicate matches for '{extracted_entity.name}'")
                
                # Cache the results if caching is enabled
                if self.cache_enabled:
                    cache_key = self._create_cache_key(
                        extracted_entity.name, 
                        extracted_entity.entity_type, 
                        extracted_entity.context
                    )
                    self.match_cache[cache_key] = deduplicated_matches.copy()
                
                return deduplicated_matches
            
            # Cache empty results if caching is enabled
            if self.cache_enabled:
                cache_key = self._create_cache_key(
                    extracted_entity.name, 
                    extracted_entity.entity_type, 
                    extracted_entity.context
                )
                self.match_cache[cache_key] = []
            
            return potential_matches
            
        except Exception as e:
            self.logger.error(f"Error finding potential matches for '{extracted_entity.name}': {e}")
            return []
    
    def _find_exact_matches(self, extracted_entity: ExtractedEntity) -> List[Dict[str, Any]]:
        """
        Find exact matches for an entity name.
        
        This uses a simple term query on the name.keyword field for exact matching.
        """
        entity_name = extracted_entity.name
        index = self.watch_list.get_index_name()
        
        # Create a simple term query for exact matching on name.keyword field
        query = {
            "query": {
                "term": {
                    "name.keyword": entity_name
                }
            },
            "size": 1
        }
        
        # Execute the query
        self.stats['exact_searches_performed'] = self.stats.get('exact_searches_performed', 0) + 1
        results = self._perform_entity_search(index, query, "exact", entity_name)
        
        # Process results
        matches = []
        for hit in results:
            source = hit['_source']
            matches.append({
                'entity_name': source.get('name', ''),
                'entity_type': source.get('entity_type', ''),
                'confidence': 1.0,  # Exact matches have maximum confidence
                'match_type': 'exact',
                'es_score': hit['_score']
            })
        
        return matches
        
    def _find_alias_matches(self, extracted_entity: ExtractedEntity) -> List[Dict[str, Any]]:
        """
        Find alias matches for an entity name.
        
        This uses a match query on the aliases field for alias matching.
        """
        entity_name = extracted_entity.name
        index = self.watch_list.get_index_name()
        
        # Create a query for alias matching
        query = {
            "query": {
                "term": {
                    "aliases.keyword": entity_name
                }
            },
            "size": 1
        }
        
        # Execute the query
        self.stats['alias_searches_performed'] = self.stats.get('alias_searches_performed', 0) + 1
        results = self._perform_entity_search(index, query, "alias", entity_name)
        
        # Process results
        matches = []
        for hit in results:
            source = hit['_source']
            matches.append({
                'entity_name': source.get('name', ''),
                'entity_type': source.get('entity_type', ''),
                'confidence': 0.9,  # Alias matches have high but not maximum confidence
                'match_type': 'alias',
                'es_score': hit['_score']
            })
        
        return matches
    
    def _find_hybrid_matches(self, extracted_entity: ExtractedEntity) -> List[Dict[str, Any]]:
        """
        Find hybrid matches using the latest retriever syntax with RRF to combine lexical and semantic search.
        
        This uses the retriever tree approach with:
        1. Standard retriever for lexical search on name and context fields
        2. Standard retriever with semantic_text for semantic search
        """
        entity_name = extracted_entity.name
        context = extracted_entity.context or ""
        index = self.watch_list.get_index_name()
        
        # Create hybrid query using the retriever tree with RRF
        query = {
            "retriever": {
                "rrf": {
                    "retrievers": [
                        # Lexical component using standard retriever
                        {
                            "standard": {
                                "query": {
                                    "bool": {
                                        "should": [
                                            # Match on name field with high boost
                                            {
                                                "match": {
                                                    "name": {
                                                        "query": entity_name,
                                                        "boost": 3.0
                                                    }
                                                }
                                            },
                                            # Match on context field with lower boost
                                            {
                                                "match": {
                                                    "context": {
                                                        "query": entity_name,
                                                        "boost": 1.5
                                                    }
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        },
                        # Semantic component using standard retriever with semantic_text
                        {
                            "standard": {
                                "query": {
                                    "bool": {
                                        "should": [
                                            # Semantic match on name
                                            {
                                                "semantic": {
                                                    "field": "name_semantic",
                                                    "query": entity_name
                                                }
                                            },
                                            # Semantic match on context if available
                                            {
                                                "semantic": {
                                                    "field": "context_semantic",
                                                    "query": entity_name
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    ],
                    "rank_constant": 20,
                    "rank_window_size": 50
                }
            },
            "size": self.max_hybrid_matches
        }
        
        # Execute the query
        self.stats['hybrid_searches_performed'] += 1
        
        # Add a small delay to ensure index is ready
        import time
        time.sleep(0.1)
        
        # Refresh the index to ensure it's ready
        try:
            self.elastic_client.es.indices.refresh(index=index)

        except Exception as e:
            pass  # Index refresh failed, continue with search
        
        results = self._perform_entity_search(index, query, "hybrid", entity_name)




        # Process results
        matches = []
        for hit in results:
            source = hit['_source']
            matches.append({
                'entity_name': source.get('name', ''),
                'entity_type': source.get('entity_type', ''),
                'confidence': hit['_score'],
                'match_type': 'hybrid',
                'explanation': f"Hybrid search match for '{entity_name}'"
            })

        return matches
    
    def _perform_entity_search(self, index: str, query: Dict, search_type: str, entity_name: str = None) -> List[Dict]:
        """
        Perform entity search using the provided query.
        
        Args:
            index: Elasticsearch index name
            query: Elasticsearch query
            search_type: Type of search (direct or hybrid)
            entity_name: Optional entity name for logging
            
        Returns:
            List of search hits
        """
        try:
            # Log the query for debugging
            self.logger.info(f"Executing {search_type} search query against index {index}")






            # Log query to file if enabled
            if self.log_queries and entity_name:
                self._log_query_to_file(entity_name, index, query, search_type)
            
            # Execute the search
            response = self.elastic_client.es.search(index=index, body=query)

            # Check if we have results
            if response.get('hits', {}).get('hits'):
                self.logger.debug(f"Found {len(response['hits']['hits'])} results for {search_type} search")
            else:
                self.logger.debug(f"No results found for {search_type} search")


            # Log response to file if enabled
            if self.log_queries and entity_name:
                self._log_query_to_file(entity_name, index, query, search_type, response)
            
            # Return hits
            return response['hits']['hits']
            
        except Exception as e:
            self.logger.error(f"Error executing {search_type} search: {e}")
            return []
    
    def _log_query_to_file(self, entity_name: str, index: str, query: Dict, search_type: str, response: Dict = None) -> None:
        """Log query and response to file for detailed analysis"""
        try:
            with open(self.query_log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write(f"ENTITY: '{entity_name}'\n")
                f.write(f"SEARCH TYPE: {search_type}\n")
                f.write(f"INDEX: {index}\n")
                f.write(f"QUERY:\n{json.dumps(query, indent=2)}\n")
                
                if response:
                    f.write(f"\nRESPONSE SUMMARY:\n")
                    f.write(f"- Total hits: {response['hits']['total']['value']}\n")
                    f.write(f"- Max score: {response['hits']['max_score']}\n")
                    
                    # Log top 5 hits
                    f.write(f"\nTOP HITS:\n")
                    for i, hit in enumerate(response['hits']['hits'][:5]):
                        source = hit['_source']
                        f.write(f"  {i+1}. {source.get('name', 'Unknown')} (Score: {hit['_score']:.3f})\n")
                        
                f.write(f"{'='*80}\n")
        except Exception as e:
            self.logger.error(f"Error logging query to file: {e}")
    
    def _deduplicate_potential_matches(self, potential_matches: List[PotentialMatch]) -> List[PotentialMatch]:
        """
        Deduplicate potential matches to avoid sending identical matches to the LLM.
        
        A match is considered a duplicate if it has the same extracted entity name,
        context, and watched entity name as another match.
        
        Args:
            potential_matches: List of potential matches to deduplicate
            
        Returns:
            List of deduplicated potential matches
        """
        if not potential_matches:
            return []
            
        # Use a set to track unique combinations of extracted entity + context + watched entity
        seen_matches = set()
        deduplicated_matches = []
        
        for match in potential_matches:
            # Create a unique key for this match
            # We include the extracted entity name, context, and watched entity name
            extracted_name = match.extracted_entity.name
            extracted_context = match.extracted_entity.context or ""
            watched_name = match.watched_entity.name
            
            # Create a unique key - we use a tuple because it's hashable
            match_key = (extracted_name, extracted_context, watched_name)
            
            # Only add this match if we haven't seen this combination before
            if match_key not in seen_matches:
                seen_matches.add(match_key)
                deduplicated_matches.append(match)
                
        return deduplicated_matches
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about entity matching operations"""
        stats = self.stats.copy()
        stats['cache_size'] = len(self.match_cache)
        stats['cache_enabled'] = self.cache_enabled
        return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self.match_cache),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Clear the entity matching cache"""
        self.match_cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0
        self.logger.info("Entity matching cache cleared")