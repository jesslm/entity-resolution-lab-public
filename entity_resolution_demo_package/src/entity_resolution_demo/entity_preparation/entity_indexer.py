#!/usr/bin/env python3
"""
Entity Indexer for Elasticsearch

Handles the indexing of enriched entities into Elasticsearch indices,
optimized for semantic search and entity resolution.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json

# Import package components
from entity_resolution_demo.search.elastic_client import ElasticClient
from entity_resolution_demo.entity_preparation.entity_enricher import EnrichedEntity
from entity_resolution_demo.entity_preparation.entity_watch_list import WatchedEntity


class EntityIndexer:
    """
    Handles indexing of enriched entities into Elasticsearch
    
    This component is responsible for:
    1. Creating and managing Elasticsearch indices for entities
    2. Indexing enriched entities with proper mappings for semantic search
    3. Providing utilities for index management and entity retrieval
    """
    
    def __init__(self, 
                 elastic_client: ElasticClient,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EntityIndexer
        
        Args:
            elastic_client: ElasticClient instance for Elasticsearch operations
            config: Optional configuration dictionary
        """
        self.elastic_client = elastic_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Index name - get from elasticsearch section of config
        elasticsearch_config = self.config.get('elasticsearch', {})
        self.entity_index = elasticsearch_config.get('entity_index', 'entity_index')
        
        # Statistics
        self.stats = {
            'entities_indexed': 0,
            'indexing_errors': 0
        }
        
        self.logger.info("EntityIndexer initialized")
    
    def create_indices(self):
        """Create or update Elasticsearch indices with proper mappings"""
        try:
            # Check if index exists and delete it if requested
            if self.config.get('recreate_indices', False) and self.elastic_client.es.indices.exists(index=self.entity_index):
                self.elastic_client.es.indices.delete(index=self.entity_index)
                self.logger.info(f"Deleted existing index: {self.entity_index}")
            
            # Create index with mappings matching the comprehensive tests
            # Removed settings for number_of_shards and number_of_replicas for serverless compatibility
            mappings = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "name_lower": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "context": {"type": "text"},
                        "aliases": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "name_semantic": {
                            "type": "semantic_text",
                            "inference_id": ".multilingual-e5-small-elasticsearch",
                            "model_settings": {
                                "task_type": "text_embedding",
                                "dimensions": 384,
                                "similarity": "cosine",
                                "element_type": "float"
                            }
                        },
                        "context_semantic": {
                            "type": "semantic_text",
                            "inference_id": ".multilingual-e5-small-elasticsearch",
                            "model_settings": {
                                "task_type": "text_embedding",
                                "dimensions": 384,
                                "similarity": "cosine",
                                "element_type": "float"
                            }
                        }
                    }
                }
                # Settings removed for serverless compatibility
            }
            
            # Create index if it doesn't exist
            if not self.elastic_client.es.indices.exists(index=self.entity_index):
                self.elastic_client.es.indices.create(index=self.entity_index, body=mappings)
                self.logger.info(f"Created index: {self.entity_index}")
            else:
                self.logger.info(f"Index already exists: {self.entity_index}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up Elasticsearch indices: {e}")
            self.stats['indexing_errors'] += 1
            return False
    
    def _create_index_if_not_exists(self, index_name: str, mapping: Dict[str, Any]):
        """Create index if it doesn't exist"""
        try:
            if not self.elastic_client.es.indices.exists(index=index_name):
                self.elastic_client.es.indices.create(
                    index=index_name,
                    body=mapping
                )
                self.logger.info(f"Created Elasticsearch index: {index_name}")
            else:
                self.logger.info(f"Elasticsearch index already exists: {index_name}")
                
        except Exception as e:
            self.logger.error(f"Error creating index {index_name}: {e}")
            raise
    
    def index_entity(self, entity: Any) -> bool:
        """
        Index a single entity into Elasticsearch
        
        Args:
            entity: Any object with at least a 'name' attribute. Can be EnrichedEntity, 
                   WatchedEntity, or any dict-like or object with required attributes.
            
        Returns:
            bool: True if indexing was successful
        """
        try:
            # Get entity name (required) - clean the name by removing disambiguation information
            if hasattr(entity, 'name'):
                raw_name = entity.name
                name = re.sub(r'\s*\([^)]*\)', '', raw_name).strip()
            elif isinstance(entity, dict) and 'name' in entity:
                raw_name = entity['name']
                name = re.sub(r'\s*\([^)]*\)', '', raw_name).strip()
            else:
                raise ValueError("Entity must have a 'name' attribute or key")
                
            # Use entity ID if available, otherwise generate from name
            if hasattr(entity, 'id') and entity.id:
                entity_id = entity.id
            elif isinstance(entity, dict) and 'id' in entity and entity['id']:
                entity_id = entity['id']
            else:
                # Fallback to name-based ID for backward compatibility
                entity_id = name.replace(' ', '_')
            name_lower = name.lower()
            
            # Get context from entity_context field (preferred) or fallback to description
            context = ""
            if hasattr(entity, 'entity_context'):
                context = entity.entity_context
            elif hasattr(entity, 'description'):
                context = entity.description
            elif isinstance(entity, dict):
                context = entity.get('entity_context', entity.get('description', ""))
            
            # Handle aliases/alternative names (optional)
            alternative_names = []
            if hasattr(entity, 'aliases') and entity.aliases:
                alternative_names.extend(entity.aliases)
            elif isinstance(entity, dict) and 'aliases' in entity and entity['aliases']:
                alternative_names.extend(entity['aliases'])
            
            # Create semantic text fields
            name_semantic = name
            context_semantic = self._create_contextual_description(entity)
            
            # Get entity_type (optional)
            entity_type = ""
            if hasattr(entity, 'entity_type'):
                entity_type = entity.entity_type
            elif isinstance(entity, dict) and 'entity_type' in entity:
                entity_type = entity['entity_type']
            
            # Get description (separate from context)
            description = ""
            if hasattr(entity, 'description'):
                description = entity.description
            elif isinstance(entity, dict) and 'description' in entity:
                description = entity['description']
            
            # Get enrichment source (for enriched entities)
            enrichment_source = ""
            if hasattr(entity, 'enrichment_source'):
                enrichment_source = entity.enrichment_source
            elif isinstance(entity, dict) and 'enrichment_source' in entity:
                enrichment_source = entity['enrichment_source']
            
            # Create the document with all fields
            entity_doc = {
                "id": entity_id,
                "name": name,
                "name_lower": name_lower,
                "entity_type": entity_type,
                "description": description,
                "context": context,
                "aliases": alternative_names,  # Renamed for clarity
                "name_semantic": name_semantic,
                "context_semantic": context_semantic,
                "enrichment_source": enrichment_source
            }
            
            # Index the entity
            response = self.elastic_client.es.index(
                index=self.entity_index,
                id=entity_id,
                body=entity_doc
            )
            
            self.stats['entities_indexed'] += 1
            self.logger.info(f"Indexed entity: {name}")
            
            return True
            
        except Exception as e:
            self.stats['indexing_errors'] += 1
            entity_name = getattr(entity, 'name', str(entity))
            self.logger.error(f"Error indexing entity {entity_name}: {e}")
            return False
    
    # _index_entity_aliases method removed as alternative names are now included directly in the entity document
    
    def _create_contextual_description(self, entity: Any) -> str:
        """Create simple contextual description for semantic matching
        
        Works with any entity-like object that has name and description attributes,
        or a dictionary with those keys.
        
        Args:
            entity: Any entity-like object or dictionary
            
        Returns:
            str: A contextual description combining name and description
        """
        # Get entity name
        name = ""
        if hasattr(entity, 'name'):
            name = entity.name
        elif isinstance(entity, dict) and 'name' in entity:
            name = entity['name']
        
        # Get description/context
        # Prioritize entity_context (enriched content) over description
        description = ""
        if hasattr(entity, 'entity_context'):
            description = entity.entity_context
        elif hasattr(entity, 'description'):
            description = entity.description
        elif isinstance(entity, dict):
            description = entity.get('entity_context', entity.get('description', ""))
        
        # Combine name and description
        return f"{name} {description}".strip()
    
    def index_entities(self, entities: List[Any]) -> Dict[str, Any]:
        """
        Batch index multiple entities using the simplified approach
        
        Args:
            entities: List of entity objects to index. Can be EnrichedEntity, WatchedEntity,
                     or any objects/dictionaries with at least a 'name' attribute/key.
            
        Returns:
            Dict with indexing statistics
        """
        results = {
            "total": len(entities),
            "success": 0,
            "errors": 0
        }
        
        try:
            # Prepare bulk indexing actions
            actions = []
            
            for entity in entities:
                # Get entity name (required)
                if hasattr(entity, 'name'):
                    name = entity.name
                elif isinstance(entity, dict) and 'name' in entity:
                    name = entity['name']
                else:
                    self.logger.warning(f"Skipping entity without name: {entity}")
                    results["errors"] += 1
                    continue
                    
                entity_id = name.replace(' ', '_')
                name_lower = name.lower()
                
                # Get context/description (optional)
                # Prioritize entity_context (enriched content) over description
                context = ""
                if hasattr(entity, 'entity_context'):
                    context = entity.entity_context
                elif hasattr(entity, 'description'):
                    context = entity.description
                elif isinstance(entity, dict):
                    context = entity.get('entity_context', entity.get('description', ""))
                
                # Handle aliases/alternative names (optional)
                alternative_names = []
                if hasattr(entity, 'aliases') and entity.aliases:
                    alternative_names.extend(entity.aliases)
                elif isinstance(entity, dict) and 'aliases' in entity and entity['aliases']:
                    alternative_names.extend(entity['aliases'])
                
                # Create semantic text fields
                name_semantic = name
                context_semantic = self._create_contextual_description(entity)
                
                # Create the document
                entity_doc = {
                    "id": entity_id,
                    "name": name,
                    "name_lower": name_lower,
                    "context": context,
                    "alternative_names": alternative_names,
                    "name_semantic": name_semantic,
                    "context_semantic": context_semantic
                }
                
                # Add to bulk actions
                action = {
                    "_index": self.entity_index,
                    "_id": entity_id,
                    "_source": entity_doc
                }
                actions.append(action)
            
            # Execute bulk indexing
            if actions:
                from elasticsearch.helpers import bulk
                success, errors = bulk(self.elastic_client.es, actions, stats_only=True)
                results["success"] = success
                results["errors"] = errors
                self.stats['entities_indexed'] += success
                self.stats['indexing_errors'] += errors
                self.logger.info(f"Batch indexed {success} entities with {errors} errors")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch indexing: {e}")
            results["errors"] = len(entities)
            self.stats['indexing_errors'] += len(entities)
            return results
    
    def refresh_indices(self):
        """Force a refresh of the index"""
        try:
            self.elastic_client.es.indices.refresh(index=self.entity_index)
            self.logger.info("Index refreshed")
            return True
        except Exception as e:
            self.logger.error(f"Error refreshing index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the entity index"""
        try:
            entity_stats = self.elastic_client.es.count(index=self.entity_index)
            
            return {
                "entity_count": entity_stats["count"],
                "indices": [self.entity_index]
            }
        except Exception as e:
            self.logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    def delete_entity(self, entity_id_or_name: str) -> bool:
        """
        Remove an entity from the index
        
        Args:
            entity_id_or_name: ID or name of the entity to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # If this looks like a UUID, use it directly as the ID
            if '-' in entity_id_or_name and len(entity_id_or_name) > 30:
                entity_id = entity_id_or_name
            else:
                # Otherwise convert name to ID format
                entity_id = entity_id_or_name.replace(' ', '_')
            
            # Delete from entity index
            self.elastic_client.es.delete(
                index=self.entity_index,
                id=entity_id
            )
            
            self.logger.info(f"Deleted entity with ID: {entity_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting entity {entity_id_or_name}: {e}")
            return False
    
    def search_entities(self, 
                       query: str, 
                       limit: int = 10,
                       use_semantic: bool = False) -> List[Dict[str, Any]]:
        """
        Search for entities using lexical matching
        
        Args:
            query: Search query
            limit: Maximum number of results
            use_semantic: Parameter kept for backwards compatibility but ignored
            
        Returns:
            List of matching entity documents
        """
        try:
            # Simple keyword search using name and context fields
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["name", "name_lower", "context", "alternative_names"]
                    }
                },
                "size": limit
            }
            
            response = self.elastic_client.es.search(
                index=self.entity_index,
                body=search_body
            )
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
            
        except Exception as e:
            self.logger.error(f"Error searching entities: {e}")
            return []
    
    def refresh_indices(self):
        """Refresh indices to make entities searchable immediately"""
        try:
            self.elastic_client.es.indices.refresh(index=self.entity_index)
            self.logger.info(f"Refreshed index: {self.entity_index}")
            return True
        except Exception as e:
            self.logger.error(f"Error refreshing index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics"""
        try:
            # Get actual document count from the index
            count_response = self.elastic_client.es.count(index=self.entity_index)
            self.logger.info(f"Count response: {count_response}")
            
            # Extract count from response
            doc_count = count_response.get('count', 0)
            self.logger.info(f"Document count: {doc_count}")
            self.stats['doc_count'] = doc_count
            
            # Use the entities_indexed count if the doc_count is 0 but we know we indexed entities
            if doc_count == 0 and self.stats['entities_indexed'] > 0:
                self.logger.warning(f"Count query returned 0 but {self.stats['entities_indexed']} entities were indexed. Using entities_indexed count.")
                self.stats['doc_count'] = self.stats['entities_indexed']
            
            # Get index details
            self.stats['indices'] = [self.entity_index]
            
            return self.stats.copy()
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return self.stats.copy()
    
    def reset_stats(self):
        """Reset indexer statistics"""
        self.stats = {
            'entities_indexed': 0,
            'indexing_errors': 0,
            'doc_count': 0,
            'indices': []
        }
        self.logger.info("Indexer statistics reset")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    from search.elastic_client import ElasticClient
    from models.entity_enricher import EntityEnricher
    from alerts.entity_watch_list import EntityWatchList
    
    # Load configuration
    import yaml
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create components
    es_client = ElasticClient(config)
    watch_list = EntityWatchList()
    enricher = EntityEnricher()
    indexer = EntityIndexer(es_client)
    
    # Test with a sample entity
    test_entity = watch_list.create_entity(
        name="Elon Musk",
        entity_type="person",
        description="CEO of Tesla and SpaceX",
        priority="HIGH",
        aliases=["E. Musk", "Elon Reeve Musk"]
    )
    
    # Enrich and index the entity
    enriched_entity = enricher.enrich_entity(test_entity)
    indexer.create_indices()
    success = indexer.index_entity(enriched_entity)
    
    print(f"Entity indexing: {'✅ Success' if success else '❌ Failed'}")
    
    # Test semantic search
    semantic_results = indexer.search_entities("Tesla CEO", use_semantic=True)
    print(f"Semantic search results: {len(semantic_results)} entities found")
    
    # Test keyword search
    keyword_results = indexer.search_entities("Elon", use_semantic=False)
    print(f"Keyword search results: {len(keyword_results)} entities found")
    
    # Get statistics
    stats = indexer.get_index_stats()
    print(f"Index stats: {stats}")
    
    # Refresh index to ensure all changes are visible
    indexer.refresh_indices()
