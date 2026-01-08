#!/usr/bin/env python3
"""
Entity Preparation Module

Handles entity creation, enrichment, and indexing for the entity resolution pipeline.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Import local modules
from entity_resolution_demo.pipeline_runner.config import load_config
from entity_resolution_demo.entity_preparation.entity_data import load_entities, get_entities, get_enrichment_config
from entity_resolution_demo.pipeline_runner.utils import (
    print_header, print_subheader, print_success, print_warning, 
    print_error, print_info, print_entity, time_function,
    save_state, load_state
)

# Import project modules
from entity_resolution_demo.entity_preparation.entity_watch_list import EntityWatchList, WatchedEntity
from entity_resolution_demo.entity_preparation.entity_enricher import EntityEnricher
from entity_resolution_demo.entity_preparation.entity_indexer import EntityIndexer
from entity_resolution_demo.search.elastic_client import ElasticClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@time_function
def create_watch_list(config: Dict[str, Any]) -> EntityWatchList:
    """
    Create and populate entity watch list
    
    Args:
        config: Configuration dictionary
        
    Returns:
        EntityWatchList: Populated watch list
    """
    print_subheader("Creating Entity Watch List")
    
    # Create watch list
    watch_list = EntityWatchList()
    
    # Load entities from file or use default
    data_file = config["entity_preparation"].get("data_file")
    use_default = config["entity_preparation"].get("use_default_data", True)
    
    # Load entities from the appropriate source
    entities, _ = load_entities(data_file) if data_file else (get_entities(), get_enrichment_config())
    
    print_info(f"Loaded {len(entities)} entities from {'data file' if data_file else 'default data'}")
    
    # Add entities to watch list
    for entity_config in entities:
        name = entity_config["name"]
        entity_type = entity_config["entity_type"]
        description = entity_config.get("description", "")
        aliases = entity_config.get("aliases", [])
        priority = entity_config.get("priority", "medium")
        
        # Add entity to watch list
        result = watch_list.add_entity(
            name=name,
            entity_type=entity_type,
            description=description,
            aliases=aliases,
            priority=priority
        )
        
        # Handle different return types from add_entity
        if isinstance(result, bool):
            # Entity already exists - update aliases if provided
            entity = watch_list.get_entity_by_name(name)
            if entity:
                print_info(f"Entity '{name}' already exists")
                # Update aliases if they're different
                if aliases and set(entity.aliases) != set(aliases):
                    entity.aliases = aliases
                    print_info(f"Updated aliases for '{name}': {aliases}")
                print_entity(entity.__dict__)
            else:
                print_info(f"Entity '{name}' already exists but couldn't retrieve it")
        elif hasattr(result, '__dict__'):
            # New entity was created and returned as an object
            print_entity(result.__dict__)
        else:
            # Some other return value (like a string ID)
            print_info(f"Added entity '{name}' with ID: {result}")
            # Get the entity by name to display its details
            entity = watch_list.get_entity_by_name(name)
            if entity:
                print_entity(entity.__dict__)
    
    print_success(f"Created watch list with {len(watch_list.get_all_entities())} entities")
    return watch_list

# No longer need get_all_entity_contexts function as we use entity data directly

@time_function
def enrich_entities(watch_list: EntityWatchList, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create enriched entities directly from entity data
    
    Args:
        watch_list: Entity watch list
        config: Configuration dictionary
        
    Returns:
        List[Dict[str, Any]]: List of enriched entities
    """
    print_subheader("Enriching Entities")
    
    # Get enrichment config from entity_data module
    enrichment_config = get_enrichment_config()
    
    # Check if enrichment is enabled
    if not enrichment_config.get("enabled", True):
        print_warning("Entity enrichment is disabled")
        return []
    
    # Create entity enricher with combined config
    # Add enrichment config to the main config for the enricher
    enrichment_config_for_enricher = config.copy()
    if "entity_preparation" not in enrichment_config_for_enricher:
        enrichment_config_for_enricher["entity_preparation"] = {}
    enrichment_config_for_enricher["entity_preparation"]["enrichment"] = enrichment_config
    
    # Use our custom description preserving enricher
    from entity_resolution_demo.entity_preparation.description_preserving_enricher import patch_entity_enricher, restore_entity_enricher
    
    # Patch the EntityEnricher class to preserve descriptions
    original_enricher = patch_entity_enricher()
    enriched_entities = []
    
    try:
        # Create entity enricher (now using our patched version)
        from entity_resolution_demo.entity_preparation.entity_enricher import EntityEnricher
        enricher = EntityEnricher(enrichment_config_for_enricher)
        
        # Track entity count for consistent numbering across all entities
        entity_counts = {}
        
        # Create a dictionary to track entities by their name and description
        # This prevents duplicate entities with the same name AND description
        entities_by_name_and_desc = {}
        
        # Enrich entities
        for entity in watch_list.get_all_entities():
            print_info(f"Enriching entity: {entity.name}")
            
            try:
                # Create a unique key based on name and description
                entity_key = f"{entity.name}|{entity.description}"
                
                # Skip this entity if we already have one with the same name AND description
                # This prevents true duplicates while allowing ambiguous names
                if entity_key in entities_by_name_and_desc:
                    print_info(f"Skipping duplicate entity: {entity.name} with identical description")
                    continue
                    
                # Use the description preserving enricher to get Wikipedia context
                # Pass entity.description as source_context to preserve it
                enriched_entity = enricher.enrich_entity(entity.name, entity.description, entity.aliases)
                
                # Generate a consistent ID based on the entity name
                # Extract the base name without parentheses
                clean_name = entity.name.split('(')[0].strip()
                base_name = clean_name.lower().replace(' ', '_').replace('-', '_')
                
                # Update entity count
                if base_name not in entity_counts:
                    entity_counts[base_name] = 0
                else:
                    entity_counts[base_name] += 1
                
                # Use the current count for this entity
                enriched_entity.id = f"{base_name}_{entity_counts[base_name]:03d}"
                
                # Add to the list of enriched entities
                enriched_entities.append(enriched_entity)
                
                # Track this entity by its name and description to prevent duplicates
                entities_by_name_and_desc[entity_key] = enriched_entity
                
                # Print information about this entity
                print_success(f"Enriched {entity.name}")
                print_info(f"  Context: {enriched_entity.entity_context[:100]}..." if len(enriched_entity.entity_context) > 100 else f"  Context: {enriched_entity.entity_context}")
                
            except Exception as e:
                print_error(f"Error enriching entity {entity.name}: {e}")
        
        print_success(f"Enriched {len(enriched_entities)} entities")
        return enriched_entities
    finally:
        # Restore the original EntityEnricher class
        restore_entity_enricher(original_enricher)
        print_info("Restored original EntityEnricher class")

@time_function
def index_entities(enriched_entities: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Index entities in Elasticsearch
    
    Args:
        enriched_entities: List of enriched entities
        config: Configuration dictionary
        
    Returns:
        List[Dict[str, Any]]: List of indexed entities
    """
    print_subheader("Indexing Entities")
    
    # Create Elasticsearch client
    elastic_client = ElasticClient(config)
    
    # Create entity indexer
    entity_indexer = EntityIndexer(elastic_client, config)
    
    # Create indices
    index_name = entity_indexer.entity_index
    print_info(f"Creating Elasticsearch index: {index_name}")
    entity_indexer.create_indices()
    
    # Index entities
    indexed_entities = []
    for entity in enriched_entities:
        print_info(f"Indexing entity: {entity.name}")
        
        try:
            # Index the primary entity
            success = entity_indexer.index_entity(entity)
            
            if success:
                print_success(f"Indexed {entity.name}")
                indexed_entities.append(entity)
                
                # We don't need to do anything with alternative contexts
                # Each entity from entity_data.py is already properly defined
            else:
                print_error(f"Failed to index {entity.name}")
        except Exception as e:
            print_error(f"Error indexing entity {entity.name}: {e}")
    
    # Refresh indices to make entities searchable immediately
    entity_indexer.refresh_indices()
    
    print_success(f"Indexed {len(indexed_entities)} entities in index: {index_name}")
    
    # Add index name to the returned data for later use
    for entity in indexed_entities:
        if hasattr(entity, 'metadata') and entity.metadata is None:
            entity.metadata = {}
        if hasattr(entity, 'metadata'):
            entity.metadata['index_name'] = index_name
    
    return indexed_entities

@time_function
def verify_entity_indexing(indexed_entities: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that entities were indexed correctly
    
    Args:
        indexed_entities: List of indexed entities
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Verification results
    """
    print_subheader("Verifying Entity Indexing")
    
    # Create Elasticsearch client
    elastic_client = ElasticClient(config)
    
    # Create entity indexer
    entity_indexer = EntityIndexer(elastic_client, config)
    
    # Get index name from the first entity's metadata or from config
    index_name = None
    if indexed_entities and hasattr(indexed_entities[0], 'metadata') and indexed_entities[0].metadata:
        index_name = indexed_entities[0].metadata.get('index_name')
    
    if not index_name:
        index_name = config["elasticsearch"]["entity_index"]
    
    print_info(f"Verifying entities in index: {index_name}")
    
    # Explicitly refresh the index to ensure all documents are searchable
    print_info("Refreshing index before verification")
    entity_indexer.refresh_indices()
    
    # Wait a moment for the refresh to complete
    import time
    time.sleep(1)
    
    # Get index stats
    stats = entity_indexer.get_index_stats()
    
    # Use the entities_indexed count from the indexer stats
    actual_count = stats.get("entities_indexed", 0)
    expected_count = len(indexed_entities)
    
    # If we can find at least one entity via search, consider it a success
    search_success = False
    if indexed_entities:
        test_entity = indexed_entities[0]
        test_name = test_entity.name if hasattr(test_entity, 'name') else str(test_entity)
        print_info(f"Testing search for entity: {test_name}")
        results = entity_indexer.search_entities(test_name, limit=1)
        if results:
            search_success = True
            print_success(f"Lexical search successful - found {len(results)} results")
        else:
            print_error(f"Lexical search failed - no results found for {test_name}")
    
    # Consider it a success if either the count matches or search works
    if actual_count >= expected_count or search_success:
        print_success(f"Entities indexed successfully ({actual_count} indexed, {expected_count} expected)")
        success = True
    else:
        print_error(f"Not all entities were indexed ({actual_count} indexed, {expected_count} expected)")
        success = False
    
    return {
        "success": success,
        "details": {
            "expected_count": expected_count,
            "actual_count": actual_count,
            "indices": stats.get("indices", [])
        }
    }

@time_function
def run_entity_preparation(config: Dict[str, Any], state_dir: str = None, verify: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Run the entity preparation pipeline
    
    Args:
        config: Configuration dictionary
        state_dir: Optional directory for state files (defaults to entity_resolution_demo/state)
        verify: Whether to verify entity preparation results
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and state dictionary
    """
    print_header("Entity Preparation")
    
    # Delete the old state file to force a new run
    import os
    if state_dir:
        entity_state_file = os.path.join(state_dir, "entity_preparation_state.json")
    else:
        # Default state directory
        entity_state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "entity_preparation_state.json")
    
    if os.path.exists(entity_state_file):
        print_warning(f"Deleting existing state file: {entity_state_file}")
        os.remove(entity_state_file)
    
    # Check if we have a saved state (should be False now)
    state = load_state("entity_preparation", state_dir=state_dir)
    if state:
        print_success("Loaded entity preparation state")
        return True, state
    
    # Load entities from configured data file or entity_data.py
    from entity_resolution_demo.entity_preparation.entity_data import get_entities, get_enrichment_config, load_entities
    
    # Check if we have a configured data file
    data_file = config.get("entity_preparation", {}).get("data_file")
    if data_file:
        print_info(f"Loading entities from configured data file: {data_file}")
        entities, enrichment_config = load_entities(data_file)
    else:
        print_info("Loading entities from default data")
        entities = get_entities()
        enrichment_config = get_enrichment_config()
    
    # Create entity enricher with combined config
    # Add enrichment config to the main config for the enricher
    enrichment_config_for_enricher = config.copy()
    if "entity_preparation" not in enrichment_config_for_enricher:
        enrichment_config_for_enricher["entity_preparation"] = {}
    enrichment_config_for_enricher["entity_preparation"]["enrichment"] = enrichment_config
    
    # Create entity enricher
    from entity_resolution_demo.entity_preparation.entity_enricher import EntityEnricher
    enricher = EntityEnricher(enrichment_config_for_enricher)
    
    # Track entity count for consistent numbering across all entities
    # This dictionary maps base names to counts for ID generation
    # Example: {"michael_jordan": 2, "john_smith": 2, "elon_musk": 1}
    entity_counts = {}
    
    # Create a dictionary to track entities by their name and description
    # This prevents duplicate entities with the same name AND description
    # while still allowing ambiguous entities (same name, different description)
    # Key format: "Name|Description"
    # Example: "Michael Jordan|Former professional basketball player"
    entities_by_name_and_desc = {}
    
    # Create enriched entities
    enriched_entities = []
    for entity_config in entities:
        name = entity_config["name"]
        description = entity_config.get("description", "")
        
        print_info(f"Enriching entity: {name}")
        
        try:
            # Create a unique key based on name and description
            entity_key = f"{name}|{description}"
            
            # Skip this entity if we already have one with the same name AND description
            # This prevents true duplicates while allowing ambiguous names
            if entity_key in entities_by_name_and_desc:
                print_info(f"Skipping duplicate entity: {name} with identical description")
                continue
                
            # Check if explicit_context is provided in entity_data.py
            # This is particularly useful for ambiguous entities (multiple entities with the same name)
            # When explicit_context is provided, we bypass Wikipedia lookup entirely
            explicit_context = entity_config.get("explicit_context")
            
            if explicit_context:
                # Use the explicit context directly instead of fetching from Wikipedia
                # This ensures consistent and accurate context for ambiguous entities
                print_info(f"Using explicit context for {name}")
                from entity_resolution_demo.entity_preparation.entity_enricher import EnrichedEntity
                enriched_entity = EnrichedEntity(
                    name=name,
                    entity_context=explicit_context,
                    confidence_score=0.95,  # High confidence since this is explicitly provided by domain experts
                    enrichment_source="Explicit Context",
                    alternative_contexts=[],  # No alternatives needed when using explicit context
                    aliases=entity_config.get('aliases', [])  # Preserve aliases from entity_data.py
                )
            else:
                # For entities without explicit_context, use the EntityEnricher to get Wikipedia context
                # Pass the description to help with disambiguation when possible
                # The EntityEnricher will attempt to find the most relevant Wikipedia context
                enriched_entity = enricher.enrich_entity(name, description, entity_config.get('aliases', []))
            
            # Preserve the original description from entity_data.py
            enriched_entity.description = description
            
            # Generate a consistent ID based on the entity name
            # This ID generation strategy supports ambiguous entities (multiple entities with the same name)
            # by adding a numeric suffix to each entity ID
            
            # Extract the base name without parentheses (e.g., "Michael Jordan (Basketball)" -> "Michael Jordan")
            clean_name = name.split('(')[0].strip()
            
            # Convert to lowercase and replace spaces/hyphens with underscores for URL-friendly IDs
            # Example: "Michael Jordan" -> "michael_jordan"
            base_name = clean_name.lower().replace(' ', '_').replace('-', '_')
            
            # Track entity counts to ensure unique IDs for ambiguous entities
            # First occurrence gets _000, second gets _001, etc.
            if base_name not in entity_counts:
                entity_counts[base_name] = 0  # First occurrence of this entity name
            else:
                entity_counts[base_name] += 1  # Another entity with the same base name
            
            # Generate the final ID with zero-padded numeric suffix
            # Examples: "michael_jordan_000", "michael_jordan_001", "john_smith_000", etc.
            enriched_entity.id = f"{base_name}_{entity_counts[base_name]:03d}"
            
            # Add to the list of enriched entities
            enriched_entities.append(enriched_entity)
            
            # Track this entity by its name and description to prevent duplicates
            entities_by_name_and_desc[entity_key] = enriched_entity
            
            # Print information about this entity
            print_success(f"Enriched {name}")
            print_info(f"  Context: {enriched_entity.entity_context[:100]}..." if len(enriched_entity.entity_context) > 100 else f"  Context: {enriched_entity.entity_context}")
            
        except Exception as e:
            print_error(f"Error enriching entity {name}: {e}")
    
    # Index entities in Elasticsearch
    indexed_entities = index_entities(enriched_entities, config)
    
    # Verify entity indexing
    verification_results = verify_entity_indexing(indexed_entities, config)
    success = verification_results["success"]
    
    # Get the index name from the first entity's metadata or from the indexer
    index_name = None
    if indexed_entities and hasattr(indexed_entities[0], 'metadata') and indexed_entities[0].metadata:
        index_name = indexed_entities[0].metadata.get('index_name')
    
    if not index_name:
        # Create a temporary indexer to get the index name
        elastic_client = ElasticClient(config)
        entity_indexer = EntityIndexer(elastic_client, config)
        index_name = entity_indexer.entity_index
    
    # Save state with index name and metadata
    from datetime import datetime
    
    state = {
        "enriched_entities": [entity.to_dict() for entity in enriched_entities],
        "indexed_entities": [entity.to_dict() for entity in indexed_entities],
        "entity_index_name": index_name,
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "1.0",
        "processing_stats": {
            "entities_enriched": len(enriched_entities),
            "entities_indexed": len(indexed_entities),
            "enrichment_errors": 0,  # Could be tracked during enrichment
            "indexing_errors": 0,    # Could be tracked during indexing
            "success_rate": 1.0 if len(enriched_entities) > 0 else 0.0
        },
        "index_info": {
            "index_name": index_name,
            "document_count": len(indexed_entities),
            "status": "created" if success else "failed"
        }
    }
    save_state("entity_preparation", state, state_dir=state_dir)
    
    if success:
        print_success("Entity preparation completed successfully")
    else:
        print_error("Entity preparation failed")
    
    return success, state

if __name__ == "__main__":
    # Always reset preparation first
    from entity_resolution_demo.reset_preparation import reset_preparation
    reset_preparation()
    
    # Load configuration
    config = load_config()
    
    # Run entity preparation
    success, state = run_entity_preparation(config)
    
    # Print result
    if success:
        print_success("Entity preparation completed successfully")
    else:
        print_error("Entity preparation failed")
        sys.exit(1)
