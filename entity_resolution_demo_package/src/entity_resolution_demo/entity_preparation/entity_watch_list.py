"""
Entity Watch List Manager

Manages a list of named entities to monitor for real-time alerts.
Supports adding, removing, and querying entities with metadata.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class WatchedEntity:
    """Represents an entity being monitored"""
    name: str
    entity_type: str  # 'person', 'organization', 'location', etc.
    priority: str  # 'high', 'medium', 'low'
    id: Optional[str] = None  # Unique identifier for the entity
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    added_date: Optional[str] = None
    
    def __post_init__(self):
        # ID is now handled by EntityWatchList.add_entity
        if self.added_date is None:
            self.added_date = datetime.now().isoformat()
        if self.aliases is None:
            self.aliases = []
        if self.metadata is None:
            self.metadata = {}


class EntityWatchList:
    """Manages the list of entities to monitor for real-time alerts"""
    
    def __init__(self, watch_list_file: str = None):
        # No longer use a persistent watch list file - load from pipeline state instead
        self.entities: Dict[str, WatchedEntity] = {}  # Key is entity ID
        self.name_to_id: Dict[str, str] = {}  # For backward compatibility
        self.logger = logging.getLogger(__name__)
        self.index_name = None  # Will be set from pipeline state
    
    def add_entity(self, 
                   name: str, 
                   entity_type: str, 
                   priority: str = "medium",
                   description: Optional[str] = None,
                   aliases: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   entity_id: Optional[str] = None) -> str:
        """Add an entity to the watch list, returns the entity ID"""
        
        # Check for duplicate names
        existing_id = self.name_to_id.get(name)
        if existing_id:
            self.logger.warning(f"Entity with name '{name}' already exists with ID {existing_id}")
            return False
        
        # Generate a name-based ID if not provided
        if entity_id is None:
            # Create base name from entity name
            base_name = name.lower().replace(' ', '_').replace('-', '_')
            
            # Find existing entities with the same base name
            count = 1
            for existing_id in self.entities.keys():
                if existing_id.startswith(f"{base_name}_"):
                    try:
                        # Extract the number from the ID
                        num = int(existing_id.split('_')[-1])
                        count = max(count, num + 1)
                    except (ValueError, IndexError):
                        pass
            
            # Create the new ID with the next available number
            entity_id = f"{base_name}_{count:03d}"
        
        # Create entity with the ID
        entity = WatchedEntity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            priority=priority,
            description=description,
            aliases=aliases or [],
            metadata=metadata or {}
        )
        
        # Store by ID and update name mapping
        self.entities[entity.id] = entity
        self.name_to_id[name] = entity.id
        
        # Add aliases to name mapping too
        if entity.aliases:
            for alias in entity.aliases:
                if alias not in self.name_to_id:
                    self.name_to_id[alias] = entity.id
        
        # No longer save to file - entities are managed in memory and saved to pipeline state
        
        self.logger.info(f"Added entity '{name}' with ID {entity.id} to watch list (priority: {priority})")
        return entity.id
    
    def remove_entity_by_id(self, entity_id: str) -> bool:
        """Remove an entity from the watch list by ID"""
        if entity_id not in self.entities:
            self.logger.warning(f"Entity with ID '{entity_id}' not found in watch list")
            return False
        
        entity = self.entities[entity_id]
        
        # Remove from name mapping
        if entity.name in self.name_to_id:
            del self.name_to_id[entity.name]
        
        # Remove aliases from name mapping
        if entity.aliases:
            for alias in entity.aliases:
                if alias in self.name_to_id and self.name_to_id[alias] == entity_id:
                    del self.name_to_id[alias]
        
        # Remove from entities
        del self.entities[entity_id]
        # No longer save to file - entities are managed in memory and saved to pipeline state
        
        self.logger.info(f"Removed entity '{entity.name}' (ID: {entity_id}) from watch list")
        return True

    def remove_entity(self, name: str) -> bool:
        """Remove entity by name (backward compatibility)"""
        entity_id = self.name_to_id.get(name)
        if entity_id:
            return self.remove_entity_by_id(entity_id)
        self.logger.warning(f"Entity '{name}' not found in watch list")
        return False
    
    def get_entity_by_id(self, entity_id: str) -> Optional[WatchedEntity]:
        """Get entity by ID (primary lookup method)"""
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[WatchedEntity]:
        """Get entity by name (backward compatibility)"""
        entity_id = self.name_to_id.get(name)
        if entity_id:
            return self.entities.get(entity_id)
        return None
        
    def get_entity(self, name: str) -> Optional[WatchedEntity]:
        """Get a specific entity from the watch list (legacy method)"""
        # For backward compatibility
        return self.get_entity_by_name(name)
    
    def get_all_entities(self) -> List[WatchedEntity]:
        """Get all entities in the watch list"""
        return list(self.entities.values())
    
    def get_entities_by_priority(self, priority: str) -> List[WatchedEntity]:
        """Get entities filtered by priority level"""
        return [entity for entity in self.entities.values() if entity.priority == priority]
    
    def get_entities_by_type(self, entity_type: str) -> List[WatchedEntity]:
        """Get entities filtered by type"""
        return [entity for entity in self.entities.values() if entity.entity_type == entity_type]
    
    def get_all_names_and_aliases(self) -> Set[str]:
        """Get all entity names and their aliases for matching"""
        names = set()
        for entity in self.entities.values():
            names.add(entity.name)
            names.update(entity.aliases)
        return names
    
    def find_entity_by_name_or_alias(self, name: str) -> Optional[WatchedEntity]:
        """Find an entity by its name or any of its aliases"""
        # Check if name is in name_to_id mapping
        entity_id = self.name_to_id.get(name)
        if entity_id:
            return self.entities.get(entity_id)
        
        return None
    
    def find_entity_by_name(self, name: str) -> Optional[WatchedEntity]:
        """Find an entity by exact name match only (no aliases)"""
        return self.entities.get(name)
    
    def update_entity_by_id(self, entity_id: str, **kwargs) -> bool:
        """Update an existing entity's properties by ID"""
        if entity_id not in self.entities:
            self.logger.warning(f"Entity with ID '{entity_id}' not found in watch list")
            return False
        
        entity = self.entities[entity_id]
        old_name = entity.name
        old_aliases = entity.aliases.copy() if entity.aliases else []
        
        # Update allowed fields
        allowed_fields = ['name', 'entity_type', 'priority', 'description', 'aliases', 'metadata']
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(entity, field, value)
        
        # Update name mapping if name changed
        if 'name' in kwargs and kwargs['name'] != old_name:
            # Remove old name from mapping
            if old_name in self.name_to_id:
                del self.name_to_id[old_name]
            # Add new name to mapping
            self.name_to_id[entity.name] = entity_id
        
        # Update alias mappings if aliases changed
        if 'aliases' in kwargs:
            # Remove old aliases from mapping
            for alias in old_aliases:
                if alias in self.name_to_id and self.name_to_id[alias] == entity_id:
                    del self.name_to_id[alias]
            # Add new aliases to mapping
            for alias in entity.aliases:
                if alias not in self.name_to_id:
                    self.name_to_id[alias] = entity_id
        
        self.save_watch_list()
        self.logger.info(f"Updated entity '{entity.name}' (ID: {entity_id})")
        return True

    def update_entity(self, name: str, **kwargs) -> bool:
        """Update entity by name (backward compatibility)"""
        entity_id = self.name_to_id.get(name)
        if entity_id:
            return self.update_entity_by_id(entity_id, **kwargs)
        self.logger.warning(f"Entity '{name}' not found in watch list")
        return False
    
    def load_from_pipeline_state(self, state_data: Dict[str, Any]):
        """Load entities from pipeline state data"""
        self.entities = {}
        self.name_to_id = {}
        
        # Load enriched entities from state
        enriched_entities = state_data.get("enriched_entities", [])
        for entity_data in enriched_entities:
            # Convert enriched entity data to WatchedEntity format
            entity = WatchedEntity(
                name=entity_data["name"],
                entity_type=entity_data.get("entity_type", "person"),
                priority=entity_data.get("priority", "medium"),
                id=entity_data.get("id"),
                description=entity_data.get("entity_context", ""),
                aliases=entity_data.get("aliases", []),
                metadata=entity_data.get("metadata", {}),
                added_date=entity_data.get("added_date")
            )
            
            # Use the ID from the data or generate one
            if not entity.id:
                base_name = entity.name.lower().replace(' ', '_').replace('-', '_')
                entity.id = f"{base_name}_000"
            
            self.entities[entity.id] = entity
            self.name_to_id[entity.name] = entity.id
            
            # Add aliases to name mapping
            if entity.aliases:
                for alias in entity.aliases:
                    if alias not in self.name_to_id:
                        self.name_to_id[alias] = entity.id
        
        # Set index name from state
        self.index_name = state_data.get("entity_index_name")
        
        self.logger.info(f"Loaded {len(self.entities)} entities from pipeline state")
    
    # Removed save_watch_list method - entities are now managed through pipeline state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the watch list"""
        total = len(self.entities)
        
        # Count by priority
        priority_counts = {}
        for entity in self.entities.values():
            priority_counts[entity.priority] = priority_counts.get(entity.priority, 0) + 1
        
        # Count by type
        type_counts = {}
        for entity in self.entities.values():
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
        
        return {
            'total_entities': total,
            'priority_breakdown': priority_counts,
            'type_breakdown': type_counts,
            'total_aliases': sum(len(entity.aliases) for entity in self.entities.values())
        }
    
    def __len__(self) -> int:
        """Return the number of entities in the watch list"""
        return len(self.entities)
    
    def __contains__(self, item) -> bool:
        """Check if an entity name, alias, or ID is in the watch list"""
        # Check if item is an ID
        if item in self.entities:
            return True
            
        # Check if item is a name or alias
        if item in self.name_to_id:
            return True
            
        return False
    
    def get_index_name(self) -> str:
        """Return the index name for Elasticsearch contextual matching
        
        This method is used by ElasticsearchContextualMatcher to determine
        which index to search for entity matches.
        """
        # Return the index name set from pipeline state
        if self.index_name:
            return self.index_name
        
        # Default to 'entities' if not specified
        return "entities"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    watch_list = EntityWatchList()
    
    # Add some sample entities
    watch_list.add_entity(
        name="Vladimir Putin",
        entity_type="person",
        priority="high",
        description="Russian President",
        aliases=["Putin", "Владимир Путин", "V. Putin"],
        metadata={"country": "Russia", "position": "President"}
    )
    
    watch_list.add_entity(
        name="Xi Jinping",
        entity_type="person",
        priority="high",
        description="Chinese President",
        aliases=["Xi", "习近平", "President Xi"],
        metadata={"country": "China", "position": "President"}
    )
    
    watch_list.add_entity(
        name="Apple Inc",
        entity_type="organization",
        priority="medium",
        description="Technology company",
        aliases=["Apple", "AAPL"],
        metadata={"sector": "Technology", "country": "USA"}
    )
    
    # Display stats
    stats = watch_list.get_stats()
    print(f"\n📊 Watch List Statistics:")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Priority breakdown: {stats['priority_breakdown']}")
    print(f"Type breakdown: {stats['type_breakdown']}")
    print(f"Total aliases: {stats['total_aliases']}")
    
    # Test finding entities
    print(f"\n🔍 Testing entity lookup:")
    test_names = ["Putin", "Xi", "Apple", "习近平", "Unknown Person"]
    for name in test_names:
        entity = watch_list.find_entity_by_name_or_alias(name)
        if entity:
            print(f"✅ Found '{name}' -> {entity.name} ({entity.priority} priority)")
        else:
            print(f"❌ '{name}' not found")
