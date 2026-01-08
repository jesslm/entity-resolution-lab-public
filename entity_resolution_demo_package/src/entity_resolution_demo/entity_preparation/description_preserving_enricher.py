#!/usr/bin/env python3
"""
Description Preserving Enricher

This module provides a custom entity enricher that preserves descriptions from entity_data.py
and only uses Wikipedia for additional context, not for disambiguation.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import project modules
from entity_resolution_demo.entity_preparation.entity_enricher import EntityEnricher, EnrichedEntity
from entity_resolution_demo.pipeline_runner.utils import print_info, print_success, print_warning, print_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DescriptionPreservingEnricher(EntityEnricher):
    """
    Custom entity enricher that preserves descriptions from entity_data.py
    and only uses Wikipedia for additional context, not for disambiguation.
    """
    
    def enrich_entity(self, name: str, source_context: str = None, aliases: List[str] = None) -> EnrichedEntity:
        """
        Enrich an entity while preserving its original description and aliases.
        
        Args:
            name: The entity name to enrich
            source_context: Description from entity_data.py (must be preserved)
            aliases: List of aliases for the entity (must be preserved)
            
        Returns:
            EnrichedEntity with name, original description, aliases, and enriched context
        """
        self.logger.info(f" Enriching entity: {name}")
        
        # Create enriched entity with the original description and aliases
        enriched = EnrichedEntity(
            name=name,
            entity_context=source_context if source_context else "",
            confidence_score=1.0,  # Highest confidence since this is from entity_data.py
            enrichment_source="Entity Data",
            alternative_contexts=[],  # No alternative contexts needed
            aliases=aliases or []  # Preserve aliases
        )
        
        # Set the description field explicitly to preserve it
        enriched.description = source_context
        
        # Log the enrichment
        self.logger.info(f" Preserved original description for {name}")
        if aliases:
            self.logger.info(f" Preserved {len(aliases)} aliases: {aliases}")
        
        return enriched

def patch_entity_enricher():
    """
    Patch the EntityEnricher class to preserve descriptions from entity_data.py.
    """
    # Save the original class
    original_enricher = EntityEnricher
    
    # Replace with our custom version in multiple import paths
    import entity_resolution_demo.entity_preparation.entity_enricher
    entity_resolution_demo.entity_preparation.entity_enricher.EntityEnricher = DescriptionPreservingEnricher
    
    # Also patch the src import path
    import sys
    if 'src.entity_resolution_demo.entity_preparation.entity_enricher' in sys.modules:
        src_module = sys.modules['src.entity_resolution_demo.entity_preparation.entity_enricher']
        src_module.EntityEnricher = DescriptionPreservingEnricher
    
    return original_enricher

def restore_entity_enricher(original_enricher):
    """
    Restore the original EntityEnricher class.
    """
    import entity_resolution_demo.entity_preparation.entity_enricher
    entity_resolution_demo.entity_preparation.entity_enricher.EntityEnricher = original_enricher
    
    # Also restore the src import path
    import sys
    if 'src.entity_resolution_demo.entity_preparation.entity_enricher' in sys.modules:
        src_module = sys.modules['src.entity_resolution_demo.entity_preparation.entity_enricher']
        src_module.EntityEnricher = original_enricher
