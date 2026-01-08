#!/usr/bin/env python3
"""
Entity Match Data Classes

Defines data structures for entity matching results to avoid circular imports.
This implementation ensures proper handling of LLM-generated explanations.

(Previously known as entity_match_CORRECTED.py)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import package components
from entity_resolution_demo.entity_preparation.entity_watch_list import WatchedEntity
from entity_resolution_demo.article_processing.article_processor import ExtractedEntity


@dataclass
class EntityMatch:
    """
    Represents a match between an extracted entity and a watched entity.
    
    This implementation ensures that LLM-generated explanations are properly
    preserved and never generated programmatically as fallbacks.
    """
    # Core match information
    extracted_entity: ExtractedEntity
    watched_entity: WatchedEntity
    confidence: float
    match_type: str  # 'exact', 'alias', 'semantic', 'hybrid', etc.
    
    # Article information
    article_id: str
    article_title: str
    article_source: str
    
    # Optional fields with default values
    match_timestamp: str = None
    is_match: bool = None  # Explicit boolean from LLM judgment
    
    # LLM-generated explanation fields (only populated when LLM is available)
    reasoning: str = None  # Rich LLM explanation
    explanation_full: str = None  # Concise summary for display
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    key_evidence: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Elasticsearch score (for reference)
    es_score: float = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.match_timestamp is None:
            self.match_timestamp = datetime.now().isoformat()
    
    def has_llm_explanation(self) -> bool:
        """Check if this match has an LLM-generated explanation"""
        return self.reasoning is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the match to a dictionary for serialization"""
        result = {
            "extracted_entity": self.extracted_entity.name,
            "extracted_entity_type": self.extracted_entity.entity_type,
            "extracted_entity_context": self.extracted_entity.context,
            "watched_entity": self.watched_entity.name,
            "watched_entity_type": self.watched_entity.entity_type,
            "watched_entity_priority": self.watched_entity.priority,
            "match_id": f"match_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}",
            "match_timestamp": self.match_timestamp,
            "match_type": self.match_type,
            "is_match": self.is_match if self.is_match is not None else self.confidence > 0.5,  # Use explicit value if available
            "article_id": self.article_id,
            "article_title": self.article_title,
            "article_source": self.article_source,
            "confidence": self.confidence,
        }
        
        # Only include LLM fields if they exist
        if self.has_llm_explanation():
            result.update({
                "reasoning": self.reasoning,
                "explanation_full": self.explanation_full,
                "confidence_factors": self.confidence_factors,
                "key_evidence": self.key_evidence,
                "risk_factors": self.risk_factors
            })
        
        if self.es_score is not None:
            result["es_score"] = self.es_score
            
        return result


@dataclass
class MatchingResult:
    """Results from matching a processed article against the watch list"""
    article_id: str
    article_title: str
    total_entities_extracted: int
    matches_found: List[EntityMatch] = field(default_factory=list)
    processing_time: float = 0.0
    
    def __post_init__(self):
        # Filter high priority matches
        self.high_priority_matches = [
            match for match in self.matches_found 
            if match.watched_entity.priority == 'high'
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization"""
        return {
            "article_id": self.article_id,
            "article_title": self.article_title,
            "total_entities_extracted": self.total_entities_extracted,
            "matches_found": [match.to_dict() for match in self.matches_found],
            "high_priority_matches": [match.to_dict() for match in self.high_priority_matches],
            "processing_time": self.processing_time
        }


@dataclass
class PotentialMatch:
    """
    Represents a potential match identified by Elasticsearch before LLM judgment.
    This is used in the first phase of the entity matching process.
    """
    extracted_entity: ExtractedEntity
    watched_entity: WatchedEntity
    match_type: str  # 'direct' or 'hybrid'
    es_score: float
    article_id: str
    article_title: str
    article_source: str
    
    def to_entity_match(self, confidence: float = None, llm_explanation: Dict[str, Any] = None) -> EntityMatch:
        """
        Convert a potential match to an entity match.
        If LLM explanation is provided, include it in the entity match.
        """
        # Use ES score as confidence if no confidence is provided
        if confidence is None:
            confidence = min(1.0, self.es_score)
            
        match = EntityMatch(
            extracted_entity=self.extracted_entity,
            watched_entity=self.watched_entity,
            confidence=confidence,
            match_type=self.match_type,
            article_id=self.article_id,
            article_title=self.article_title,
            article_source=self.article_source,
            es_score=self.es_score
        )
        
        # Add LLM explanation if available
        if llm_explanation:
            # Core fields
            match.reasoning = llm_explanation.get('reasoning')
            match.explanation_full = llm_explanation.get('explanation_full')
            match.confidence_factors = llm_explanation.get('confidence_factors', {})
            match.key_evidence = llm_explanation.get('key_evidence', [])
            match.risk_factors = llm_explanation.get('risk_factors', [])
            
            # Set is_match field directly from LLM judgment
            if 'is_match' in llm_explanation:
                match.is_match = llm_explanation['is_match']
            
            # Update match_type if provided by LLM
            if 'match_type' in llm_explanation and llm_explanation['match_type'] != 'unknown':
                match.match_type = llm_explanation['match_type']
            
            # Update confidence with LLM confidence if available
            if 'confidence' in llm_explanation:
                match.confidence = llm_explanation['confidence']
        
        return match
