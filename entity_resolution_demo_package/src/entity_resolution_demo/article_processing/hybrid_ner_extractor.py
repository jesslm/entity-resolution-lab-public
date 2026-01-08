#!/usr/bin/env python3
"""
Hybrid NER Extractor: Elasticsearch NER + Pattern-based Title Extraction

Combines the power of Elasticsearch's native NER for proper nouns (names, organizations)
with intelligent pattern matching for job titles, roles, and positions that traditional
NER models miss.
"""

import re
import logging
from typing import List, Dict, Set, Any, Optional
from dataclasses import dataclass

try:
    from .elasticsearch_ner_extractor import ElasticsearchNERExtractor
    from ..search.elastic_client import ElasticClient
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.elasticsearch_ner_extractor import ElasticsearchNERExtractor
    from search.elastic_client import ElasticClient


@dataclass
class TitleEntity:
    """Represents a job title/role entity extracted by pattern matching"""
    title: str
    category: str  # EXECUTIVE, POLITICAL, PROFESSIONAL, etc.
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    extraction_method: str = "pattern_based_title"


class HybridNERExtractor:
    """
    Hybrid NER extractor that combines Elasticsearch NER with pattern-based title extraction.
    
    This approach gives us the best of both worlds:
    - Elasticsearch NER for proper nouns (names, organizations, locations)
    - Pattern matching for job titles, roles, and positions
    """
    
    # Comprehensive job title patterns organized by category
    # NOTE: COMPOUND_EXECUTIVE must come first to match compound entities before simple titles
    TITLE_PATTERNS = {
        # NEW: Compound entity patterns (Organization + Title) - Simplified approach
        'COMPOUND_EXECUTIVE': [
            # Forward patterns: Company + Title (e.g., "Tesla CEO")
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(CEO)\b',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(CTO)\b',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(CFO)\b',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(COO)\b',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(President)\b',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(Founder)\b',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(Chairman)\b',
            
            # Reverse patterns: Title + of + Company (e.g., "CEO of Tesla")
            r'\b(CEO)\s+of\s+([A-Z][a-zA-Z]+)(?=\s+[A-Z][a-z]|\s*$)',
            r'\b(CTO)\s+of\s+([A-Z][a-zA-Z]+)(?=\s+[A-Z][a-z]|\s*$)',
            r'\b(CFO)\s+of\s+([A-Z][a-zA-Z]+)(?=\s+[A-Z][a-z]|\s*$)',
            r'\b(President)\s+of\s+([A-Z][a-zA-Z]+)(?=\s+[A-Z][a-z]|\s*$)',
            r'\b(Founder)\s+of\s+([A-Z][a-zA-Z]+)(?=\s+[A-Z][a-z]|\s*$)',
        ],
        
        'EXECUTIVE': [
            # C-Suite
            r'\b(?:chief\s+)?(?:executive|operating|financial|technology|information|marketing|data|security)\s+officer\b',
            r'\b(?:ceo|coo|cfo|cto|cio|cmo|cdo|cso)\b',
            r'\bchief\s+(?:executive|operating|financial|technology|information|marketing|data|security)\b',
            
            # Executive roles
            r'\b(?:executive\s+)?(?:vice\s+)?president\b',
            r'\b(?:senior\s+)?(?:vice\s+)?president\s+of\s+\w+\b',
            r'\bdirector\s+(?:of\s+\w+|general)\b',
            r'\bmanaging\s+director\b',
            r'\bexecutive\s+director\b',
            r'\bsenior\s+director\b',
        ],
        
        'POLITICAL': [
            # Government positions
            r'\bpresident\b(?!\s+of\s+(?:the\s+)?(?:company|corporation|board))',  # Exclude corporate presidents
            r'\bvice\s+president\b(?!\s+of\s+(?:the\s+)?(?:company|corporation|board))',
            r'\bprime\s+minister\b',
            r'\bchancellor\b',
            r'\bsecretary\s+of\s+\w+\b',
            r'\bminister\s+(?:of\s+\w+|for\s+\w+)?\b',
            r'\bsenator\b',
            r'\bcongressman|congresswoman\b',
            r'\brepresentative\b',
            r'\bgovernor\b',
            r'\bmayor\b',
            r'\bambassador\b',
        ],
        
        'PROFESSIONAL': [
            # Professional titles
            r'\b(?:senior\s+)?(?:software\s+)?(?:engineer|developer|architect|analyst|consultant)\b',
            r'\b(?:project\s+|program\s+)?manager\b',
            r'\b(?:team\s+)?lead(?:er)?\b',
            r'\bsupervisor\b',
            r'\bcoordinator\b',
            r'\bspecialist\b',
            r'\bexpert\b',
            r'\bprofessor\b',
            r'\bdoctor\b',
            r'\bdr\.\b',
            r'\blawyer\b',
            r'\battorney\b',
        ],
        
        'ACADEMIC': [
            # Academic positions
            r'\bprofessor\b',
            r'\bassociate\s+professor\b',
            r'\bassistant\s+professor\b',
            r'\bdean\b',
            r'\bprovost\b',
            r'\bchancellor\b(?=.*(?:university|college))',
            r'\bpresident\b(?=.*(?:university|college))',
            r'\bresearcher\b',
            r'\bscientist\b',
        ],
        
        'MILITARY': [
            # Military ranks
            r'\b(?:general|admiral|colonel|major|captain|lieutenant|sergeant|corporal|private)\b',
            r'\bcommander\b',
            r'\bofficer\b',
        ]
    }
    
    def __init__(self, elastic_client: ElasticClient):
        """Initialize the hybrid NER extractor."""
        self.elastic_client = elastic_client
        self.elasticsearch_ner = ElasticsearchNERExtractor(elastic_client)
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.TITLE_PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # Statistics
        self.stats = {
            'total_extractions': 0,
            'elasticsearch_entities': 0,
            'pattern_entities': 0,
            'location_descriptors': 0,
            'combined_entities': 0
        }
    
    def extract_entities_hybrid(self, text: str, context_window: int = 50) -> List[Dict[str, Any]]:
        """
        Extract entities using both Elasticsearch NER and pattern-based title extraction.
        
        Args:
            text: Input text to process
            context_window: Number of characters around entity for context
            
        Returns:
            List of all extracted entities (names + titles) with metadata
        """
        self.stats['total_extractions'] += 1
        
        # Step 1: Extract proper nouns using Elasticsearch NER
        elasticsearch_entities = self.elasticsearch_ner.extract_names_with_elasticsearch(text)
        self.stats['elasticsearch_entities'] += len(elasticsearch_entities)
        
        # Step 2: Extract job titles using pattern matching (informed by NER results)
        title_entities = self._extract_titles_with_patterns(text, context_window, elasticsearch_entities)
        self.stats['pattern_entities'] += len(title_entities)
        
        # Step 3: Extract location descriptors and possessive relationships
        location_descriptors = self._extract_location_descriptors(text, context_window)
        self.stats['location_descriptors'] += len(location_descriptors)
        
        # Step 4: Combine and deduplicate results
        combined_entities = self._combine_entities(elasticsearch_entities, title_entities, location_descriptors)
        self.stats['combined_entities'] += len(combined_entities)
        
        # Step 4: Filter out problematic entities (common NER model issues)
        filtered_entities = self._filter_problematic_entities(combined_entities)
        
        location_count = len(location_descriptors) if location_descriptors else 0
        self.logger.info(f"Hybrid extraction: {len(elasticsearch_entities)} NER + {len(title_entities)} titles + {location_count} location descriptors = {len(filtered_entities)} total (after filtering)")
        
        return filtered_entities
    
    def _filter_problematic_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out problematic entities that are common NER model issues.
        
        This includes:
        - Single-character entities (like periods, commas)
        - Punctuation-only entities
        - Common false positives
        """
        filtered_entities = []
        filtered_count = 0
        filtered_reasons = {}
        
        for entity in entities:
            entity_name = entity.get('name', '').strip()
            entity_type = entity.get('type', entity.get('class_name', 'Unknown'))
            
            # Filter out single-character entities
            if len(entity_name) <= 1:
                filtered_count += 1
                reason = f"single-character entity: '{entity_name}'"
                filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
                self.logger.debug(f"Filtered out {reason} ({entity_type})")
                continue
            
            # Filter out punctuation-only entities
            if entity_name in ['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '"', "'", '-', '_']:
                filtered_count += 1
                reason = f"punctuation entity: '{entity_name}'"
                filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
                self.logger.debug(f"Filtered out {reason} ({entity_type})")
                continue
            
            # Filter out common false positives
            false_positives = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            if entity_name.lower() in false_positives and entity_type in ['PERSON', 'ORGANIZATION']:
                filtered_count += 1
                reason = f"common false positive: '{entity_name}'"
                filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
                self.logger.debug(f"Filtered out {reason} ({entity_type})")
                continue
            
            # Keep the entity
            filtered_entities.append(entity)
        
        # Log filtering summary
        if filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} problematic entities:")
            for reason, count in filtered_reasons.items():
                self.logger.info(f"  - {reason}: {count} entities")
        
        return filtered_entities
    
    def _extract_titles_with_patterns(self, text: str, context_window: int = 50, ner_entities: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract job titles and roles using pattern matching, including compound entities."""
        title_entities = []
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract context around the match
                    context_start = max(0, start_pos - context_window)
                    context_end = min(len(text), end_pos + context_window)
                    context = text[context_start:context_end]
                    
                    # Handle compound entities differently
                    if category == 'COMPOUND_EXECUTIVE':
                        # Extract compound entities (e.g., "Tesla CEO")
                        compound_entity = self._process_compound_entity(match, text, context, start_pos, end_pos, ner_entities)
                        if compound_entity:
                            title_entities.append(compound_entity)
                    else:
                        # Handle regular title patterns
                        title = match.group().strip()
                        confidence = self._calculate_title_confidence(title, context, category)
                        
                        title_entity = {
                            'name': title,
                            'type': 'TITLE',
                            'category': category,
                            'confidence': confidence,
                            'context': context,
                            'position': start_pos,
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'extraction_method': 'pattern_based_title'
                        }
                        
                        title_entities.append(title_entity)
        
        # Remove duplicates and handle overlapping matches
        # Priority: COMPOUND_TITLE > TITLE (compound entities take precedence)
        unique_titles = []
        
        # Sort entities by priority (compound first) and then by position
        def get_priority(entity):
            if entity.get('type') == 'COMPOUND_TITLE':
                return 0  # Highest priority
            else:
                return 1  # Lower priority
        
        sorted_entities = sorted(title_entities, key=lambda e: (get_priority(e), e['start_pos']))
        
        # Track occupied positions to avoid overlaps
        occupied_ranges = []
        
        for entity in sorted_entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for occupied_start, occupied_end in occupied_ranges:
                if not (end_pos <= occupied_start or start_pos >= occupied_end):
                    overlaps = True
                    break
            
            if not overlaps:
                unique_titles.append(entity)
                occupied_ranges.append((start_pos, end_pos))
        
        return unique_titles
    
    def _extract_location_descriptors(self, text: str, context_window: int = 50) -> List[Dict[str, Any]]:
        """
        Extract location descriptors and possessive location relationships.
        
        This method captures:
        - Possessive relationships: "Tesla's Austin headquarters"
        - Location descriptors: "headquarters", "office", "facility"
        - Compound location entities: "Tesla Austin headquarters"
        """
        location_descriptors = []
        
        # Define location descriptor patterns
        location_descriptor_words = [
            'headquarters', 'office', 'facility', 'campus', 'building', 
            'center', 'hub', 'base', 'station', 'location', 'site',
            'main office', 'corporate facility', 'regional center'
        ]
        
        # Pattern 1: Organization + possessive + location + descriptor
        # "Tesla's Austin headquarters", "Microsoft's Seattle office"
        possessive_location_pattern = r'(\w+)\'s\s+(\w+)\s+(' + '|'.join(location_descriptor_words) + r')'
        
        # Pattern 2: Organization + location + descriptor  
        # "Tesla Austin headquarters", "Microsoft Seattle office"
        location_descriptor_pattern = r'(\w+)\s+(\w+)\s+(' + '|'.join(location_descriptor_words) + r')'
        
        # Pattern 3: Standalone location descriptors
        # "headquarters", "main office", "corporate facility"
        standalone_descriptors = r'\b(' + '|'.join(location_descriptor_words) + r')\b'
        
        # Apply patterns
        patterns = [
            (possessive_location_pattern, 'possessive_location'),
            (location_descriptor_pattern, 'location_descriptor'),
            (standalone_descriptors, 'standalone_descriptor')
        ]
        
        for pattern_str, pattern_name in patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                
                # Extract context around the match
                context_start = max(0, start_pos - context_window)
                context_end = min(len(text), end_pos + context_window)
                context = text[context_start:context_end]
                
                if pattern_name == 'possessive_location':
                    org, location, descriptor = match.groups()
                    entity_name = f"{org}'s {location} {descriptor}"
                    entity_type = "COMPOUND_LOCATION"
                    confidence = 0.85  # High confidence for clear possessive patterns
                elif pattern_name == 'location_descriptor':
                    org, location, descriptor = match.groups()
                    entity_name = f"{org} {location} {descriptor}"
                    entity_type = "COMPOUND_LOCATION"
                    confidence = 0.80  # High confidence for clear location patterns
                else:  # standalone_descriptors
                    entity_name = match.group(1)
                    entity_type = "LOCATION_DESCRIPTOR"
                    confidence = 0.75  # Medium confidence for standalone descriptors
                
                location_entity = {
                    'name': entity_name,
                    'type': entity_type,
                    'confidence': confidence,
                    'context': context,
                    'position': start_pos,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'extraction_method': 'location_descriptor_pattern'
                }
                
                location_descriptors.append(location_entity)
        
        # Remove duplicates and handle overlapping matches
        unique_descriptors = []
        occupied_ranges = []
        
        # Sort by position
        sorted_descriptors = sorted(location_descriptors, key=lambda e: e['start_pos'])
        
        for entity in sorted_descriptors:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for occupied_start, occupied_end in occupied_ranges:
                if not (end_pos <= occupied_start or start_pos >= occupied_end):
                    overlaps = True
                    break
            
            if not overlaps:
                unique_descriptors.append(entity)
                occupied_ranges.append((start_pos, end_pos))
        
        return unique_descriptors
    
    def _process_compound_entity(self, match, text: str, context: str, start_pos: int, end_pos: int, ner_entities: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Process compound entity matches (e.g., 'Tesla CEO', 'CEO of Apple')."""
        full_match = match.group().strip()
        groups = match.groups()
        
        # Handle different compound patterns
        if len(groups) >= 2:
            # Extract components based on pattern type
            if 'of' in full_match.lower():
                # Reverse pattern: "CEO of Apple" -> groups are (title, organization)
                title_part = groups[0].strip() if groups[0] else ""
                org_part = groups[1].strip() if groups[1] else ""
                compound_name = f"{org_part} {title_part}"  # "Apple CEO"
                primary_component = org_part  # The organization
                secondary_component = title_part  # The title
            else:
                # Forward pattern: "Tesla CEO" -> groups are (organization, title)
                org_part = groups[0].strip() if groups[0] else ""
                title_part = groups[1].strip() if groups[1] else ""
                compound_name = full_match  # "Tesla CEO"
                primary_component = org_part  # The organization
                secondary_component = title_part  # The title
            
            # POST-PROCESSING CLEANUP
            # Filter out unwanted matches using NER information
            if self._should_filter_compound_entity(primary_component, secondary_component, context, ner_entities):
                return None
            
            # Clean up the organization name (remove articles)
            primary_component = self._clean_organization_name(primary_component)
            if 'of' in full_match.lower():
                compound_name = f"{primary_component} {secondary_component}"
            else:
                compound_name = f"{primary_component} {secondary_component}"
            
            # Calculate enhanced confidence for compound entities
            confidence = self._calculate_compound_confidence(compound_name, context, primary_component, secondary_component)
            
            compound_entity = {
                'name': compound_name,
                'type': 'COMPOUND_TITLE',
                'category': 'COMPOUND_EXECUTIVE',
                'confidence': confidence,
                'context': context,
                'position': start_pos,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'extraction_method': 'compound_entity',
                'organization': primary_component,
                'title': secondary_component,
                'compound_type': 'organization_title'
            }
            
            return compound_entity
        
        return None
    
    def _calculate_compound_confidence(self, compound_name: str, context: str, organization: str, title: str) -> float:
        """Calculate confidence score for compound entities with enhanced scoring."""
        base_confidence = 0.75  # More realistic base confidence for compound entities
        
        # Boost for well-known organizations
        well_known_orgs = ['tesla', 'apple', 'microsoft', 'google', 'amazon', 'meta', 'facebook', 'twitter', 'netflix', 'uber', 'airbnb']
        if organization.lower() in well_known_orgs:
            base_confidence += 0.08
        
        # Boost for executive titles
        executive_titles = ['ceo', 'cto', 'cfo', 'coo', 'president', 'founder', 'chairman']
        if title.lower() in executive_titles:
            base_confidence += 0.05
        
        # Context analysis
        context_lower = context.lower()
        
        # Business context indicators
        business_indicators = ['company', 'corporation', 'inc', 'ltd', 'business', 'startup', 'firm']
        if any(indicator in context_lower for indicator in business_indicators):
            base_confidence += 0.03
        
        # News/media context (often mentions company executives)
        news_indicators = ['announced', 'said', 'stated', 'according to', 'reported']
        if any(indicator in context_lower for indicator in news_indicators):
            base_confidence += 0.02
        
        return min(base_confidence, 0.90)  # Cap at 0.90 for more realistic scoring
    
    def _should_filter_compound_entity(self, organization: str, title: str, context: str, ner_entities: List[Dict[str, Any]] = None) -> bool:
        """Filter out unwanted compound entity matches using NER information."""
        org_lower = organization.lower()
        title_lower = title.lower()
        context_lower = context.lower()
        
        # Filter out articles and common words that aren't organizations
        invalid_orgs = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'their', 'our', 'my']
        if org_lower in invalid_orgs:
            return True
        
        # Filter out conjunctions and connecting words (these break natural English patterns)
        conjunctions = [
            # Basic conjunctions
            'and', 'or', 'but', 'with', 'from', 'to', 'for', 'by', 'at', 'in', 'on',
            # Temporal conjunctions
            'while', 'when', 'as', 'during', 'before', 'after', 'since', 'until',
            # Causal conjunctions
            'because', 'since', 'so', 'therefore', 'thus', 'hence',
            # Conditional conjunctions
            'if', 'unless', 'provided', 'assuming', 'supposing',
            # Comparative conjunctions
            'than', 'like', 'unlike', 'similar', 'different'
        ]
        if any(conj in org_lower for conj in conjunctions):
            return True
        
        # Use NER results to filter out person names
        # "Person + Title" is not natural English - we say "Tesla CEO" not "John CEO"
        if ner_entities:
            for ner_entity in ner_entities:
                ner_name = ner_entity.get('name', '').lower()
                ner_class = ner_entity.get('class_name', '')
                
                # If the "organization" part matches a PERSON entity from NER, filter it out
                if ner_class == 'PERSON' and (org_lower == ner_name or org_lower in ner_name or ner_name in org_lower):
                    return True
        
        # Filter out generic words that aren't company names
        generic_words = ['company', 'corporation', 'business', 'organization', 'firm', 'group']
        if org_lower in generic_words:
            return True
        
        return False
    
    def _clean_organization_name(self, org_name: str) -> str:
        """Clean up organization name by removing articles and common prefixes."""
        # Remove leading articles
        words = org_name.split()
        if words and words[0].lower() in ['the', 'a', 'an']:
            words = words[1:]
        
        # Return cleaned name
        return ' '.join(words) if words else org_name
    
    def _calculate_title_confidence(self, title: str, context: str, category: str) -> float:
        """Calculate confidence score for a title based on context and specificity."""
        base_confidence = 0.7  # Base confidence for pattern matches
        
        # Boost confidence for specific titles
        specific_titles = ['president', 'ceo', 'cfo', 'cto', 'director', 'manager']
        if any(specific in title.lower() for specific in specific_titles):
            base_confidence += 0.1
        
        # Boost confidence based on context clues
        context_lower = context.lower()
        
        # Political context
        if category == 'POLITICAL':
            political_context = ['government', 'administration', 'cabinet', 'congress', 'senate']
            if any(word in context_lower for word in political_context):
                base_confidence += 0.15
        
        # Executive context
        elif category == 'EXECUTIVE':
            business_context = ['company', 'corporation', 'firm', 'business', 'organization']
            if any(word in context_lower for word in business_context):
                base_confidence += 0.15
        
        # Academic context
        elif category == 'ACADEMIC':
            academic_context = ['university', 'college', 'school', 'research', 'study']
            if any(word in context_lower for word in academic_context):
                base_confidence += 0.15
        
        # Cap confidence at 1.0
        return min(1.0, base_confidence)
    
    def _get_nationality_descriptors(self):
        """Get list of common nationality/descriptor words."""
        return {
            'nationalities': [
                'american', 'russian', 'chinese', 'british', 'french', 'german',
                'italian', 'spanish', 'japanese', 'korean', 'indian', 'canadian',
                'australian', 'brazilian', 'mexican', 'european', 'asian', 'african',
                'former', 'current', 'senior', 'junior', 'chief', 'deputy', 'acting'
            ],
            'descriptors': [
                'former', 'current', 'senior', 'junior', 'chief', 'deputy', 'acting',
                'interim', 'temporary', 'permanent', 'elected', 'appointed', 'retired'
            ]
        }
    
    def _get_title_words(self):
        """Get list of common title words."""
        return [
            'president', 'minister', 'secretary', 'director', 'manager', 'officer',
            'leader', 'chief', 'head', 'chairman', 'ceo', 'cfo', 'cto', 'coo',
            'governor', 'mayor', 'senator', 'representative', 'ambassador', 'general',
            'admiral', 'colonel', 'captain', 'judge', 'justice', 'professor', 'doctor'
        ]
    
    def _is_valid_compound_combination(self, entity1, entity2):
        """
        Validate if two adjacent entities should be combined into a compound entity.
        
        Args:
            entity1: First entity (should be MISCELLANEOUS or ORGANIZATION)
            entity2: Second entity (should be TITLE)
            
        Returns:
            tuple: (is_valid, compound_type, confidence_boost)
        """
        # Get validation word lists
        nationality_data = self._get_nationality_descriptors()
        title_words = self._get_title_words()
        
        # Extract entity text (lowercase for comparison)
        text1 = entity1.get('text', entity1.get('name', '')).lower().strip()
        text2 = entity2.get('text', entity2.get('name', '')).lower().strip()
        
        # Check type compatibility
        type1 = entity1.get('class_name', entity1.get('entity_type', entity1.get('type', '')))
        type2 = entity2.get('class_name', entity2.get('entity_type', entity2.get('type', '')))
        
        # Valid combinations (following natural English word order)
        valid_combinations = [
            ('MISCELLANEOUS', 'TITLE'),  # "Russian leader", "American president"
            ('ORGANIZATION', 'TITLE'),   # "Tesla CEO", "Microsoft President"
            # Note: ('PERSON', 'TITLE') is not natural English - we say "President Biden" not "Biden President"
        ]
        
        if (type1, type2) not in valid_combinations:
            return False, None, 0.0
        
        # Validate semantic compatibility
        is_nationality = text1 in nationality_data['nationalities']
        is_descriptor = text1 in nationality_data['descriptors']
        is_title = text2 in title_words
        
        if not is_title:
            return False, None, 0.0
        
        # Calculate confidence boost based on word lists
        confidence_boost = 0.0
        compound_type = 'COMPOUND_TITLE'
        
        if is_nationality:
            confidence_boost = 0.15  # High confidence for nationality + title
            compound_type = 'NATIONAL_TITLE'
        elif is_descriptor:
            confidence_boost = 0.10  # Medium confidence for descriptor + title
            compound_type = 'DESCRIPTIVE_TITLE'
        elif type1 == 'ORGANIZATION':
            confidence_boost = 0.12  # Good confidence for organization + title
            compound_type = 'ORGANIZATIONAL_TITLE'
        else:
            confidence_boost = 0.05  # Low confidence for other combinations
        
        return True, compound_type, confidence_boost
    
    def _are_entities_adjacent(self, entity1, entity2, max_distance=10):
        """
        Check if two entities are adjacent in the text.
        
        Args:
            entity1: First entity
            entity2: Second entity
            max_distance: Maximum character distance to consider adjacent
            
        Returns:
            bool: True if entities are adjacent
        """
        # Get positions
        end1 = entity1.get('end_pos', entity1.get('end', 0))
        start2 = entity2.get('start_pos', entity2.get('start', 0))
        
        # Check if entity2 comes right after entity1
        distance = start2 - end1
        return 0 <= distance <= max_distance
    
    def _create_compound_entity(self, entity1, entity2, compound_type, confidence_boost):
        """
        Create a compound entity from two adjacent entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            compound_type: Type of compound entity
            confidence_boost: Confidence boost to apply
            
        Returns:
            dict: Compound entity
        """
        # Combine text
        text1 = entity1.get('text', entity1.get('name', ''))
        text2 = entity2.get('text', entity2.get('name', ''))
        combined_text = f"{text1} {text2}"
        
        # Calculate positions
        start_pos = entity1.get('start_pos', entity1.get('start', 0))
        end_pos = entity2.get('end_pos', entity2.get('end', 0))
        
        # Calculate combined confidence with more realistic scoring
        conf1 = entity1.get('confidence', entity1.get('confidence_score', 0.5))
        conf2 = entity2.get('confidence', entity2.get('confidence_score', 0.5))
        
        # Use original confidence scores without normalization
            
        base_confidence = (conf1 + conf2) / 2
        final_confidence = min(0.90, base_confidence + confidence_boost)
        
        # Create compound entity
        compound_entity = {
            'text': combined_text,
            'name': combined_text,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'position': start_pos,  # For compatibility with notebook display code
            'class_name': compound_type,
            'entity_type': compound_type,
            'type': compound_type,
            'confidence': final_confidence,
            'confidence_score': final_confidence,
            'extraction_method': 'compound_validation',
            'component_entities': [entity1, entity2],
            'compound_type': compound_type,
            'context': f"Compound entity combining '{text1}' and '{text2}'"  # For notebook display
        }
        
        return compound_entity
    
    def _find_adjacent_compounds(self, entities):
        """
        Find and create compound entities from adjacent entity pairs.
        
        Args:
            entities: List of entities to analyze
            
        Returns:
            tuple: (compound_entities, used_indices)
        """
        compound_entities = []
        used_indices = set()
        
        # Sort entities by position
        sorted_entities = sorted(enumerate(entities), key=lambda x: x[1].get('start_pos', x[1].get('start', 0)))
        
        for i in range(len(sorted_entities) - 1):
            idx1, entity1 = sorted_entities[i]
            idx2, entity2 = sorted_entities[i + 1]
            
            # Skip if already used in a compound
            if idx1 in used_indices or idx2 in used_indices:
                continue
            
            # Check if entities are adjacent
            if self._are_entities_adjacent(entity1, entity2):
                # Validate compound combination
                is_valid, compound_type, confidence_boost = self._is_valid_compound_combination(entity1, entity2)
                
                if is_valid:
                    # Create compound entity
                    compound_entity = self._create_compound_entity(
                        entity1, entity2, compound_type, confidence_boost
                    )
                    compound_entities.append(compound_entity)
                    
                    # Mark entities as used
                    used_indices.add(idx1)
                    used_indices.add(idx2)
        
        return compound_entities, used_indices
    
    def _combine_entities(self, elasticsearch_entities: List[Dict[str, Any]], 
                         title_entities: List[Dict[str, Any]], 
                         location_descriptors: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Combine and deduplicate entities from both extraction methods.
        
        Priority order:
        1. Adjacent compound entities (highest priority)
        2. Existing compound entities from pattern extraction
        3. TITLE entities 
        4. Elasticsearch entities (lowest priority for overlaps)
        """
        combined = []
        
        # Convert ElasticsearchNEREntity objects to dictionaries for processing
        es_entities_dict = []
        for entity in elasticsearch_entities:
            if hasattr(entity, 'entity'):  # It's an ElasticsearchNEREntity object
                es_entities_dict.append({
                    'name': entity.entity,
                    'text': entity.entity,
                    'type': entity.class_name,
                    'class_name': entity.class_name,
                    'entity_type': entity.class_name,
                    'confidence': entity.class_probability,
                    'confidence_score': entity.class_probability,
                    'start_pos': entity.start_pos,
                    'end_pos': entity.end_pos,
                    'position': entity.start_pos,
                    'context': entity.context,
                    'extraction_method': 'elasticsearch_ner'
                })
            else:  # It's already a dictionary
                es_entities_dict.append(entity)
        
        # Step 0: Find adjacent compound entities from all available entities
        all_entities = es_entities_dict + title_entities
        if location_descriptors:
            all_entities.extend(location_descriptors)
        compound_entities, used_indices = self._find_adjacent_compounds(all_entities)
        
        # Add compound entities first (highest priority)
        for entity in compound_entities:
            combined.append(entity)
        
        # Step 1: Add existing compound entities from pattern extraction
        existing_compound_entities = [e for e in title_entities if e.get('type') == 'COMPOUND_TITLE']
        for entity in existing_compound_entities:
            # Check if this entity overlaps with our new compound entities
            overlaps = False
            for existing_entity in combined:
                if self._entities_overlap(entity, existing_entity):
                    overlaps = True
                    break
            if not overlaps:
                combined.append(entity)
        
        # Step 2: Add regular title entities that weren't used in compounds
        regular_title_entities = [e for i, e in enumerate(title_entities) 
                                if e.get('type') != 'COMPOUND_TITLE' and i not in used_indices]
        for title_entity in regular_title_entities:
            overlaps = False
            for existing_entity in combined:
                if self._entities_overlap(title_entity, existing_entity):
                    overlaps = True
                    break
            
            if not overlaps:
                combined.append(title_entity)
        
        # Step 2.5: Add location descriptors
        if location_descriptors:
            for location_entity in location_descriptors:
                overlaps = False
                for existing_entity in combined:
                    if self._entities_overlap(location_entity, existing_entity):
                        overlaps = True
                        break
                
                if not overlaps:
                    combined.append(location_entity)
        
        # Step 3: Add Elasticsearch entities that weren't used in compounds
        for i, es_entity in enumerate(es_entities_dict):
            # Skip if this entity was used in a compound
            if i not in used_indices:
                es_entity['extraction_method'] = es_entity.get('extraction_method', 'elasticsearch_ner')
                
                # Check for overlaps with existing entities
                overlaps = False
                for existing_entity in combined:
                    if self._entities_overlap(es_entity, existing_entity):
                        # Special case: if ES entity is an organization and overlapping entity is compound,
                        # skip the ES entity (compound entity is more informative)
                        if (es_entity.get('type') == 'ORGANIZATION' and 
                            existing_entity.get('type') in ['COMPOUND_TITLE', 'NATIONAL_TITLE', 'DESCRIPTIVE_TITLE', 'ORGANIZATIONAL_TITLE']):
                            overlaps = True
                            break
                        # For other overlaps, still skip
                        elif self._should_skip_overlap(es_entity, existing_entity):
                            overlaps = True
                            break
                
                if not overlaps:
                    combined.append(es_entity)
        
        # Sort by position in text
        combined.sort(key=lambda x: x.get('start_pos', x.get('position', 0)))
        
        return combined
    
    def _should_skip_overlap(self, es_entity: Dict[str, Any], existing_entity: Dict[str, Any]) -> bool:
        """Determine if an Elasticsearch entity should be skipped due to overlap."""
        # Always skip if overlapping with compound entities
        if existing_entity.get('type') == 'COMPOUND_TITLE':
            return True
        
        # For other cases, use standard overlap logic
        return True
    
    def _entities_overlap(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if two entities overlap in the text."""
        start1 = entity1.get('start_pos', entity1.get('position', 0))
        end1 = entity1.get('end_pos', start1 + len(entity1.get('name', '')))
        
        start2 = entity2.get('start_pos', entity2.get('position', 0))
        end2 = entity2.get('end_pos', start2 + len(entity2.get('name', '')))
        
        # Check for overlap
        return not (end1 <= start2 or end2 <= start1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return self.stats.copy()
    
    def extract_names_with_elasticsearch(self, text: str) -> List[Dict[str, Any]]:
        """Compatibility method for existing code that expects this method name."""
        # Call the NER extractor directly, not the hybrid method
        return self.elasticsearch_ner.extract_names_with_elasticsearch(text)


# Compatibility function for existing code
def create_hybrid_ner_extractor(elastic_client: ElasticClient) -> HybridNERExtractor:
    """Factory function to create hybrid NER extractor."""
    return HybridNERExtractor(elastic_client)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # This would require proper Elasticsearch setup
    print("Hybrid NER Extractor - combines Elasticsearch NER with pattern-based title extraction")
    print("For testing, use the test script or integrate with the web UI")
