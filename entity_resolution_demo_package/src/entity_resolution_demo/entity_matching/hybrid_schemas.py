"""
Hybrid schemas for optimized function calling performance.
Keeps original detailed prompts for quality, optimizes output schema for performance.
"""

from pydantic import BaseModel, Field
from typing import List, Literal

# Essential match types
MatchType = Literal[
    "exact", "nickname", "partial_last", "initial", "missing_components", 
    "out_of_order", "phonetic", "cultural", "semantic", "title_role", "unlikely"
]

class HybridNameMatchResult(BaseModel):
    """Hybrid result schema with essential fields and optimized structure."""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    is_match: bool = Field(..., description="Whether the names refer to the same entity")
    match_type: MatchType = Field(..., description="Type of match pattern identified")
    reasoning: str = Field(..., description="Brief explanation of the decision")
    # Keep one additional field for context
    key_evidence: List[str] = Field(default_factory=list, description="Key evidence supporting the decision")

# Hybrid function definitions for individual processing
HYBRID_INDIVIDUAL_FUNCTION_DEFINITIONS = [
    {
        "name": "analyze_name_match",
        "description": """You are an expert at entity resolution. Your task is to determine whether two names refer to the same entity.

## Match Types and Examples:

**EXACT MATCHES:**
- "John Smith" vs "John Smith" (identical names)
- "Tesla" vs "Tesla, Inc." (company name with/without legal suffix)
- "Dr. Sarah Johnson" vs "Sarah Johnson" (title vs name)

**NICKNAME MATCHES:**
- "Bob" vs "Robert" (common nickname)
- "Bill" vs "William" (common nickname)
- "Liz" vs "Elizabeth" (common nickname)
- "Mike" vs "Michael" (common nickname)

**PARTIAL LAST NAME MATCHES:**
- "Smith" vs "John Smith" (last name vs full name)
- "Johnson" vs "Robert Johnson" (last name vs full name)
- "Gates" vs "Bill Gates" (last name vs full name)

**INITIAL MATCHES:**
- "J. Smith" vs "John Smith" (initial vs full name)
- "A. Johnson" vs "Alice Johnson" (initial vs full name)
- "M. Williams" vs "Mary Williams" (initial vs full name)

**MISSING COMPONENTS:**
- "Phil Carr" vs "Phillip Charles Carr" (shortened first name, missing middle)
- "Mary" vs "Mary Elizabeth" (missing middle name)
- "John" vs "John Michael Smith" (missing middle and last)

**OUT OF ORDER:**
- "Smith, John" vs "John Smith" (last, first vs first, last)
- "Johnson, Robert" vs "Robert Johnson" (last, first vs first, last)

**PHONETIC MATCHES:**
- "Smith" vs "Smyth" (phonetic spelling)
- "Johnson" vs "Jonson" (phonetic spelling)
- "Taylor" vs "Tayler" (phonetic spelling)

**CULTURAL VARIATIONS:**
- "José" vs "Jose" (accented vs unaccented)
- "François" vs "Francois" (accented vs unaccented)
- "Müller" vs "Mueller" (umlaut vs standard)

**SEMANTIC MATCHES:**
- "electric car manufacturer" vs "Tesla" (description vs company)
- "social media platform" vs "Facebook" (description vs company)
- "search engine" vs "Google" (description vs company)

**TITLE/ROLE MATCHES:**
- "POTUS" vs "Joe Biden" (title vs person)
- "CEO of Apple" vs "Tim Cook" (role vs person)
- "Tesla CEO" vs "Tesla Inc" (CEO role vs company)

**UNLIKELY MATCHES:**
- "John Smith" vs "Jane Smith" (different first names)
- "Apple Inc." vs "Microsoft Corp." (different companies)
- "Dr. Smith" vs "Dr. Johnson" (different last names)

## Confidence Scoring:

- **0.9-1.0**: Very high confidence (exact matches, unique names with strong evidence)
- **0.7-0.8**: High confidence (nickname matches, clear partial matches)
- **0.5-0.6**: Medium confidence (ambiguous cases, some evidence)
- **0.3-0.4**: Low confidence (weak evidence, multiple possibilities)
- **0.0-0.2**: Very low confidence (unlikely matches, conflicting evidence)

## Key Evidence Guidelines:

- Include specific name components that match
- Note any titles, roles, or context that supports the match
- Mention any obvious differences that might indicate non-match
- Keep evidence concise but informative

Analyze the name pair carefully and provide a structured result.""",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score from 0.0 to 1.0"
                },
                "is_match": {
                    "type": "boolean",
                    "description": "Whether the names refer to the same entity"
                },
                "match_type": {
                    "type": "string",
                    "enum": ["exact", "nickname", "partial_last", "initial", "missing_components", 
                            "out_of_order", "phonetic", "cultural", "semantic", "title_role", "unlikely"],
                    "description": "Type of match pattern identified"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the decision"
                },
                "key_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key evidence supporting the decision"
                }
            },
            "required": ["confidence", "is_match", "match_type", "reasoning", "key_evidence"]
        }
    }
]

# Hybrid function definitions for batch processing
HYBRID_BATCH_FUNCTION_DEFINITIONS = [
    {
        "name": "analyze_batch_matches",
        "description": """You are an expert at entity resolution. Your task is to determine whether multiple name pairs refer to the same entities.

## Match Types and Examples:

**EXACT MATCHES:**
- "John Smith" vs "John Smith" (identical names)
- "Tesla" vs "Tesla, Inc." (company name with/without legal suffix)
- "Dr. Sarah Johnson" vs "Sarah Johnson" (title vs name)

**NICKNAME MATCHES:**
- "Bob" vs "Robert" (common nickname)
- "Bill" vs "William" (common nickname)
- "Liz" vs "Elizabeth" (common nickname)
- "Mike" vs "Michael" (common nickname)

**PARTIAL LAST NAME MATCHES:**
- "Smith" vs "John Smith" (last name vs full name)
- "Johnson" vs "Robert Johnson" (last name vs full name)
- "Gates" vs "Bill Gates" (last name vs full name)

**INITIAL MATCHES:**
- "J. Smith" vs "John Smith" (initial vs full name)
- "A. Johnson" vs "Alice Johnson" (initial vs full name)
- "M. Williams" vs "Mary Williams" (initial vs full name)

**MISSING COMPONENTS:**
- "Phil Carr" vs "Phillip Charles Carr" (shortened first name, missing middle)
- "Mary" vs "Mary Elizabeth" (missing middle name)
- "John" vs "John Michael Smith" (missing middle and last)

**OUT OF ORDER:**
- "Smith, John" vs "John Smith" (last, first vs first, last)
- "Johnson, Robert" vs "Robert Johnson" (last, first vs first, last)

**PHONETIC MATCHES:**
- "Smith" vs "Smyth" (phonetic spelling)
- "Johnson" vs "Jonson" (phonetic spelling)
- "Taylor" vs "Tayler" (phonetic spelling)

**CULTURAL VARIATIONS:**
- "José" vs "Jose" (accented vs unaccented)
- "François" vs "Francois" (accented vs unaccented)
- "Müller" vs "Mueller" (umlaut vs standard)

**SEMANTIC MATCHES:**
- "electric car manufacturer" vs "Tesla" (description vs company)
- "social media platform" vs "Facebook" (description vs company)
- "search engine" vs "Google" (description vs company)

**TITLE/ROLE MATCHES:**
- "POTUS" vs "Joe Biden" (title vs person)
- "CEO of Apple" vs "Tim Cook" (role vs person)
- "Tesla CEO" vs "Tesla Inc" (CEO role vs company)

**UNLIKELY MATCHES:**
- "John Smith" vs "Jane Smith" (different first names)
- "Apple Inc." vs "Microsoft Corp." (different companies)
- "Dr. Smith" vs "Dr. Johnson" (different last names)

## Confidence Scoring:

- **0.9-1.0**: Very high confidence (exact matches, unique names with strong evidence)
- **0.7-0.8**: High confidence (nickname matches, clear partial matches)
- **0.5-0.6**: Medium confidence (ambiguous cases, some evidence)
- **0.3-0.4**: Low confidence (weak evidence, multiple possibilities)
- **0.0-0.2**: Very low confidence (unlikely matches, conflicting evidence)

## Key Evidence Guidelines:

- Include specific name components that match
- Note any titles, roles, or context that supports the match
- Mention any obvious differences that might indicate non-match
- Keep evidence concise but informative

**IMPORTANT**: Evaluate each name pair INDEPENDENTLY. Never refer to other pairs in your reasoning.""",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence score from 0.0 to 1.0"
                            },
                            "is_match": {
                                "type": "boolean",
                                "description": "Whether the names refer to the same entity"
                            },
                            "match_type": {
                                "type": "string",
                                "enum": ["exact", "nickname", "partial_last", "initial", "missing_components", 
                                        "out_of_order", "phonetic", "cultural", "semantic", "title_role", "unlikely"],
                                "description": "Type of match pattern identified"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of the decision"
                            },
                            "key_evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Key evidence supporting the decision"
                            }
                        },
                        "required": ["confidence", "is_match", "match_type", "reasoning", "key_evidence"]
                    },
                    "description": "Array of match results for each name pair"
                }
            },
            "required": ["results"]
        }
    }
]
