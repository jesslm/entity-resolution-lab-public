"""
Minimal schemas for optimized function calling performance.
Focuses on essential fields only to reduce JSON output size and increase batch capacity.
"""

from pydantic import BaseModel, Field
from typing import List, Literal

# Essential match types
MatchType = Literal[
    "exact", "nickname", "partial_last", "initial", "missing_components", 
    "out_of_order", "phonetic", "cultural", "semantic", "title_role", "unlikely"
]

class MinimalNameMatchResult(BaseModel):
    """Minimal result schema with only essential fields."""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    is_match: bool = Field(..., description="Whether the names refer to the same entity")
    match_type: MatchType = Field(..., description="Type of match pattern identified")
    reasoning: str = Field(..., description="Brief explanation of the decision")

# Enhanced minimal function definitions for individual processing
MINIMAL_INDIVIDUAL_FUNCTION_DEFINITIONS = [
    {
        "name": "analyze_name_match",
        "description": """Analyze a single name pair for entity resolution. Consider these match patterns:

EXACT: "John Smith" vs "John Smith", "Tesla" vs "Tesla, Inc."
NICKNAME: "Bob" vs "Robert", "Bill" vs "William", "Liz" vs "Elizabeth"  
PARTIAL_LAST: "Smith" vs "John Smith", "Johnson" vs "Robert Johnson"
INITIAL: "J. Smith" vs "John Smith", "A. Johnson" vs "Alice Johnson"
MISSING_COMPONENTS: "Phil Carr" vs "Phillip Charles Carr", "Mary" vs "Mary Elizabeth"
OUT_OF_ORDER: "Smith, John" vs "John Smith", "Johnson, Robert" vs "Robert Johnson"
PHONETIC: "Smith" vs "Smyth", "Johnson" vs "Jonson", "Taylor" vs "Tayler"
CULTURAL: "José" vs "Jose", "François" vs "Francois", "Müller" vs "Mueller"
SEMANTIC: "electric car manufacturer" vs "Tesla", "social media platform" vs "Facebook"
TITLE_ROLE: "POTUS" vs "Joe Biden", "CEO of Apple" vs "Tim Cook"
UNLIKELY: "John Smith" vs "Jane Smith", "Apple Inc." vs "Microsoft Corp."

CRITICAL RULES:
- Tesla Inc vs Tesla, Inc. = EXACT match (same company, different formatting)
- Tesla CEO vs Tesla Inc = TITLE_ROLE match (CEO refers to the company)
- John Smith vs Jane Smith = UNLIKELY (different first names, same last name)
- President Biden vs Joe Biden = TITLE_ROLE match (President is title for Joe Biden)
- Dr. Smith vs John Smith = TITLE_ROLE match (Dr. is title, Smith is last name)

Use high confidence (0.8-1.0) for unique names and strong evidence, medium confidence (0.5-0.7) for ambiguous cases, and low confidence (0.0-0.4) for weak evidence.""",
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
                }
            },
            "required": ["confidence", "is_match", "match_type", "reasoning"]
        }
    }
]

# Enhanced minimal function definitions for batch processing
MINIMAL_BATCH_FUNCTION_DEFINITIONS = [
    {
        "name": "analyze_batch_matches",
        "description": """Analyze multiple name pairs for entity resolution. Consider these match patterns:

EXACT: "John Smith" vs "John Smith", "Tesla" vs "Tesla, Inc."
NICKNAME: "Bob" vs "Robert", "Bill" vs "William", "Liz" vs "Elizabeth"  
PARTIAL_LAST: "Smith" vs "John Smith", "Johnson" vs "Robert Johnson"
INITIAL: "J. Smith" vs "John Smith", "A. Johnson" vs "Alice Johnson"
MISSING_COMPONENTS: "Phil Carr" vs "Phillip Charles Carr", "Mary" vs "Mary Elizabeth"
OUT_OF_ORDER: "Smith, John" vs "John Smith", "Johnson, Robert" vs "Robert Johnson"
PHONETIC: "Smith" vs "Smyth", "Johnson" vs "Jonson", "Taylor" vs "Tayler"
CULTURAL: "José" vs "Jose", "François" vs "Francois", "Müller" vs "Mueller"
SEMANTIC: "electric car manufacturer" vs "Tesla", "social media platform" vs "Facebook"
TITLE_ROLE: "POTUS" vs "Joe Biden", "CEO of Apple" vs "Tim Cook"
UNLIKELY: "John Smith" vs "Jane Smith", "Apple Inc." vs "Microsoft Corp."

CRITICAL RULES:
- Tesla Inc vs Tesla, Inc. = EXACT match (same company, different formatting)
- Tesla CEO vs Tesla Inc = TITLE_ROLE match (CEO refers to the company)
- John Smith vs Jane Smith = UNLIKELY (different first names, same last name)
- President Biden vs Joe Biden = TITLE_ROLE match (President is title for Joe Biden)
- Dr. Smith vs John Smith = TITLE_ROLE match (Dr. is title, Smith is last name)

Use high confidence (0.8-1.0) for unique names and strong evidence, medium confidence (0.5-0.7) for ambiguous cases, and low confidence (0.0-0.4) for weak evidence.

IMPORTANT: Evaluate each name pair INDEPENDENTLY. Never refer to other pairs in your reasoning.""",
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
                            }
                        },
                        "required": ["confidence", "is_match", "match_type", "reasoning"]
                    },
                    "description": "Array of match results for each name pair"
                }
            },
            "required": ["results"]
        }
    }
]
