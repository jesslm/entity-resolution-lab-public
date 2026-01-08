"""
Shared Function Definitions for Entity Resolution

This module contains the shared function definitions used by both individual
and batch processing implementations to ensure consistency across all approaches.
"""

# Shared match types - must be consistent across all implementations
SHARED_MATCH_TYPES = [
    "exact", "nickname", "partial_last", "initial", "missing_components",
    "out_of_order", "phonetic", "cultural", "semantic", "title_role", "unlikely"
]

# Shared field definitions for individual match results
INDIVIDUAL_MATCH_SCHEMA = {
    "confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "Match confidence score (0-1). Use 0.8-1.0 for unique names with strong evidence, 0.5-0.7 for ambiguous cases, 0.0-0.4 for weak evidence or common names without context"
    },
    "is_match": {
        "type": "boolean",
        "description": "Whether the names refer to the same entity. True for matches with confidence > 0.5, False otherwise"
    },
    "match_type": {
        "type": "string",
        "enum": SHARED_MATCH_TYPES,
        "description": f"""Type of match: {', '.join(SHARED_MATCH_TYPES)}. Examples:
- 'exact': "John Smith" vs "John Smith", "Tesla" vs "Tesla, Inc."
- 'nickname': "Bob" vs "Robert", "Bill" vs "William", "Liz" vs "Elizabeth"
- 'partial_last': "Smith" vs "John Smith", "Johnson" vs "Robert Johnson"
- 'initial': "J. Smith" vs "John Smith", "A. Johnson" vs "Alice Johnson"
- 'missing_components': "Phil Carr" vs "Phillip Charles Carr", "Mary" vs "Mary Elizabeth"
- 'out_of_order': "Smith, John" vs "John Smith", "Johnson, Robert" vs "Robert Johnson"
- 'phonetic': "Smith" vs "Smyth", "Johnson" vs "Jonson", "Taylor" vs "Tayler"
- 'cultural': "José" vs "Jose", "François" vs "Francois", "Müller" vs "Mueller"
- 'semantic': "electric car manufacturer" vs "Tesla", "social media platform" vs "Facebook"
- 'title_role': "POTUS" vs "Joe Biden", "CEO of Apple" vs "Tim Cook"
- 'unlikely': "John Smith" vs "Jane Smith", "Apple Inc." vs "Microsoft Corp." """
    },
    "reasoning": {
        "type": "string",
        "description": "Detailed explanation of the decision process, including analysis of name similarity, context support, and any risk factors like common surnames or missing context"
    },
    "explanation_full": {
        "type": "string",
        "description": "Concise summary suitable for display to users, explaining the match decision in simple terms"
    },
    "confidence_factors": {
        "type": "object",
        "properties": {
            "name_similarity": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "How similar the names are (0.0-1.0)"
            },
            "context_support": {
                "type": "number", 
                "minimum": 0,
                "maximum": 1,
                "description": "How much context supports the match (0.0-1.0)"
            },
            "name_uniqueness": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "How unique/rare the names are (0.0-1.0)"
            }
        },
        "required": ["name_similarity", "context_support", "name_uniqueness"],
        "description": "Specific factors that influenced confidence scoring"
    },
    "key_evidence": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "maxItems": 4,
        "description": "Key pieces of evidence supporting or contradicting the match, such as nickname relationships, shared components, or contextual clues"
    },
    "risk_factors": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Potential concerns or ambiguities that lower confidence, such as 'common last name', 'initial-only match', 'no contextual support', 'missing context'"
    }
}

# Individual match function definition with comprehensive examples
INDIVIDUAL_FUNCTION_DEFINITIONS = [
    {
        "name": "analyze_name_match",
        "description": """Analyze if two names refer to the same entity. Consider these match patterns with examples:

EXACT: "John Smith" vs "John Smith", "Tesla" vs "Tesla, Inc."
NICKNAME: "Bob" vs "Robert", "Bill" vs "William", "Liz" vs "Elizabeth"  
PARTIAL_LAST: "Smith" vs "John Smith", "Johnson" vs "Robert Johnson"
INITIAL: "J. Smith" vs "John Smith", "A. Johnson" vs "Alice Johnson"
MISSING_COMPONENTS: "Phil Carr" vs "Phillip Charles Carr", "Mary" vs "Mary Elizabeth"
OUT_OF_ORDER: "Smith, John" vs "John Smith", "Johnson, Robert" vs "Robert Johnson"
PHONETIC: "Smith" vs "Smyth", "Johnson" vs "Jonson", "Taylor" vs "Tayler"
CULTURAL: "José" vs "Jose", "François" vs "Francois", "Müller" vs "Mueller"
SEMANTIC: "electric car manufacturer" vs "Tesla", "social media platform" vs "Facebook"
TITLE_ROLE: "POTUS" vs "Joe Biden", "CEO of Apple" vs "Tim Cook", "Mayor" vs "John Smith"
UNLIKELY: "John Smith" vs "Jane Smith", "Apple Inc." vs "Microsoft Corp."

Use high confidence (0.8-1.0) for unique names and strong evidence, medium confidence (0.5-0.7) for ambiguous cases, and low confidence (0.0-0.4) for weak evidence or common names without context.""",
        "parameters": {
            "type": "object",
            "properties": INDIVIDUAL_MATCH_SCHEMA,
            "required": ["confidence", "is_match", "match_type", "reasoning", "explanation_full", "confidence_factors", "key_evidence", "risk_factors"]
        }
    }
]

# Batch match function definition
BATCH_FUNCTION_DEFINITIONS = [
    {
        "name": "analyze_batch_matches",
        "description": """Analyze multiple name pairs for entity resolution. Consider these match patterns with examples:

EXACT: "John Smith" vs "John Smith", "Tesla" vs "Tesla, Inc."
NICKNAME: "Bob" vs "Robert", "Bill" vs "William", "Liz" vs "Elizabeth"  
PARTIAL_LAST: "Smith" vs "John Smith", "Johnson" vs "Robert Johnson"
INITIAL: "J. Smith" vs "John Smith", "A. Johnson" vs "Alice Johnson"
MISSING_COMPONENTS: "Phil Carr" vs "Phillip Charles Carr", "Mary" vs "Mary Elizabeth"
OUT_OF_ORDER: "Smith, John" vs "John Smith", "Johnson, Robert" vs "Robert Johnson"
PHONETIC: "Smith" vs "Smyth", "Johnson" vs "Jonson", "Taylor" vs "Tayler"
CULTURAL: "José" vs "Jose", "François" vs "Francois", "Müller" vs "Mueller"
SEMANTIC: "electric car manufacturer" vs "Tesla", "social media platform" vs "Facebook"
TITLE_ROLE: "POTUS" vs "Joe Biden", "CEO of Apple" vs "Tim Cook", "Mayor" vs "John Smith"
UNLIKELY: "John Smith" vs "Jane Smith", "Apple Inc." vs "Microsoft Corp."

Use high confidence (0.8-1.0) for unique names and strong evidence, medium confidence (0.5-0.7) for ambiguous cases, and low confidence (0.0-0.4) for weak evidence or common names without context.

IMPORTANT: Evaluate each name pair INDEPENDENTLY. Never refer to other pairs in your reasoning.""",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": INDIVIDUAL_MATCH_SCHEMA,
                        "required": ["confidence", "is_match", "match_type", "reasoning", "explanation_full", "confidence_factors", "key_evidence", "risk_factors"]
                    },
                    "description": "Array of match results for each name pair"
                }
            },
            "required": ["results"]
        }
    }
]

# Validation function to ensure consistency
def validate_function_definitions():
    """Validate that function definitions are consistent"""
    
    # Check that individual and batch schemas are identical
    individual_schema = INDIVIDUAL_FUNCTION_DEFINITIONS[0]["parameters"]["properties"]
    batch_schema = BATCH_FUNCTION_DEFINITIONS[0]["parameters"]["properties"]["results"]["items"]["properties"]
    
    if individual_schema != batch_schema:
        raise ValueError("Individual and batch schemas must be identical for consistency")
    
    # Check that match types are consistent
    individual_match_types = individual_schema["match_type"]["enum"]
    batch_match_types = batch_schema["match_type"]["enum"]
    
    if individual_match_types != batch_match_types:
        raise ValueError("Match types must be identical between individual and batch processing")
    
    if individual_match_types != SHARED_MATCH_TYPES:
        raise ValueError("Match types must match the shared definition")
    
    return True

# Test the validation
if __name__ == "__main__":
    try:
        validate_function_definitions()
        print("✅ Function definitions are consistent")
    except ValueError as e:
        print(f"❌ Function definitions are inconsistent: {e}")
