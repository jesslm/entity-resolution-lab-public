"""
Entity matching section for the tutorial notebook generator.
"""

def get_entity_matching_section():
    """
    Returns the code for the entity matching section.
    """
    return """
    def _add_entity_matching_section(self):
        """Add entity matching section."""
        self.notebook.cells.append(new_markdown_cell("""
## Entity Matching

The third step in the entity resolution process is to match the extracted entities against our watch list. This is where the real power of the system comes into play, with multiple matching strategies:

1. **Exact Matching**: Direct string comparison
2. **Lexical Matching**: BM25 text similarity
3. **Semantic Matching**: Vector embeddings similarity
4. **Hybrid Matching**: Combination of lexical and semantic approaches
5. **Role-Based Matching**: Matching based on roles/titles (e.g., "US President" → "Joe Biden")

Let's match the entities we extracted from our sample article:
"""))

        self.notebook.cells.append(new_code_cell("""
from entity_resolution_demo.entity_matching.real_time_entity_matcher import RealTimeEntityMatcher

# Create entity matcher
try:
    matcher = RealTimeEntityMatcher(
        watch_list=watch_list,
        elastic_client=elastic_client,
        config=config
    )
    print("✅ Successfully created entity matcher")
except Exception as e:
    print(f"❌ Error creating entity matcher: {e}")
    print("Using mock entity matcher for demonstration purposes")
    
    # Create a mock entity matcher
    class MockEntityMatcher:
        def match_article(self, processed_article):
            from entity_resolution_demo.entity_matching.entity_matching import EntityMatchingResult, EntityMatch
            from entity_resolution_demo.entity_preparation.entity_watch_list import WatchedEntity
            
            # Create some mock matches based on the processed article
            matches = []
            
            # For each extracted entity, try to find a match in our watch list
            for extracted_entity in processed_article.extracted_entities:
                # Check for known entities
                if extracted_entity.name == "Joe Biden" or "President" in extracted_entity.name:
                    watched_entity = WatchedEntity(
                        id="joe_biden_000",
                        name="Joe Biden",
                        entity_type="PERSON",
                        description="President of the United States",
                        priority="high"
                    )
                    match = EntityMatch(
                        extracted_entity=extracted_entity,
                        watched_entity=watched_entity,
                        confidence=0.95,
                        match_type="role_based"
                    )
                    matches.append(match)
                    
                elif extracted_entity.name == "Elon Musk" or "Tesla" in extracted_entity.name:
                    watched_entity = WatchedEntity(
                        id="elon_musk_000",
                        name="Elon Musk",
                        entity_type="PERSON",
                        description="CEO of Tesla and SpaceX",
                        priority="high"
                    )
                    match = EntityMatch(
                        extracted_entity=extracted_entity,
                        watched_entity=watched_entity,
                        confidence=0.98,
                        match_type="exact"
                    )
                    matches.append(match)
                    
                elif extracted_entity.name == "Taylor Swift":
                    watched_entity = WatchedEntity(
                        id="taylor_swift_000",
                        name="Taylor Swift",
                        entity_type="PERSON",
                        description="American singer-songwriter",
                        priority="medium"
                    )
                    match = EntityMatch(
                        extracted_entity=extracted_entity,
                        watched_entity=watched_entity,
                        confidence=0.99,
                        match_type="exact"
                    )
                    matches.append(match)
                    
                elif extracted_entity.name == "Microsoft":
                    watched_entity = WatchedEntity(
                        id="microsoft_000",
                        name="Microsoft",
                        entity_type="ORGANIZATION",
                        description="Technology company",
                        priority="high"
                    )
                    match = EntityMatch(
                        extracted_entity=extracted_entity,
                        watched_entity=watched_entity,
                        confidence=0.97,
                        match_type="exact"
                    )
                    matches.append(match)
            
            # Create result
            result = EntityMatchingResult(
                matches_found=matches,
                total_extracted_entities=len(processed_article.extracted_entities),
                total_matches=len(matches),
                processing_time=0.5
            )
            
            return result
    
    matcher = MockEntityMatcher()
    print("📝 Using mock entity matcher with simulated matches")

# Match entities
try:
    result = matcher.match_article(processed_article)
    print("✅ Successfully matched entities")
except Exception as e:
    print(f"❌ Error matching entities: {e}")
    # Create a minimal result for demonstration
    from entity_resolution_demo.entity_matching.entity_matching import EntityMatchingResult
    result = EntityMatchingResult(
        matches_found=[],
        total_extracted_entities=len(processed_article.extracted_entities),
        total_matches=0,
        processing_time=0.0
    )
    print("⚠️ Created empty matching result for demonstration")

# Print matches
if result.matches_found:
    print(f"Found {len(result.matches_found)} matches:")
    for i, match in enumerate(result.matches_found):
        print(f"Match {i+1}: {match.extracted_entity.name} → {match.watched_entity.name}")
        print(f"  Confidence: {match.confidence:.2f}")
        print(f"  Match Type: {match.match_type}")
        if hasattr(match, 'reasoning') and match.reasoning:
            explanation = match.reasoning[:150] + "..." if len(match.reasoning) > 150 else match.reasoning
            print(f"  Explanation: {explanation}")
        print()
else:
    print("No matches found")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### LLM-Powered Explanations

One of the most powerful features of our entity resolution system is the ability to generate detailed explanations for matches using LLMs. Let's enhance our matches with LLM explanations:
"""))

        self.notebook.cells.append(new_code_cell("""
from entity_resolution_demo.entity_matching.enhanced_batch_match_judge import EnhancedBatchMatchJudge

# Create batch match judge
try:
    batch_judge = EnhancedBatchMatchJudge(
        config=config,
        batch_size=5,
        es_client=elastic_client.es if hasattr(elastic_client, 'es') else None
    )
    print("✅ Successfully created batch match judge")
except Exception as e:
    print(f"❌ Error creating batch match judge: {e}")
    print("Using mock batch match judge for demonstration purposes")
    
    # Create a mock batch match judge
    class MockBatchMatchJudge:
        def batch_judge_matches(self, batch):
            from entity_resolution_demo.entity_matching.entity_matching import EntityMatch
            from entity_resolution_demo.entity_preparation.entity_watch_list import WatchedEntity
            from entity_resolution_demo.article_processing.article_processor import ExtractedEntity
            
            # Process each match in the batch
            processed_matches = []
            for item in batch:
                # Create extracted and watched entities
                extracted_entity = ExtractedEntity(
                    name=item['query_name'],
                    entity_type="PERSON" if item['query_name'] in ["Joe Biden", "Elon Musk", "Taylor Swift"] else "ORGANIZATION",
                    confidence=0.9,
                    context=item['context'],
                    start_pos=0,
                    end_pos=0
                )
                
                watched_entity = WatchedEntity(
                    id=item['candidate_name'].lower().replace(' ', '_') + "_000",
                    name=item['candidate_name'],
                    entity_type="PERSON" if item['candidate_name'] in ["Joe Biden", "Elon Musk", "Taylor Swift"] else "ORGANIZATION",
                    description="",
                    priority="high"
                )
                
                # Create match with LLM explanation
                match = EntityMatch(
                    extracted_entity=extracted_entity,
                    watched_entity=watched_entity,
                    confidence=0.95,
                    match_type="exact"
                )
                
                # Add LLM explanation fields
                match.reasoning = f"The extracted entity '{item['query_name']}' matches the watched entity '{item['candidate_name']}' with high confidence based on the name similarity and context."
                match.key_evidence = ["Name similarity", "Context support"]
                match.risk_factors = ["None identified"]
                match.explanation_full = f"This is a strong match between '{item['query_name']}' and '{item['candidate_name']}' based on exact name matching and contextual evidence."
                
                processed_matches.append(match)
            
            return processed_matches
    
    batch_judge = MockBatchMatchJudge()
    print("📝 Using mock batch match judge with simulated explanations")

# Prepare batch for processing
batch = []
for i, match in enumerate(result.matches_found):
    batch.append({
        'pair_index': i,
        'query_name': match.extracted_entity.name,
        'candidate_name': match.watched_entity.name,
        'context': match.extracted_entity.context or ""
    })

# Process batch
if batch:
    try:
        processed_batch = batch_judge.batch_judge_matches(batch)
        print("✅ Successfully generated LLM explanations")
        
        # Print enhanced explanations
        print("\\nEnhanced LLM Explanations:")
        for i, processed_match in enumerate(processed_batch):
            print(f"\\nMatch: {processed_match.extracted_entity.name} → {processed_match.watched_entity.name}")
            print(f"Confidence: {processed_match.confidence:.2f}")
            print(f"Match Type: {processed_match.match_type}")
            
            if hasattr(processed_match, 'reasoning') and processed_match.reasoning:
                print(f"Reasoning: {processed_match.reasoning}")
                
            if hasattr(processed_match, 'key_evidence') and processed_match.key_evidence:
                print(f"Key Evidence: {', '.join(processed_match.key_evidence[:2])}")
                
            if hasattr(processed_match, 'risk_factors') and processed_match.risk_factors:
                print(f"Risk Factors: {', '.join(processed_match.risk_factors[:2])}")
    except Exception as e:
        print(f"❌ Error generating LLM explanations: {e}")
        print("Continuing with basic matches")
else:
    print("No matches to process")
"""))
"""
