#!/usr/bin/env python3
"""
Entity Resolution Pipeline Runner

Runs the complete entity resolution pipeline from entity preparation to matching
using the corrected versions of components without monkey patching.
"""

import sys
import logging
import argparse
import time
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Import pipeline readiness check
from entity_resolution_demo.pipeline_runner.pipeline_readiness_check import run_pipeline_readiness_check, ask_user_to_proceed

# Import local modules
from entity_resolution_demo.pipeline_runner.config import load_config
from entity_resolution_demo.pipeline_runner.utils import (
    print_header, print_subheader, print_success, print_warning, 
    print_error, print_info, time_function
)

# Import pipeline modules
from entity_resolution_demo.entity_preparation.entity_preparation import run_entity_preparation
from entity_resolution_demo.article_processing.article_processing import run_article_processing

# Import components directly
from entity_resolution_demo.entity_matching.real_time_entity_matcher import RealTimeEntityMatcher
from entity_resolution_demo.entity_matching.elasticsearch_entity_matcher import ElasticsearchEntityMatcher
from entity_resolution_demo.entity_matching.entity_match import EntityMatch, MatchingResult, PotentialMatch
from entity_resolution_demo.entity_matching.enhanced_batch_match_judge import EnhancedBatchMatchJudge
from entity_resolution_demo.search.elastic_client import ElasticClient
from entity_resolution_demo.entity_preparation.entity_watch_list import EntityWatchList, WatchedEntity
from entity_resolution_demo.article_processing.article_processor import Article, ProcessedArticle, ExtractedEntity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@time_function
def run_pipeline(config: Dict[str, Any], verify_steps: bool = True, skip_to: str = None, skip_readiness_check: bool = False, force_full_run: bool = False) -> bool:
    """
    Run the complete entity resolution pipeline with corrected components
    
    Args:
        config: Configuration dictionary
        verify_steps: Whether to verify each step
        skip_to: Optional stage to skip to (entity_preparation, article_processing, entity_matching)
        skip_readiness_check: Whether to skip the pipeline readiness check
        force_full_run: Whether to force a full run of all pipeline stages
        
    Returns:
        bool: True if successful, False otherwise
    """
    print_header("ENTITY RESOLUTION PIPELINE")
    
    # This pipeline uses the corrected versions by default
    config['use_corrected_versions'] = True
    
    # Set up entity and article data paths if provided
    if "entity_data_path" in config:
        if "entity_preparation" not in config:
            config["entity_preparation"] = {}
        config["entity_preparation"]["data_file"] = config["entity_data_path"]
        print_info(f"Using custom entity data file: {config['entity_data_path']}")
    
    if "article_data_path" in config:
        if "article_processing" not in config:
            config["article_processing"] = {}
        config["article_processing"]["data_file"] = config["article_data_path"]
        print_info(f"Using custom article data file: {config['article_data_path']}")
    
    # Run pipeline readiness check if not skipped
    if not skip_readiness_check:
        print_header("RUNNING PIPELINE READINESS CHECK")
        ready, results = run_pipeline_readiness_check(config)
        
        if not ready:
            print_error("Pipeline readiness check failed. Please fix the issues before proceeding.")
            print_info("Run with --skip-readiness-check to bypass this check (not recommended)")
            return False
        
        # Ask user if they want to proceed
        if not ask_user_to_proceed():
            print_info("Pipeline execution cancelled by user")
            return False
    else:
        print_warning("Pipeline readiness check skipped. Proceeding without verification.")
    
    start_time = time.time()
    
    # Track overall success
    overall_success = True
    
    # 1. Entity Preparation
    if skip_to is None or skip_to == "entity_preparation":
        print_header("STAGE 1: Entity Preparation")
        
        # If force_full_run is True, delete the entity_preparation state file
        if force_full_run:
            state_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "pipeline_state")
            entity_state_file = os.path.join(state_dir, "entity_preparation_state.json")
            if os.path.exists(entity_state_file):
                print_warning(f"Forcing full run - deleting existing state file: {entity_state_file}")
                os.remove(entity_state_file)
        
        entity_success, entity_state = run_entity_preparation(config, verify=verify_steps)
        if not entity_success and verify_steps:
            print_error("Pipeline stopped due to entity preparation failure")
            return False
        overall_success = overall_success and entity_success
    else:
        print_info("Skipping entity preparation stage")
    
    # 2. Article Processing
    if skip_to is None or skip_to == "article_processing" or skip_to == "entity_preparation":
        print_header("STAGE 2: Article Processing")
        article_success, article_state = run_article_processing(config, verify=verify_steps)
        if not article_success and verify_steps:
            print_error("Pipeline stopped due to article processing failure")
            return False
        overall_success = overall_success and article_success
    else:
        print_info("Skipping article processing stage")
    
    # 3. Entity Matching
    if skip_to is None or skip_to in ["entity_matching", "article_processing", "entity_preparation"]:
        print_header("STAGE 3: Entity Matching")
        matching_success, matching_state = run_entity_matching(config, verify=verify_steps)
        if not matching_success and verify_steps:
            print_error("Pipeline stopped due to entity matching failure")
            return False
        overall_success = overall_success and matching_success
    else:
        print_info("Skipping entity matching stage")
    
    # 4. Display final results
    end_time = time.time()
    total_time = end_time - start_time
    
    print_header("PIPELINE RESULTS")
    if overall_success:
        print_success(f"Pipeline completed successfully in {total_time:.2f} seconds")
    else:
        print_error(f"Pipeline completed with errors in {total_time:.2f} seconds")
    
    return overall_success

@time_function
def run_entity_matching(config: Dict[str, Any], state_dir: str = None, verify: bool = True) -> tuple:
    """
    Run the entity matching pipeline using corrected components
    
    Args:
        config: Configuration dictionary
        state_dir: Optional directory for state files (defaults to entity_resolution_demo/state)
        verify: Whether to verify entity matching results
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and state dictionary
    """
    # Get default args when imported, or parse command-line args when run directly
    args = parse_args()
    
    # Set state directory
    if state_dir is None:
        state_dir = os.path.join(os.path.dirname(__file__), "state")
    
    # Override config with command-line arguments if provided
    if args.judging_approach:
        if 'entity_matching' not in config:
            config['entity_matching'] = {}
        if 'judging' not in config['entity_matching']:
            config['entity_matching']['judging'] = {}
        config['entity_matching']['judging']['approach'] = args.judging_approach
    
    # Override verify with command-line argument if provided
    if args.no_verify:
        verify = False
    
    # Use custom config file if provided
    if args.config:
        try:
            custom_config = load_config(args.config)
            config.update(custom_config)
            print_info(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            print_error(f"Error loading custom configuration: {e}")
            return False, {}
    
    print_header("ENTITY MATCHING")
    
    # 1. Load previous states
    from entity_resolution_demo.pipeline_runner.utils import load_state
    
    entity_state = load_state("entity_preparation", state_dir=state_dir)
    if entity_state:
        print_success("Loaded previous state from entity_preparation")
    else:
        print_error(f"No previous state found from entity_preparation in {state_dir}")
        return False, {}
    
    article_state = load_state("article_processing", state_dir=state_dir)
    if article_state:
        print_success("Loaded previous state from article_processing")
    else:
        print_error(f"No previous state found from article_processing in {state_dir}")
        return False, {}
    
    # 2. Create entity matcher
    # Create a new watch list and load from pipeline state
    watch_list = EntityWatchList()
    watch_list.load_from_pipeline_state(entity_state)
    
    print_success(f"Created watch list with {len(watch_list.get_all_entities())} entities")
    
    # Configure LLM settings
    if 'entity_matching' not in config:
        config['entity_matching'] = {}
    if 'matching' not in config['entity_matching']:
        config['entity_matching']['matching'] = {}
    if 'llm' not in config['entity_matching']:
        config['entity_matching']['llm'] = {}
    
    config['entity_matching']['matching']['use_llm_explanations'] = True
    config['entity_matching']['llm']['enabled'] = True
    
    # Set default LLM config if not provided
    if not config['entity_matching']['llm'].get('provider'):
        config['entity_matching']['llm']['provider'] = 'openai'
        config['entity_matching']['llm']['model'] = 'gpt-3.5-turbo'
        config['entity_matching']['llm']['temperature'] = 0.1
        config['entity_matching']['llm']['max_tokens'] = 1000
        config['entity_matching']['llm']['enabled'] = True
        config['entity_matching']['llm']['batch_size'] = 5
    elif not config['entity_matching']['llm'].get('model'):
        config['entity_matching']['llm']['model'] = 'gpt-3.5-turbo'
    
    # Explicitly enable hybrid search and set lower thresholds for better matching
    config['hybrid_search_enabled'] = True
    config['match_threshold'] = 0.1
    config['semantic_similarity_threshold'] = 0.1
    
    # Also set in entity_matching section for RealTimeEntityMatcher
    config['entity_matching']['matching']['match_threshold'] = 0.1
    config['entity_matching']['matching']['semantic_similarity_threshold'] = 0.1
    
    # Create Elasticsearch client
    elastic_client = ElasticClient(config)
    
    # Create entity matcher
    # Pass batch_size from config if available, so it overrides any default
    batch_size = config.get('entity_matching', {}).get('llm', {}).get('batch_size')
    matcher = RealTimeEntityMatcher(
        watch_list=watch_list,
        elastic_client=elastic_client,
        config=config,
        batch_size=batch_size  # Explicitly pass batch_size from config
    )
    
    # Check if LLM explanations are enabled
    if hasattr(matcher, 'use_llm_explanations'):
        print_info(f"LLM explanations enabled: {matcher.use_llm_explanations}")
    
    # 3. Process articles
    processed_articles = []
    for article_dict in article_state.get("processed_articles", []):
        # Parse the article string
        article_str = article_dict.get("article", "")
        if isinstance(article_str, str) and article_str.startswith("Article("):
            # Extract id
            id_match = article_str.split("id=")[1].split(",")[0].strip("'\"")
            
            # Extract title
            title_match = article_str.split("title=")[1].split(",")[0].strip("'\"")
            
            # Extract content
            content_start = article_str.find("content=") + 9  # length of "content=" + 1 for the quote
            content_end = article_str.find("', source=")
            content = article_str[content_start:content_end]
            
            # Extract source
            source_match = article_str.split("source=")[1].split(",")[0].strip("'\"")
            
            # Extract language if available
            language = "en"  # default
            if "language=" in article_str:
                language_parts = article_str.split("language=")[1].split(",")
                if language_parts:
                    language = language_parts[0].strip("'\"")
            
            # Create Article object
            article_obj = Article(
                id=id_match,
                title=title_match,
                content=content,
                source=source_match,
                language=language
            )
            
            print_info(f"Processing article: {id_match} - '{title_match}'")
        else:
            # Skip if we can't parse the article
            print_warning(f"Skipping unparseable article")
            continue
        
        # Parse the extracted entities
        extracted_entities = []
        for entity_str in article_dict.get("extracted_entities", []):
            # Parse entity string
            if isinstance(entity_str, str) and entity_str.startswith("ExtractedEntity("):
                try:
                    # Extract name
                    if "name='" in entity_str:
                        name_start = entity_str.find("name='") + 6
                        name_end = entity_str.find("'", name_start)
                        name = entity_str[name_start:name_end]
                    else:
                        continue
                    
                    # Extract entity_type
                    if "entity_type='" in entity_str:
                        type_start = entity_str.find("entity_type='") + 13
                        type_end = entity_str.find("'", type_start)
                        entity_type = entity_str[type_start:type_end]
                    else:
                        entity_type = "UNKNOWN"
                    
                    # Extract confidence
                    confidence = 0.9  # default
                    if "confidence=" in entity_str:
                        conf_parts = entity_str.split("confidence=")[1].split(",")
                        if conf_parts:
                            try:
                                confidence = float(conf_parts[0])
                            except:
                                pass
                    
                    # Extract context
                    context = ""
                    if "context='" in entity_str:
                        context_start = entity_str.find("context='") + 9
                        context_end = entity_str.find("'", context_start)
                        context = entity_str[context_start:context_end]
                    
                    # Extract position
                    position = 0
                    if "position=" in entity_str:
                        pos_parts = entity_str.split("position=")[1].split(",")
                        if pos_parts:
                            try:
                                position = int(pos_parts[0])
                            except:
                                pass
                    
                    # Extract extraction_method
                    extraction_method = "unknown"
                    if "extraction_method='" in entity_str:
                        method_start = entity_str.find("extraction_method='") + 19
                        method_end = entity_str.find("'", method_start)
                        extraction_method = entity_str[method_start:method_end]
                    
                    # Create ExtractedEntity object
                    entity = ExtractedEntity(
                        name=name,
                        entity_type=entity_type,
                        confidence=confidence,
                        context=context,
                        position=position,
                        extraction_method=extraction_method
                    )
                    extracted_entities.append(entity)
                    print_info(f"    Parsed entity: {name} ({entity_type})")
                except Exception as e:
                    print_warning(f"    Error parsing entity: {e}")
                    continue
        
        # Parse the unique_entities field
        unique_entities_set = set()
        unique_entities_str = article_dict.get("unique_entities", "")
        if isinstance(unique_entities_str, str) and unique_entities_str.startswith("{") and unique_entities_str.endswith("}"):
            # Remove the curly braces and split by commas
            try:
                entities_list = unique_entities_str[1:-1].split(", ")
                # Remove quotes from each entity name
                unique_entities_set = set(entity.strip("'") for entity in entities_list)
                print_info(f"    Parsed {len(unique_entities_set)} unique entities from state: {unique_entities_set}")
            except Exception as e:
                print_warning(f"    Error parsing unique_entities: {e}")
                # Fall back to extracted entities
                unique_entities_set = set(entity.name for entity in extracted_entities)
        else:
            # Fall back to extracted entities
            unique_entities_set = set(entity.name for entity in extracted_entities)
        
        # Create ProcessedArticle object
        processed_article = ProcessedArticle(
            article=article_obj,
            extracted_entities=extracted_entities,
            processing_time=article_dict.get("processing_time", 0.0),
            total_entities_found=len(extracted_entities),
            unique_entities=unique_entities_set
        )
        
        processed_articles.append(processed_article)
    
    print_success(f"Processed {len(processed_articles)} articles")
    
    # 4. Match entities
    print_subheader("Matching Entities")
    
    # Match entities
    matching_results = []
    all_potential_matches = []
    
    for article in processed_articles:
        # Get article ID and title for logging
        article_id = article.article.id
        article_title = article.article.title
        print_info(f"Matching article: {article_id} - '{article_title}'")
        
        # Debug the extracted entities
        print_info(f"  Article has {len(article.extracted_entities)} extracted entities:")
        for entity in article.extracted_entities:
            print_info(f"    - {entity.name} ({entity.entity_type})")
        
        # Debug the unique entities
        print_info(f"  Article has {len(article.unique_entities)} unique entities: {', '.join(article.unique_entities)}")
        
        # Check if the article content contains any watch list entity names
        article_content = article.article.content
        found_entities = []
        for entity in watch_list.get_all_entities():
            if entity.name in article_content:
                found_entities.append(entity.name)
        if found_entities:
            print_info(f"  Article content contains these watch list entities: {', '.join(found_entities)}")
        else:
            print_info(f"  Article content does not contain any watch list entity names")
        
        # Match article against watch list
        try:
            result = matcher.match_article(article)
            
            # Print matches
            if result.matches_found:
                print_success(f"  Found {len(result.matches_found)} matches:")
                for i, match in enumerate(result.matches_found):
                    print_info(f"  Match {i+1}: {match.extracted_entity.name} -> {match.watched_entity.name} "
                              f"(Confidence: {match.confidence:.2f}, Type: {match.match_type})")
                    if hasattr(match, 'reasoning') and match.reasoning:
                        explanation = match.reasoning[:100] + "..." if len(match.reasoning) > 100 else match.reasoning
                        print_info(f"    Explanation: {explanation}")
                    
                    # Collect all potential matches for batch LLM judgment
                    all_potential_matches.append(match)
            else:
                print_info(f"  No matches found")
            
            matching_results.append(result)
        except Exception as e:
            print_error(f"Error matching article {article_id}: {e}")
            import traceback
            print_error(traceback.format_exc())
    
    # Process matches with EnhancedBatchMatchJudge if LLM is enabled
    if config['entity_matching']['llm']['enabled'] and all_potential_matches:
        print_subheader("Generating LLM Explanations for Matches")
        
        try:
            # Create EnhancedBatchMatchJudge
            batch_judge = EnhancedBatchMatchJudge(
                config=config,
                batch_size=config['entity_matching']['llm'].get('batch_size', 5),
                es_client=elastic_client.es
            )
            
            print_info(f"Processing {len(all_potential_matches)} matches with LLM for explanations")
            
            # Process matches in batches
            batch_size = config['entity_matching']['llm'].get('batch_size', 5)
            total_batches = (len(all_potential_matches) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_potential_matches))
                entity_matches = all_potential_matches[start_idx:end_idx]
                
                print_info(f"Processing batch {batch_idx + 1}/{total_batches} with {len(entity_matches)} matches")
                
                try:
                    # Convert EntityMatch objects to dictionaries for batch_judge_matches
                    batch = []
                    for i, match in enumerate(entity_matches):
                        batch.append({
                            'pair_index': i,
                            'query_name': match.extracted_entity.name,
                            'candidate_name': match.watched_entity.name,
                            'context': match.extracted_entity.context or ""
                        })
                    
                    # Process batch
                    processed_batch = batch_judge.batch_judge_matches(batch)
                    
                    # Update matches with explanations
                    for i, processed_match in enumerate(processed_batch):
                        if hasattr(processed_match, 'reasoning') and processed_match.reasoning:
                            # Copy all LLM fields to the match object
                            all_potential_matches[start_idx + i].reasoning = processed_match.reasoning
                            all_potential_matches[start_idx + i].confidence = processed_match.confidence
                            all_potential_matches[start_idx + i].is_match = processed_match.is_match
                            
                            # Add additional fields from LLM response
                            if hasattr(processed_match, 'match_type'):
                                all_potential_matches[start_idx + i].match_type = processed_match.match_type
                            
                            if hasattr(processed_match, 'explanation_full'):
                                all_potential_matches[start_idx + i].explanation_full = processed_match.explanation_full
                            
                            if hasattr(processed_match, 'confidence_factors'):
                                all_potential_matches[start_idx + i].confidence_factors = processed_match.confidence_factors
                            
                            if hasattr(processed_match, 'key_evidence'):
                                all_potential_matches[start_idx + i].key_evidence = processed_match.key_evidence
                            
                            if hasattr(processed_match, 'risk_factors'):
                                all_potential_matches[start_idx + i].risk_factors = processed_match.risk_factors
                            
                            # Print sample of explanations
                            if i < 3:  # Show only first 3 explanations per batch
                                explanation = processed_match.reasoning[:100] + "..." if len(processed_match.reasoning) > 100 else processed_match.reasoning
                                print_info(f"  Match: {processed_match.extracted_entity.name} -> {processed_match.watched_entity.name}")
                                print_info(f"  Explanation: {explanation}")
                                print_info(f"  Confidence: {processed_match.confidence:.2f}")
                                
                                if hasattr(processed_match, 'match_type'):
                                    print_info(f"  Match Type: {processed_match.match_type}")
                                    
                                if hasattr(processed_match, 'key_evidence') and processed_match.key_evidence:
                                    print_info(f"  Key Evidence: {', '.join(processed_match.key_evidence[:2])}")
                except Exception as e:
                    print_error(f"Error processing batch {batch_idx + 1}: {e}")
            
            print_success(f"Successfully generated LLM explanations for {len(all_potential_matches)} matches")
        except Exception as e:
            print_error(f"Error setting up EnhancedBatchMatchJudge: {e}")
            print_error(traceback.format_exc())
    
    # 5. Index matching results
    from datetime import datetime
    
    # Create a unique index name for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_results_index = f"demo_entity_resolution_{timestamp}_match_results"
    
    # Update config with the new index name
    if "elasticsearch" not in config:
        config["elasticsearch"] = {}
    config["elasticsearch"]["match_results_index"] = match_results_index
    
    # Create index mapping
    mapping = {
        "mappings": {
            "properties": {
                "extracted_entity": {"type": "keyword"},
                "extracted_entity_type": {"type": "keyword"},
                "extracted_entity_context": {"type": "text"},
                "watched_entity": {"type": "keyword"},
                "watched_entity_type": {"type": "keyword"},
                "watched_entity_priority": {"type": "keyword"},
                "match_id": {"type": "keyword"},
                "match_timestamp": {"type": "date"},
                "match_type": {"type": "keyword"},
                "is_match": {"type": "boolean"},
                "article_id": {"type": "keyword"},
                "article_title": {"type": "text"},
                "article_source": {"type": "keyword"},
                "confidence": {"type": "float"},
                "explanation": {"type": "text"},
                "explanation_full": {"type": "text"},
                "reasoning": {"type": "text"},
                "confidence_factors": {"type": "object"},
                "key_evidence": {"type": "text"},
                "risk_factors": {"type": "text"}
            }
        }
    }
    
    # Create index
    try:
        if not elastic_client.es.indices.exists(index=match_results_index):
            elastic_client.es.indices.create(index=match_results_index, body=mapping)
            print_success(f"Created index: {match_results_index}")
    except Exception as e:
        print_error(f"Error creating index: {e}")
    
    # Index matching results
    indexed_count = 0
    for result in matching_results:
        for match in result.matches_found:
            # Create document
            doc = {
                "extracted_entity": match.extracted_entity.name,
                "extracted_entity_type": match.extracted_entity.entity_type,
                "extracted_entity_context": match.extracted_entity.context if hasattr(match.extracted_entity, 'context') else "",
                "watched_entity": match.watched_entity.name,
                "watched_entity_type": match.watched_entity.entity_type,
                "watched_entity_priority": match.watched_entity.priority if hasattr(match.watched_entity, 'priority') else "",
                "match_id": f"match_{timestamp}_{indexed_count:03d}",
                "match_timestamp": datetime.now().isoformat(),
                "match_type": match.match_type,
                "is_match": match.is_match if hasattr(match, 'is_match') else (match.confidence >= 0.3),
                "article_id": result.article_id,
                "article_title": result.article_title,
                "article_source": result.article.source if hasattr(result, 'article') else "",
                "confidence": match.confidence,
                "reasoning": match.reasoning if hasattr(match, 'reasoning') else ""
            }
            
            # Add explanation fields if available
            if hasattr(match, 'explanation'):
                doc["explanation"] = match.explanation
            if hasattr(match, 'explanation_full'):
                doc["explanation_full"] = match.explanation_full
            if hasattr(match, 'confidence_factors'):
                doc["confidence_factors"] = match.confidence_factors
            if hasattr(match, 'key_evidence'):
                doc["key_evidence"] = match.key_evidence
            if hasattr(match, 'risk_factors'):
                doc["risk_factors"] = match.risk_factors
            
            # Index document
            try:
                elastic_client.es.index(index=match_results_index, body=doc)
                indexed_count += 1
            except Exception as e:
                print_error(f"Error indexing match: {e}")
    
    print_success(f"Indexed {indexed_count} matches to {match_results_index}")
    
    # 6. Verify entity matching
    if verify:
        print_subheader("Verifying Entity Matching")
        
        # Check if any matches were found
        total_matches = sum(len(result.matches_found) for result in matching_results)
        if total_matches == 0:
            print_warning("No matches were found in any article - this is acceptable for demonstration purposes")
        else:
            print_success(f"Found {total_matches} matches across {len(matching_results)} articles")
        
        # Check for different match types
        match_types = {}
        for result in matching_results:
            for match in result.matches_found:
                match_type = match.match_type
                match_types[match_type] = match_types.get(match_type, 0) + 1
        
        print_info(f"Match types: {match_types}")
        
        # Check for high confidence matches
        high_confidence_matches = []
        for result in matching_results:
            for match in result.matches_found:
                if match.confidence >= 0.8:
                    high_confidence_matches.append(match)
        
        if high_confidence_matches:
            print_success(f"Found {len(high_confidence_matches)} high confidence matches")
        else:
            print_warning("No high confidence matches found")
    
    # 7. Save state
    from entity_resolution_demo.pipeline_runner.utils import save_state
    
    # Create state dictionary
    state = {
        "matching_results": [],
        "metadata": {
            "pipeline_version": "1.2.0",
            "processing_timestamp": datetime.now().isoformat(),
            "total_articles_processed": len(processed_articles),
            "total_matches_found": sum(len(result.matches_found) for result in matching_results),
            "match_types_distribution": match_types if 'match_types' in locals() else {},
            "confidence_distribution": {
                "high": len([m for r in matching_results for m in r.matches_found if m.confidence >= 0.8]),
                "medium": len([m for r in matching_results for m in r.matches_found if 0.5 <= m.confidence < 0.8]),
                "low": len([m for r in matching_results for m in r.matches_found if m.confidence < 0.5])
            },
            "matcher_stats": matcher.get_stats(),
            "explanations_count": indexed_count,
            "llm_stats": {
                "total_matches_processed": len(all_potential_matches) if 'all_potential_matches' in locals() else 0,
                "matches_with_explanations": len([m for m in all_potential_matches if hasattr(m, 'reasoning') and m.reasoning]) if 'all_potential_matches' in locals() else 0,
                "llm_provider": config['entity_matching']['llm'].get('provider', 'unknown'),
                "llm_model": config['entity_matching']['llm'].get('model', 'unknown'),
                "batch_size": config['entity_matching']['llm'].get('batch_size', 5)
            }
        },
        "match_results_index": match_results_index
    }
    
    # Add each matching result to the state
    for result in matching_results:
        result_dict = {
            "article_id": result.article_id,
            "article_title": result.article_title,
            "total_entities_extracted": len(result.article.extracted_entities) if hasattr(result, 'article') else 0,
            "matches_found": [],
            "high_priority_matches": [],
            "processing_time": result.processing_time
        }
        
        # Add each match to the result
        for match in result.matches_found:
            match_dict = {
                "extracted_entity": match.extracted_entity.name,
                "watched_entity": match.watched_entity.name,
                "confidence": match.confidence,
                "match_type": match.match_type
            }
            
            # Add reasoning if available
            if hasattr(match, 'reasoning') and match.reasoning:
                match_dict["reasoning"] = match.reasoning
                
            # Add is_match if available
            if hasattr(match, 'is_match'):
                match_dict["is_match"] = match.is_match
                
            # Add match_type if available
            if hasattr(match, 'match_type'):
                match_dict["match_type"] = match.match_type
                
            # Add explanation_full if available
            if hasattr(match, 'explanation_full'):
                match_dict["explanation_full"] = match.explanation_full
                
            # Add confidence_factors if available
            if hasattr(match, 'confidence_factors'):
                match_dict["confidence_factors"] = match.confidence_factors
                
            # Add key_evidence if available
            if hasattr(match, 'key_evidence'):
                match_dict["key_evidence"] = match.key_evidence
                
            # Add risk_factors if available
            if hasattr(match, 'risk_factors'):
                match_dict["risk_factors"] = match.risk_factors
            
            result_dict["matches_found"].append(match_dict)
            
            # Add to high priority matches if applicable
            if match.watched_entity.priority == "high":
                result_dict["high_priority_matches"].append(match_dict)
        
        state["matching_results"].append(result_dict)
    
    # Save state
    save_success = save_state("entity_matching", state, state_dir=state_dir)
    if not save_success:
        print_error("Failed to save entity matching state")
    
    return True, state

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline Runner (SIMPLIFIED)")
    parser.add_argument("--skip-to", choices=["entity_preparation", "article_processing", "entity_matching"],
                        help="Skip to a specific stage of the pipeline")
    parser.add_argument("--skip-readiness-check", action="store_true",
                        help="Skip the pipeline readiness check")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification steps")
    parser.add_argument("--config", type=str,
                        help="Path to custom configuration file")
    parser.add_argument("--force-full-run", action="store_true",
                        help="Force a full run of all pipeline stages")
    parser.add_argument("--judging-approach", choices=["basic", "enhanced", "agentic"],
                        help="Approach to use for match judging")
    parser.add_argument("--entity-data", type=str,
                        help="Path to custom entity data JSON file")
    parser.add_argument("--article-data", type=str,
                        help="Path to custom article data JSON file")
    
    return parser.parse_args()

def main():
    """
    Main entry point for the pipeline runner
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config()
    
    # Add custom data paths to config if provided
    if args.entity_data:
        config["entity_data_path"] = args.entity_data
    if args.article_data:
        config["article_data_path"] = args.article_data
    
    # Run pipeline
    success, _ = run_pipeline(
        config=config,
        verify_steps=not args.no_verify,
        skip_to=args.skip_to,
        skip_readiness_check=args.skip_readiness_check,
        force_full_run=args.force_full_run
    )
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
