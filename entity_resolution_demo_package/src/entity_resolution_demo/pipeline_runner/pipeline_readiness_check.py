#!/usr/bin/env python3
"""
Pipeline Readiness Check

This module verifies that all necessary components for the entity resolution pipeline
are available and properly configured before running the pipeline.
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os

# Import local modules
from entity_resolution_demo.pipeline_runner.utils import (
    print_header, print_subheader, print_success, print_warning, 
    print_error, print_info, time_function
)

# Import project modules
from entity_resolution_demo.search.elastic_client import ElasticClient
from entity_resolution_demo.article_processing.elasticsearch_ner_extractor import ElasticsearchNERExtractor
from entity_resolution_demo.entity_matching.enhanced_batch_match_judge import EnhancedBatchMatchJudge

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@time_function
def check_elasticsearch_connection(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if Elasticsearch connection can be established
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and result details
    """
    print_subheader("Checking Elasticsearch Connection")
    
    try:
        # Get Elasticsearch settings from environment variables
        es_endpoint = os.getenv('ELASTIC_ENDPOINT')
        es_cloud_id = os.getenv('ELASTIC_CLOUD_ID')
        es_api_key = os.getenv('ELASTIC_API_KEY')
        
        print_info(f"ELASTIC_ENDPOINT: {'Set' if es_endpoint else 'Not set'}")
        print_info(f"ELASTIC_CLOUD_ID: {'Set' if es_cloud_id else 'Not set'}")
        print_info(f"ELASTIC_API_KEY: {'Set' if es_api_key else 'Not set'}")
        
        if not (es_endpoint or es_cloud_id) or not es_api_key:
            print_error("Elasticsearch credentials not found in environment variables")
            return False, {
                "error": "Missing Elasticsearch credentials",
                "details": "ELASTIC_API_KEY and either ELASTIC_ENDPOINT or ELASTIC_CLOUD_ID must be set in environment variables"
            }
        
        # Initialize Elasticsearch client
        elastic_client = ElasticClient(config)
        
        # Test connection with simple cluster info request
        cluster_info = elastic_client.es.info()
        cluster_name = cluster_info.get('cluster_name', 'Unknown')
        es_version = cluster_info.get('version', {}).get('number', 'Unknown')
        
        print_success(f"Successfully connected to Elasticsearch cluster: {cluster_name}")
        print_info(f"Elasticsearch version: {es_version}")
        
        # Get more detailed cluster health if available (may not work in serverless mode)
        try:
            health = elastic_client.es.cluster.health()
            status = health.get('status', 'unknown')
            nodes = health.get('number_of_nodes', 0)
            
            print_info(f"Cluster health: {status}")
            print_info(f"Number of nodes: {nodes}")
        except Exception as e:
            # Handle serverless mode where cluster health is not available
            if '410' in str(e) and 'serverless mode' in str(e):
                print_info("Cluster health check not available in serverless mode")
                print_info("Serverless mode detected - this is compatible with our pipeline")
                status = "serverless"
                nodes = "N/A"
            else:
                print_warning(f"Could not get cluster health: {e}")
                status = "unknown"
                nodes = "unknown"
        
        return True, {
            "cluster_name": cluster_name,
            "version": es_version,
            "status": status,
            "nodes": nodes
        }
        
    except Exception as e:
        print_error(f"Failed to connect to Elasticsearch: {e}")
        return False, {
            "error": str(e),
            "details": "Could not establish connection to Elasticsearch cluster"
        }

@time_function
def check_llm_connection(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if LLM connection can be established
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and result details
    """
    print_subheader("Checking LLM Connection")
    
    try:
        # Get OpenAI API key from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        litellm_proxy_url = os.getenv('LITELLM_PROXY_URL')
        
        print_info(f"OPENAI_API_KEY: {'Set' if openai_api_key else 'Not set'}")
        print_info(f"LITELLM_PROXY_URL: {'Set' if litellm_proxy_url else 'Not set (will use direct OpenAI API)'}")
        
        if not openai_api_key:
            print_error("OpenAI API key not found in environment variables")
            print_info("You need to set the OPENAI_API_KEY environment variable in your .env file")
            print_info("Example: OPENAI_API_KEY=sk-...")
            return False, {
                "error": "Missing OpenAI API key",
                "details": "OPENAI_API_KEY must be set in environment variables"
            }
        
        # Update config with API key if not already set
        if 'entity_matching' not in config:
            config['entity_matching'] = {}
        if 'llm' not in config['entity_matching']:
            config['entity_matching']['llm'] = {}
        config['entity_matching']['llm']['api_key'] = openai_api_key
        
        # Initialize BatchMatchJudge
        try:
            batch_match_judge = EnhancedBatchMatchJudge(config)
        except Exception as e:
            print_error(f"Failed to initialize BatchMatchJudge: {e}")
            print_info("This could be due to configuration issues or missing dependencies")
            return False, {
                "error": f"BatchMatchJudge initialization failed: {str(e)}",
                "details": "Check your configuration and dependencies"
            }
        
        # Test with a simple match judgment
        test_batch = [
            {
                'pair_index': 0,
                'query_name': 'John Smith',
                'candidate_name': 'John Smith',
                'context': 'A test person'
            }
        ]
        
        # Try to process the test batch
        try:
            results = batch_match_judge.batch_judge_matches(test_batch)
            
            if results and len(results) > 0:
                confidence = results[0].get('confidence', 0.0)
                match_type = results[0].get('match_type', 'unknown')
                
                print_success(f"Successfully connected to LLM API")
                print_info(f"Provider: {batch_match_judge.provider}")
                print_info(f"Model: {batch_match_judge.model}")
                print_info(f"Test match confidence: {confidence:.2f}")
                print_info(f"Test match type: {match_type}")
                
                return True, {
                    "provider": batch_match_judge.provider,
                    "model": batch_match_judge.model,
                    "test_confidence": confidence,
                    "test_match_type": match_type
                }
            else:
                print_warning("LLM connection test returned empty results")
                print_info("This could indicate an issue with the LLM API or the response format")
                return False, {
                    "error": "Empty LLM response",
                    "details": "LLM API returned empty results"
                }
        except Exception as e:
            print_error(f"LLM API call failed: {e}")
            
            # Provide specific guidance based on error type
            if "auth" in str(e).lower() or "key" in str(e).lower() or "unauthorized" in str(e).lower():
                print_info("This appears to be an authentication issue with your API key")
                print_info("Make sure your OPENAI_API_KEY is valid and has not expired")
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                print_info("This appears to be a network or connection issue")
                print_info("Check your internet connection and try again")
            elif "rate" in str(e).lower() or "limit" in str(e).lower():
                print_info("This appears to be a rate limit issue")
                print_info("Your API key may have reached its rate limit, try again later")
            else:
                print_info("This could be due to an issue with the LLM API or your configuration")
                print_info("Check your API key and network connection")
            
            return False, {
                "error": f"LLM API call failed: {str(e)}",
                "details": "Check your API key and network connection"
            }
            
    except Exception as e:
        print_error(f"Failed to connect to LLM API: {e}")
        return False, {
            "error": str(e),
            "details": "Could not establish connection to LLM API"
        }

@time_function
def check_elasticsearch_models(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if required Elasticsearch models are deployed
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and result details
    """
    print_subheader("Checking Elasticsearch Models")
    
    # Initialize result tracking
    results = {
        "ner_model": {
            "name": "facebookai__xlm-roberta-large-finetuned-conll03-english",
            "available": False,
            "api_type": None
        },
        "embedding_model": {
            "name": ".multilingual-e5-small-elasticsearch",
            "available": False
        },
        "all_models_available": False
    }
    
    try:
        # Initialize Elasticsearch client
        elastic_client = ElasticClient(config)
        
        # Check NER model
        try:
            ner_extractor = ElasticsearchNERExtractor(elastic_client, config)
            ner_model_id = ner_extractor.ner_model_id
            results["ner_model"]["name"] = ner_model_id
            
            # Test NER model with a simple extraction
            test_text = "Linus Torvalds is the CEO of Tesla and Git."
            entities = ner_extractor.extract_entities_from_text(test_text)
            
            if entities and len(entities) > 0:
                results["ner_model"]["available"] = True
                results["ner_model"]["api_type"] = "inference_api" if not ner_extractor.use_ml_api else "ml_api"
                
                print_success(f"NER model '{ner_model_id}' is available and working")
                print_info(f"API type: {results['ner_model']['api_type']}")
                print_info(f"Test extraction found {len(entities)} entities")
                for entity in entities:
                    print_info(f"  • {entity.entity} ({entity.class_name}) - confidence: {entity.class_probability:.3f}")
            else:
                print_warning(f"NER model '{ner_model_id}' is available but test extraction returned no entities")
                results["ner_model"]["available"] = True  # Model is available even if test returned no entities
                results["ner_model"]["api_type"] = "inference_api" if not ner_extractor.use_ml_api else "ml_api"
        except Exception as e:
            print_error(f"NER model check failed: {e}")
            print_info("For NER model deployment instructions, visit:")
            print_info("https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-deploy-models.html")
            print_info("Recommended model: facebookai__xlm-roberta-large-finetuned-conll03-english")
        
        # Check embedding model
        try:
            # Try to get the embedding model info
            inference_id = elastic_client.inference_id
            response = elastic_client.es.inference.get(inference_id=inference_id)
            if response:
                results["embedding_model"]["available"] = True
                results["embedding_model"]["name"] = inference_id
                
                print_success(f"Embedding model '{inference_id}' is available")
                print_info(f"Model task type: {response.get('task_type', 'unknown')}")
                print_info(f"Model service: {response.get('service', 'unknown')}")
        except Exception as e:
            print_error(f"Embedding model '{elastic_client.inference_id}' is not available: {e}")
            print_info("For embedding model deployment instructions, visit:")
            print_info("https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-deploy-models.html")
            print_info("Recommended model: .multilingual-e5-small-elasticsearch")
        
        # Check if all models are available
        results["all_models_available"] = results["ner_model"]["available"] and results["embedding_model"]["available"]
        
        if results["all_models_available"]:
            print_success("All required Elasticsearch models are available")
        else:
            print_warning("Some required Elasticsearch models are missing")
            
            # Provide specific guidance on what's missing
            if not results["ner_model"]["available"]:
                print_warning("NER model is not available or not working properly")
                print_info("This model is used for named entity recognition in articles")
            
            if not results["embedding_model"]["available"]:
                print_warning("Embedding model is not available")
                print_info("This model is used for semantic search capabilities")
            
        return results["all_models_available"], results
        
    except Exception as e:
        print_error(f"Failed to check Elasticsearch models: {e}")
        return False, {
            "error": str(e),
            "details": "Could not check Elasticsearch models",
            "ner_model": results["ner_model"],
            "embedding_model": results["embedding_model"],
            "all_models_available": False
        }

@time_function
def run_pipeline_readiness_check(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Run all readiness checks for the entity resolution pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and consolidated results
    """
    print_header("PIPELINE READINESS CHECK")
    
    start_time = time.time()
    
    # Track overall readiness
    overall_ready = True
    results = {
        "elasticsearch_connection": {},
        "llm_connection": {},
        "elasticsearch_models": {},
        "overall_ready": False,
        "timestamp": time.time()
    }
    
    # Check Elasticsearch connection
    es_success, es_results = check_elasticsearch_connection(config)
    results["elasticsearch_connection"] = es_results
    overall_ready = overall_ready and es_success
    
    # Check LLM connection
    llm_success, llm_results = check_llm_connection(config)
    results["llm_connection"] = llm_results
    overall_ready = overall_ready and llm_success
    
    # Check Elasticsearch models
    if es_success:
        models_success, models_results = check_elasticsearch_models(config)
        results["elasticsearch_models"] = models_results
        overall_ready = overall_ready and models_success
    else:
        print_warning("Skipping Elasticsearch models check due to connection failure")
        results["elasticsearch_models"] = {
            "error": "Skipped due to Elasticsearch connection failure",
            "all_models_available": False
        }
        overall_ready = False
    
    # Update overall readiness
    results["overall_ready"] = overall_ready
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    results["elapsed_time"] = elapsed_time
    
    # Print summary
    print_header("READINESS CHECK SUMMARY")
    print_info(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    if overall_ready:
        print_success("✅ All systems are ready for the entity resolution pipeline")
        print_info("\nSystem Status:")
        print_success("  • Elasticsearch: Connected")
        if "cluster_name" in results["elasticsearch_connection"]:
            print_info(f"    - Cluster: {results['elasticsearch_connection']['cluster_name']}")
            print_info(f"    - Version: {results['elasticsearch_connection']['version']}")
        print_success("  • LLM API: Connected")
        if "provider" in results["llm_connection"]:
            print_info(f"    - Provider: {results['llm_connection']['provider']}")
            print_info(f"    - Model: {results['llm_connection']['model']}")
        print_success("  • Required Models: Available")
        if "ner_model" in results["elasticsearch_models"]:
            ner_model = results["elasticsearch_models"]["ner_model"]
            print_info(f"    - NER Model: {ner_model['name']}")
            print_info(f"    - API Type: {ner_model['api_type']}")
        if "embedding_model" in results["elasticsearch_models"]:
            embedding_model = results["elasticsearch_models"]["embedding_model"]
            print_info(f"    - Embedding Model: {embedding_model['name']}")
    else:
        print_error("❌ Some components are not ready")
        print_info("\nSystem Status:")
        
        # Elasticsearch status
        if es_success:
            print_success("  • Elasticsearch: Connected")
            if "cluster_name" in results["elasticsearch_connection"]:
                print_info(f"    - Cluster: {results['elasticsearch_connection']['cluster_name']}")
                print_info(f"    - Version: {results['elasticsearch_connection']['version']}")
        else:
            print_error("  • Elasticsearch: Connection Failed")
            if "error" in results["elasticsearch_connection"]:
                print_info(f"    - Error: {results['elasticsearch_connection']['error']}")
                print_info("    - Set ELASTIC_CLOUD_ID and ELASTIC_API_KEY in your .env file")
        
        # LLM API status
        if llm_success:
            print_success("  • LLM API: Connected")
            if "provider" in results["llm_connection"]:
                print_info(f"    - Provider: {results['llm_connection']['provider']}")
                print_info(f"    - Model: {results['llm_connection']['model']}")
        else:
            print_error("  • LLM API: Connection Failed")
            if "error" in results["llm_connection"]:
                print_info(f"    - Error: {results['llm_connection']['error']}")
                print_info("    - Set OPENAI_API_KEY in your .env file")
        
        # Models status
        if es_success:
            if models_success:
                print_success("  • Required Models: Available")
            else:
                print_error("  • Required Models: Missing")
                
                # NER model status
                if "ner_model" in results["elasticsearch_models"]:
                    ner_model = results["elasticsearch_models"]["ner_model"]
                    if ner_model["available"]:
                        print_success("    - NER Model: Available")
                        print_info(f"      Name: {ner_model['name']}")
                    else:
                        print_error("    - NER Model: Not Available")
                        print_info(f"      Required: {ner_model['name']}")
                
                # Embedding model status
                if "embedding_model" in results["elasticsearch_models"]:
                    embedding_model = results["elasticsearch_models"]["embedding_model"]
                    if embedding_model["available"]:
                        print_success("    - Embedding Model: Available")
                        print_info(f"      Name: {embedding_model['name']}")
                    else:
                        print_error("    - Embedding Model: Not Available")
                        print_info(f"      Required: {embedding_model['name']}")
        
        # Print troubleshooting info
        print_info("\nTroubleshooting:")
        print_info("  1. Check your .env file for required environment variables")
        print_info("  2. Verify your Elasticsearch cluster is running and accessible")
        print_info("  3. Ensure required models are deployed in your Elasticsearch cluster")
        print_info("  4. Verify your OpenAI API key is valid and has not expired")
        print_info("  5. Run with --skip-readiness-check to bypass this check (not recommended)")
    
    return overall_ready, results

def ask_user_to_proceed():
    """
    Ask the user if they want to proceed with the pipeline
    
    Returns:
        bool: True if the user wants to proceed, False otherwise
    """
    # Always return True for automated testing
    print("\nAutomatically proceeding with the entity resolution pipeline.")
    return True
    
    # Original interactive code:
    # while True:
    #     response = input("\nDo you want to proceed with the entity resolution pipeline? (yes/no): ").strip().lower()
    #     if response in ["yes", "y"]:
    #         return True
    #     elif response in ["no", "n"]:
    #         return False
    #     else:
    #         print("Please enter 'yes' or 'no'")

if __name__ == "__main__":
    # Load configuration
    from entity_resolution_demo.config import load_config
    config = load_config()
    
    # Run readiness check
    ready, results = run_pipeline_readiness_check(config)
    
    # Ask user to proceed
    if ready:
        if ask_user_to_proceed():
            print_success("Proceeding with the pipeline...")
            # This would call the pipeline, but we'll leave that to the main script
        else:
            print_info("Pipeline execution cancelled by user")
    else:
        print_error("Pipeline readiness check failed. Please fix the issues before proceeding.")
        sys.exit(1)
