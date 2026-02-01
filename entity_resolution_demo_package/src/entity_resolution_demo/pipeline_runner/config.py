#!/usr/bin/env python3
"""
Configuration for Entity Resolution Pipeline Demo

Centralizes all configuration options for the entity resolution pipeline demonstration.
"""

import os
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# State directory for saving intermediate results
STATE_DIR = Path(__file__).parent / "state"
STATE_DIR.mkdir(exist_ok=True)

# Default configuration
from datetime import datetime

# Generate timestamp for index names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

DEFAULT_CONFIG = {
    # Elasticsearch configuration
    "elasticsearch": {
        "use_cloud": True,
        # Support either ELASTIC_ENDPOINT or ELASTIC_CLOUD_ID
        "endpoint": os.environ.get("ELASTIC_ENDPOINT", ""),
        "cloud_id": os.environ.get("ELASTIC_CLOUD_ID", ""),
        "api_key": os.environ.get("ELASTIC_API_KEY", ""),
        "index_prefix": f"demo_entity_resolution_{timestamp}_",
        "entity_index": f"demo_entity_resolution_{timestamp}_entities",
        "processed_articles_index": f"demo_entity_resolution_{timestamp}_articles",
        "match_results_index": f"demo_entity_resolution_{timestamp}_match_results",
        "recreate_indices": False
    },
    
    # LLM configuration at top level for compatibility with EnhancedBatchMatchJudge
    "llm": {
        "provider": "openai",
        "model": "gpt-4.1",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "max_tokens": 1000,
        "temperature": 0.0,
        "batch_size": 5
    },
    
    # Entity preparation configuration - references entity_data.py
    "entity_preparation": {
        "data_file": None,  # Set to a file path to load from a file instead of entity_data.py
        "use_default_data": True  # Set to False to use only data from data_file
    },
    
    # Article processing configuration - references article_data.py
    "article_processing": {
        "data_file": None,  # Set to a file path to load from a file instead of article_data.py
        "use_default_data": True  # Set to False to use only data from data_file
    },
    
    # Match judgment configuration
    "match_judgment": {
        "use_enhanced_prompt": True,  # Use the enhanced prompt for better explanations
        "cache_judgments": True,      # Cache judgments to avoid redundant LLM calls
        "cache_ttl": 3600            # Cache time-to-live in seconds
    },
    
    # Entity matching configuration
    "entity_matching": {
        "matching": {
            "exact_match_threshold": 1.0,
            "semantic_match_threshold": 0.6,  # Lowered threshold for more semantic matches
            "hybrid_search_enabled": True,
            "use_llm_explanations": True,
            "use_enhanced_prompt": True,  # Enable enhanced prompts for better explanations
            "contextual_match_threshold": 0.5,  # Lowered threshold for more contextual matches
            "enable_role_based_matching": True,  # Explicitly enable role-based matching
            "skip_exact_matches": False  # Don't skip semantic/contextual matching when exact match is found
        },
        "llm": {
            "enabled": True,
            "provider": "openai",
            "model": "gpt-4.1",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "max_tokens": 1000,
            "temperature": 0.0,
            "batch_size": 5
        }
    }
}

def load_config(config_path=None):
    """
    Load configuration from file or use default
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Deep merge the configurations
                _deep_merge(config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    # Check for required environment variables
    if not (config["elasticsearch"].get("endpoint") or config["elasticsearch"].get("cloud_id")):
        logger.warning("Neither ELASTIC_ENDPOINT nor ELASTIC_CLOUD_ID environment variable is set")
    
    if not config["elasticsearch"]["api_key"]:
        logger.warning("ELASTIC_API_KEY environment variable not set")
    
    # Check if LLM explanations are enabled and API key is set
    if config["entity_matching"].get("matching", {}).get("use_llm_explanations", False) and not config["entity_matching"]["llm"]["api_key"]:
        logger.warning("OPENAI_API_KEY environment variable not set, LLM explanations will not work")
    
    return config

def _deep_merge(base, update):
    """Deep merge two dictionaries"""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
