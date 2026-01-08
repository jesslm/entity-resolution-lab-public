"""
Notebook helper functions for the entity resolution tutorial.

This module contains utility functions to keep notebook cells clean and focused.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import spacy
from spacy import displacy


def load_and_preview_data(entities_path: Path, articles_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preview entities and articles data."""
    # Load entities
    with open(entities_path, 'r') as f:
        entities_data = json.load(f)
    
    # Handle nested structure - extract entities array
    if isinstance(entities_data, dict) and 'entities' in entities_data:
        entities_list = entities_data['entities']
    else:
        entities_list = entities_data
    
    entities_df = pd.DataFrame(entities_list)
    
    # Load articles
    with open(articles_path, 'r') as f:
        articles_data = json.load(f)
    
    # Handle nested structure - extract articles array
    if isinstance(articles_data, dict) and 'articles' in articles_data:
        articles_list = articles_data['articles']
    else:
        articles_list = articles_data
        
    articles_df = pd.DataFrame(articles_list)
    
    return entities_df, articles_df


def create_interactive_widgets():
    """Create interactive widgets for parameter tuning."""
    import ipywidgets as widgets
    
    semantic_threshold = widgets.FloatSlider(
        value=0.3,
        min=0.1,
        max=0.9,
        step=0.1,
        description='Semantic Threshold:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    
    llm_enabled = widgets.Checkbox(
        value=True,
        description='Enable LLM Explanations',
        style={'description_width': 'initial'}
    )
    
    return semantic_threshold, llm_enabled


def update_config_from_widgets(config: Dict[str, Any], widgets: tuple) -> None:
    """Update configuration based on widget values."""
    semantic_threshold, llm_enabled = widgets
    
    config['entity_matching']['matching']['semantic_match_threshold'] = semantic_threshold.value
    config['entity_matching']['llm']['enabled'] = llm_enabled.value


def analyze_enrichment_results(enriched_entities: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze and create DataFrame from enrichment results."""
    enriched_df = pd.DataFrame(enriched_entities)
    
    # Add computed columns
    enriched_df['context_length'] = enriched_df['entity_context'].str.len()
    enriched_df['has_wikipedia'] = enriched_df['entity_context'].str.contains('Wikipedia', na=False)
    
    return enriched_df


def create_entity_visualizations(processed_articles: List[Any], max_articles: int = 2) -> None:
    """Create spaCy visualizations for processed articles."""
    try:
        nlp = spacy.load("en_core_web_sm")
        
        for i, article in enumerate(processed_articles[:max_articles]):
            article_text = getattr(article, 'article', '')
            if not article_text:
                continue
                
            print(f"\n--- Article {i+1}: {getattr(article, 'title', 'Untitled')} ---")
            
            # Process with spaCy for visualization
            doc = nlp(article_text[:500])  # Limit for better visualization
            
            # Create visualization
            html = displacy.render(doc, style="ent", jupyter=True, options={
                "colors": {
                    "PERSON": "#ff9999",
                    "ORG": "#99ccff", 
                    "GPE": "#99ff99",
                    "EVENT": "#ffcc99",
                    "WORK_OF_ART": "#cc99ff"
                }
            })
            
            # Show entity statistics
            entities = getattr(article, 'extracted_entities', [])
            print(f"Entities found: {len(entities)}")
            for entity in entities[:5]:  # Show first 5 entities
                print(f"   - {entity} ({getattr(entity, 'label_', 'UNKNOWN')})")
                
    except Exception as e:
        print(f"⚠️ Could not load spaCy model for visualization: {e}")
        print("   Install with: python -m spacy download en_core_web_sm")


def extract_entities_to_dataframe(processed_articles: List[Any]) -> pd.DataFrame:
    """Extract all entities from processed articles into a DataFrame."""
    all_entities = []
    
    for article in processed_articles:
        if hasattr(article, 'extracted_entities'):
            entities = article.extracted_entities
        else:
            entities = article.get('extracted_entities', [])
        
        for entity in entities:
            entity_dict = {
                'article_id': getattr(article, 'article_id', 'unknown'),
                'article_title': getattr(article, 'title', 'unknown'),
                'entity_text': str(entity),
                'entity_type': getattr(entity, 'label_', 'UNKNOWN'),
                'start_pos': getattr(entity, 'start', 0),
                'end_pos': getattr(entity, 'end', 0),
                'confidence': getattr(entity, 'confidence', 1.0)
            }
            all_entities.append(entity_dict)
    
    return pd.DataFrame(all_entities)


def analyze_matching_results(matching_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze and create DataFrame from matching results."""
    all_matches = []
    
    for result in matching_results:
        article_id = result.get('article_id', 'unknown')
        matches = result.get('matches_found', [])
        
        for match in matches:
            match_dict = {
                'article_id': article_id,
                'extracted_entity': match.get('extracted_entity', ''),
                'watched_entity': match.get('watched_entity', ''),
                'confidence': match.get('confidence', 0.0),
                'match_type': match.get('match_type', ''),
                'reasoning': match.get('reasoning', ''),
                'explanation_full': match.get('explanation_full', '')
            }
            all_matches.append(match_dict)
    
    return pd.DataFrame(all_matches)


def export_results_to_csv(results_df: pd.DataFrame, filename: str) -> None:
    """Export results DataFrame to CSV file."""
    output_path = Path(f"results_{filename}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"✅ Results exported to: {output_path}")


def export_results_to_json(results: List[Dict[str, Any]], filename: str) -> None:
    """Export results list to JSON file."""
    output_path = Path(f"results_{filename}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results exported to: {output_path}")


def setup_environment_variables(project_root: Path) -> bool:
    """
    Set up environment variables with interactive fallback.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        bool: True if all required variables are set, False otherwise
    """
    import os
    
    # First, try to load from .env file
    env_file = project_root / '.env'
    env_loaded = False

    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✅ Found .env file, attempting to load environment variables...")
            env_loaded = True
        except ImportError:
            print("⚠️ python-dotenv not available, loading .env manually...")
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                env_loaded = True
                print("✅ Loaded .env file manually")
            except Exception as e:
                print(f"❌ Error loading .env file: {e}")
                env_loaded = False
    else:
        print("ℹ️ No .env file found")

    # Check if all required environment variables are set
    required_vars = ['ELASTIC_CLOUD_ID', 'ELASTIC_API_KEY', 'OPENAI_API_KEY']
    missing_vars = []

    for var in required_vars:
        if os.getenv(var):
            print(f"   ✅ {var}: SET")
        else:
            print(f"   ❌ {var}: NOT SET")
            missing_vars.append(var)

    # If variables are missing, provide interactive setup
    if missing_vars:
        print(f"\n⚠️ Missing required environment variables: {', '.join(missing_vars)}")
        print("\n🔧 Interactive Setup:")
        print("Please provide your API credentials below:")
        print("(You can also create a .env file in the project root with these variables)")
        
        for var in missing_vars:
            if var == 'ELASTIC_CLOUD_ID':
                print(f"\n📝 {var}:")
                print("   This is your Elastic Cloud deployment ID")
                print("   Format: deployment-name:base64-encoded-endpoint")
                print("   Example: my-deployment:dXMtY2VudHJhbDEuZ2NwLmVsYXN0aWMuY2xvdWQk...")
            elif var == 'ELASTIC_API_KEY':
                print(f"\n📝 {var}:")
                print("   This is your Elastic Cloud API key")
                print("   Format: base64-encoded-credentials")
                print("   Example: TzItOEZaa0JuY29pZV9tOGVtU3o6WGFQQzdVQlpWMjVRTktPVXdMRjlYUQ==")
            elif var == 'OPENAI_API_KEY':
                print(f"\n📝 {var}:")
                print("   This is your OpenAI API key")
                print("   Format: sk-...")
                print("   Example: sk-3BtuUB2tgLefwfG2AOAymQ...")
            
            # Get user input
            value = input(f"   Enter your {var}: ").strip()
            if value:
                os.environ[var] = value
                print(f"   ✅ {var} set")
            else:
                print(f"   ❌ {var} not provided")

    # Final verification
    print(f"\n🔍 Final Verification:")
    all_set = True
    for var in required_vars:
        if os.getenv(var):
            print(f"   ✅ {var}: SET")
        else:
            print(f"   ❌ {var}: NOT SET")
            all_set = False

    if not all_set:
        print(f"\n❌ CRITICAL ERROR: Cannot continue without all required API credentials!")
        print("Please ensure you have:")
        print("   1. Elastic Cloud deployment ID and API key")
        print("   2. OpenAI API key")
        print("   3. Either set them interactively above or create a .env file")
        print("\nThe tutorial cannot proceed without these credentials.")
        return False
    
    print(f"\n🎉 All API credentials configured successfully!")
    return True


def clean_pipeline_state(project_root: Path) -> None:
    """Clean up any existing state files to ensure fresh processing."""
    state_dir = project_root / 'pipeline_state'
    
    print("\n🧹 Cleaning up previous state files for fresh tutorial run...")
    
    if state_dir.exists():
        deleted_count = 0
        for p in state_dir.glob('*_state.json'):
            try:
                p.unlink()
                print(f"   🗑️ Deleted: {p.name}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Could not delete {p.name}: {e}")
        if deleted_count == 0:
            print("   ℹ️ No state files found to clean")
        else:
            print(f"   ✅ Cleaned {deleted_count} state files")
    else:
        print("   ℹ️ No state directory found")
