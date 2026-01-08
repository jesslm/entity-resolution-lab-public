"""
Enhanced Notebook Helper Functions for Modular Pipeline Architecture

This module provides specialized helper functions for each pipeline stage,
supporting the modular notebook architecture with rich visualizations,
analysis, and interactive components.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import existing helpers
from .notebook_helpers import (
    setup_environment_variables,
    clean_pipeline_state,
    load_and_preview_data,
    create_interactive_widgets,
    update_config_from_widgets
)

# Import state management
from .state_manager import (
    get_state_manager,
    check_pipeline_readiness,
    create_state_summary,
    save_entity_preparation_state,
    load_entity_preparation_state,
    save_article_processing_state,
    load_article_processing_state,
    save_entity_matching_state,
    load_entity_matching_state
)


# ============================================================================
# ENTITY PREPARATION HELPER FUNCTIONS
# ============================================================================

def create_entity_watch_list_display(entities: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame for displaying entity watch list with metadata.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        DataFrame with entity information
    """
    if not entities:
        return pd.DataFrame()
    
    df = pd.DataFrame(entities)
    
    # Add computed columns
    df['alias_count'] = df['aliases'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['has_description'] = df['description'].notna() & (df['description'] != '')
    
    # Handle priority field (may not exist in all datasets)
    if 'priority' in df.columns:
        df['priority_level'] = df['priority'].map({'high': 3, 'medium': 2, 'low': 1})
    else:
        df['priority'] = 'medium'  # Default priority
        df['priority_level'] = 2   # Default priority level
    
    return df


def analyze_enrichment_results(enriched_entities: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze enrichment results and create summary statistics.
    
    Args:
        enriched_entities: List of enriched entity dictionaries
        
    Returns:
        DataFrame with enrichment analysis
    """
    if not enriched_entities:
        return pd.DataFrame()
    
    enriched_df = pd.DataFrame(enriched_entities)
    
    # Add computed columns
    enriched_df['context_length'] = enriched_df['entity_context'].str.len()
    enriched_df['has_wikipedia'] = enriched_df['enrichment_source'].apply(
        lambda x: 'Wikipedia' in x or 'Primary' in x if isinstance(x, str) else False
    )
    enriched_df['confidence_score'] = enriched_df.get('confidence_score', 0.8)
    
    return enriched_df


def create_index_verification_report(indexer) -> Dict[str, Any]:
    """
    Create a verification report for Elasticsearch indexing.
    
    Args:
        indexer: EntityIndexer instance
        
    Returns:
        Dictionary with verification results
    """
    try:
        # Get index information
        index_name = indexer.entity_index  # Correct attribute name
        client = indexer.elastic_client.es
        
        # Check if index exists
        index_exists = client.indices.exists(index=index_name)
        
        if not index_exists:
            return {
                "status": "error",
                "message": f"Index '{index_name}' does not exist",
                "index_name": index_name,
                "document_count": 0
            }
        
        # Get index stats
        stats = client.indices.stats(index=index_name)
        doc_count = stats['indices'][index_name]['total']['docs']['count']
        
        # Get mapping
        mapping = client.indices.get_mapping(index=index_name)
        
        return {
            "status": "success",
            "index_name": index_name,
            "document_count": doc_count,
            "mapping_fields": list(mapping[index_name]['mappings']['properties'].keys()),
            "index_size": stats['indices'][index_name]['total']['store']['size_in_bytes']
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "index_name": getattr(indexer, 'index_name', 'unknown')
        }


def demonstrate_semantic_search(indexer, query: str) -> List[Dict[str, Any]]:
    """
    Demonstrate semantic vs lexical search capabilities.
    
    Args:
        indexer: EntityIndexer instance
        query: Search query
        
    Returns:
        List of search results
    """
    try:
        client = indexer.elastic_client.es
        index_name = indexer.index_name
        
        # Perform semantic search
        semantic_query = {
            "query": {
                "match": {
                    "name_semantic": query
                }
            },
            "size": 5
        }
        
        semantic_results = client.search(index=index_name, body=semantic_query)
        
        # Perform lexical search
        lexical_query = {
            "query": {
                "match": {
                    "name": query
                }
            },
            "size": 5
        }
        
        lexical_results = client.search(index=index_name, body=lexical_query)
        
        return {
            "semantic_results": semantic_results['hits']['hits'],
            "lexical_results": lexical_results['hits']['hits'],
            "query": query
        }
        
    except Exception as e:
        return {"error": str(e), "query": query}


def export_entity_preparation_results(state: Dict[str, Any], format: str = "csv") -> str:
    """
    Export entity preparation results in various formats.
    
    Args:
        state: State data from entity preparation
        format: Export format ('csv', 'json', 'excel')
        
    Returns:
        Path to exported file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        # Export enriched entities
        enriched_entities = state.get('results', {}).get('enriched_entities', [])
        if enriched_entities:
            df = pd.DataFrame(enriched_entities)
            filename = f"entity_preparation_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            return filename
    
    elif format == "json":
        filename = f"entity_preparation_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        return filename
    
    elif format == "excel":
        filename = f"entity_preparation_results_{timestamp}.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Enriched entities sheet
            enriched_entities = state.get('results', {}).get('enriched_entities', [])
            if enriched_entities:
                df = pd.DataFrame(enriched_entities)
                df.to_excel(writer, sheet_name='Enriched Entities', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Entities', 'Enriched Entities', 'Average Context Length'],
                'Value': [
                    len(enriched_entities),
                    len(enriched_entities),
                    np.mean([len(e.get('entity_context', '')) for e in enriched_entities]) if enriched_entities else 0
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        return filename
    
    else:
        raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# ARTICLE PROCESSING HELPER FUNCTIONS
# ============================================================================

def create_article_preview_display(articles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame for displaying article previews with metadata.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        DataFrame with article information
    """
    if not articles:
        return pd.DataFrame()
    
    df = pd.DataFrame(articles)
    
    # Add computed columns
    df['content_length'] = df['content'].str.len()
    df['title_length'] = df['title'].str.len()
    
    # Handle optional columns gracefully
    if 'url' in df.columns:
        df['has_url'] = df['url'].notna() & (df['url'] != '')
    else:
        df['has_url'] = False
    
    if 'published_date' in df.columns:
        df['has_published_date'] = df['published_date'].notna() & (df['published_date'] != '')
    else:
        df['has_published_date'] = False
    
    return df


def analyze_extraction_results(extracted_entities: List[Any]) -> Dict[str, Any]:
    """
    Analyze extraction results and create summary statistics.
    
    Args:
        extracted_entities: List of ExtractedEntity objects or dictionaries
        
    Returns:
        Dictionary with extraction analysis
    """
    if not extracted_entities:
        return {
            'total_entities': 0,
            'entity_types': {},
            'average_confidence': 0.0,
            'confidence_distribution': {}
        }
    
    # Handle both ExtractedEntity objects and dictionaries
    entity_data = []
    for entity in extracted_entities:
        if hasattr(entity, 'entity_type'):  # ExtractedEntity object
            entity_data.append({
                'label': entity.entity_type,
                'confidence': entity.confidence,
                'text': entity.name
            })
        elif isinstance(entity, dict):  # Dictionary
            entity_data.append({
                'label': entity.get('label', entity.get('entity_type', 'UNKNOWN')),
                'confidence': entity.get('confidence', 0.0),
                'text': entity.get('text', entity.get('name', ''))
            })
    
    if not entity_data:
        return {
            'total_entities': 0,
            'entity_types': {},
            'average_confidence': 0.0,
            'confidence_distribution': {}
        }
    
    # Calculate statistics
    total_entities = len(entity_data)
    entity_types = {}
    confidence_scores = []
    
    for entity in entity_data:
        label = entity['label']
        entity_types[label] = entity_types.get(label, 0) + 1
        confidence_scores.append(entity['confidence'])
    
    average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    return {
        'total_entities': total_entities,
        'entity_types': entity_types,
        'average_confidence': average_confidence,
        'confidence_distribution': {
            'high': sum(1 for c in confidence_scores if c > 0.8),
            'medium': sum(1 for c in confidence_scores if 0.5 <= c <= 0.8),
            'low': sum(1 for c in confidence_scores if c < 0.5)
        }
    }


def demonstrate_hybrid_extraction(extractor, sample_text: str) -> Dict[str, Any]:
    """
    Demonstrate hybrid extraction capabilities with sample text.
    
    Args:
        extractor: HybridNERExtractor instance
        sample_text: Text to extract entities from
        
    Returns:
        Dictionary with extraction results
    """
    try:
        entities = extractor.extract_entities_hybrid(sample_text)
        
        # Analyze extraction methods
        extraction_methods = {}
        entity_types = {}
        
        for entity in entities:
            method = entity.get('extraction_method', 'unknown')
            entity_type = entity.get('type', 'unknown')
            
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        return {
            "entities": entities,
            "extraction_methods": extraction_methods,
            "entity_types": entity_types,
            "total_entities": len(entities),
            "sample_text": sample_text
        }
        
    except Exception as e:
        return {"error": str(e), "sample_text": sample_text}


def create_entity_type_visualization(extracted_entities: List[Any]) -> Dict[str, int]:
    """
    Create entity type distribution from extracted entities.
    
    Args:
        extracted_entities: List of ExtractedEntity objects or dictionaries
        
    Returns:
        Dictionary with entity type counts
    """
    if not extracted_entities:
        return {}
    
    entity_types = {}
    
    # Handle both ExtractedEntity objects and dictionaries
    for entity in extracted_entities:
        if hasattr(entity, 'entity_type'):  # ExtractedEntity object
            label = entity.entity_type
        elif isinstance(entity, dict):  # Dictionary
            label = entity.get('label', entity.get('entity_type', 'UNKNOWN'))
        else:
            continue
            
        entity_types[label] = entity_types.get(label, 0) + 1
    
    return entity_types


def compare_extraction_methods(processed_articles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare performance of different extraction methods.
    
    Args:
        processed_articles: List of processed article dictionaries
        
    Returns:
        DataFrame with method comparison
    """
    if not processed_articles:
        return pd.DataFrame()
    
    # Flatten extracted entities
    all_entities = []
    for article in processed_articles:
        entities = article.get('extracted_entities', [])
        for entity in entities:
            entity['article_id'] = article.get('id', 'unknown')
            all_entities.append(entity)
    
    if not all_entities:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_entities)
    
    # Group by extraction method
    method_stats = df.groupby('extraction_method').agg({
        'confidence': ['mean', 'std', 'count'],
        'type': 'nunique'
    }).round(3)
    
    method_stats.columns = ['avg_confidence', 'std_confidence', 'entity_count', 'unique_types']
    
    return method_stats


def export_article_processing_results(state: Dict[str, Any], format: str = "csv") -> str:
    """
    Export article processing results in various formats.
    
    Args:
        state: State data from article processing
        format: Export format ('csv', 'json', 'excel')
        
    Returns:
        Path to exported file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        # Export extracted entities
        processed_articles = state.get('results', {}).get('processed_articles', [])
        all_entities = []
        for article in processed_articles:
            entities = article.get('extracted_entities', [])
            for entity in entities:
                entity['article_id'] = article.get('id', 'unknown')
                entity['article_title'] = article.get('title', 'unknown')
                all_entities.append(entity)
        
        if all_entities:
            df = pd.DataFrame(all_entities)
            filename = f"article_processing_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            return filename
    
    elif format == "json":
        filename = f"article_processing_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        return filename
    
    elif format == "excel":
        filename = f"article_processing_results_{timestamp}.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Extracted entities sheet
            processed_articles = state.get('results', {}).get('processed_articles', [])
            all_entities = []
            for article in processed_articles:
                entities = article.get('extracted_entities', [])
                for entity in entities:
                    entity['article_id'] = article.get('id', 'unknown')
                    entity['article_title'] = article.get('title', 'unknown')
                    all_entities.append(entity)
            
            if all_entities:
                df = pd.DataFrame(all_entities)
                df.to_excel(writer, sheet_name='Extracted Entities', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Articles', 'Total Entities', 'Average Entities per Article'],
                'Value': [
                    len(processed_articles),
                    len(all_entities),
                    len(all_entities) / len(processed_articles) if processed_articles else 0
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        return filename
    
    else:
        raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# ENTITY MATCHING HELPER FUNCTIONS
# ============================================================================

def demonstrate_search_strategies(matcher, sample_entity) -> Dict[str, Any]:
    """
    Demonstrate all three search strategies with sample entity.
    
    Args:
        matcher: ElasticsearchEntityMatcher instance
        sample_entity: ExtractedEntity to search for
        
    Returns:
        Dictionary with search strategy results
    """
    try:
        # This would need to be implemented based on the actual matcher interface
        # For now, return a placeholder structure
        return {
            "exact_matches": [],
            "alias_matches": [],
            "hybrid_matches": [],
            "sample_entity": sample_entity.name if hasattr(sample_entity, 'name') else str(sample_entity),
            "note": "Search strategy demonstration requires actual matcher implementation"
        }
        
    except Exception as e:
        return {"error": str(e)}


def analyze_match_quality(entity_matches: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze match quality and create detailed statistics.
    
    Args:
        entity_matches: List of entity match dictionaries
        
    Returns:
        DataFrame with match quality analysis
    """
    if not entity_matches:
        return pd.DataFrame()
    
    df = pd.DataFrame(entity_matches)
    
    # Add computed columns
    df['confidence_level'] = pd.cut(df['confidence'], bins=[0, 0.5, 0.8, 1.0], labels=['Low', 'Medium', 'High'])
    df['has_explanation'] = df['reasoning'].notna() & (df['reasoning'] != '')
    
    return df


def demonstrate_llm_judgment(judge, sample_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Demonstrate LLM-powered judgment with sample matches.
    
    Args:
        judge: EnhancedBatchMatchJudge instance
        sample_matches: List of potential matches to judge
        
    Returns:
        Dictionary with judgment results
    """
    try:
        # This would need to be implemented based on the actual judge interface
        # For now, return a placeholder structure
        return {
            "judged_matches": [],
            "batch_size": len(sample_matches),
            "note": "LLM judgment demonstration requires actual judge implementation"
        }
        
    except Exception as e:
        return {"error": str(e)}


def create_match_type_visualization(entity_matches: List[Dict[str, Any]]) -> None:
    """
    Create visualizations for match types and confidence distributions.
    
    Args:
        entity_matches: List of entity match dictionaries
    """
    if not entity_matches:
        print("No matches to visualize")
        return
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = pd.DataFrame(entity_matches)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Match type distribution
        match_type_counts = df['match_type'].value_counts()
        ax1.pie(match_type_counts.values, labels=match_type_counts.index, autopct='%1.1f%%')
        ax1.set_title('Match Type Distribution')
        
        # Confidence distribution
        ax2.hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Confidence Score Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib/Seaborn not available for visualization")
        # Fallback to text summary
        df = pd.DataFrame(entity_matches)
        print("\nMatch Type Distribution:")
        print(df['match_type'].value_counts())
        print(f"\nConfidence Statistics:")
        print(f"Mean: {df['confidence'].mean():.3f}")
        print(f"Std: {df['confidence'].std():.3f}")
        print(f"Min: {df['confidence'].min():.3f}")
        print(f"Max: {df['confidence'].max():.3f}")


def compare_matching_strategies(matching_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare performance of different matching strategies.
    
    Args:
        matching_results: List of matching result dictionaries
        
    Returns:
        DataFrame with strategy comparison
    """
    if not matching_results:
        return pd.DataFrame()
    
    # This would need to be implemented based on the actual matching results structure
    # For now, return a placeholder
    return pd.DataFrame({
        'strategy': ['exact', 'alias', 'hybrid'],
        'matches_found': [0, 0, 0],
        'average_confidence': [0.0, 0.0, 0.0]
    })


def demonstrate_hybrid_search(matcher, query: str) -> Dict[str, Any]:
    """
    Demonstrate hybrid search capabilities with RRF.
    
    Args:
        matcher: ElasticsearchEntityMatcher instance
        query: Search query
        
    Returns:
        Dictionary with hybrid search results
    """
    try:
        # This would need to be implemented based on the actual matcher interface
        return {
            "lexical_results": [],
            "semantic_results": [],
            "hybrid_results": [],
            "query": query,
            "note": "Hybrid search demonstration requires actual matcher implementation"
        }
        
    except Exception as e:
        return {"error": str(e)}


def analyze_confidence_factors(entity_matches: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze confidence factor breakdown for matches.
    
    Args:
        entity_matches: List of entity match dictionaries
        
    Returns:
        DataFrame with confidence factor analysis
    """
    if not entity_matches:
        return pd.DataFrame()
    
    df = pd.DataFrame(entity_matches)
    
    # Extract confidence factors if available
    confidence_factors = []
    for _, match in df.iterrows():
        factors = match.get('confidence_factors', {})
        if isinstance(factors, dict):
            factors['match_id'] = match.get('id', 'unknown')
            confidence_factors.append(factors)
    
    if confidence_factors:
        return pd.DataFrame(confidence_factors)
    else:
        return pd.DataFrame()


def export_entity_matching_results(state: Dict[str, Any], format: str = "csv") -> str:
    """
    Export entity matching results in various formats.
    
    Args:
        state: State data from entity matching
        format: Export format ('csv', 'json', 'excel')
        
    Returns:
        Path to exported file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        # Export entity matches
        entity_matches = state.get('results', {}).get('entity_matches', [])
        if entity_matches:
            df = pd.DataFrame(entity_matches)
            filename = f"entity_matching_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            return filename
    
    elif format == "json":
        filename = f"entity_matching_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        return filename
    
    elif format == "excel":
        filename = f"entity_matching_results_{timestamp}.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Entity matches sheet
            entity_matches = state.get('results', {}).get('entity_matches', [])
            if entity_matches:
                df = pd.DataFrame(entity_matches)
                df.to_excel(writer, sheet_name='Entity Matches', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Matches', 'Average Confidence', 'Match Rate'],
                'Value': [
                    len(entity_matches),
                    np.mean([m.get('confidence', 0) for m in entity_matches]) if entity_matches else 0,
                    state.get('results', {}).get('matching_summary', {}).get('match_rate', 0)
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        return filename
    
    else:
        raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# NAVIGATION HELPER FUNCTIONS
# ============================================================================

def get_notebook_links() -> Dict[str, str]:
    """
    Get links to all notebooks in the pipeline.
    
    Returns:
        Dictionary mapping stage names to notebook paths
    """
    return {
        "entity_preparation": "01_entity_preparation.ipynb",
        "article_processing": "02_article_processing.ipynb",
        "entity_matching": "03_entity_matching.ipynb",
        "full_pipeline": "00_full_pipeline.ipynb"
    }


def create_status_dashboard() -> str:
    """
    Create HTML dashboard showing pipeline status.
    
    Returns:
        HTML dashboard string
    """
    return create_state_summary()


def generate_next_steps(current_stage: str) -> List[str]:
    """
    Generate next steps based on current pipeline stage.
    
    Args:
        current_stage: Current pipeline stage
        
    Returns:
        List of next step recommendations
    """
    stage_order = ["entity_preparation", "article_processing", "entity_matching"]
    
    try:
        current_index = stage_order.index(current_stage)
        if current_index < len(stage_order) - 1:
            next_stage = stage_order[current_index + 1]
            return [
                f"✅ {current_stage} completed successfully",
                f"➡️ Next: Run {next_stage} notebook",
                f"📁 Notebook: {get_notebook_links().get(next_stage, 'unknown')}",
                f"🔗 Or run the full pipeline: {get_notebook_links()['full_pipeline']}"
            ]
        else:
            return [
                f"✅ {current_stage} completed successfully",
                f"🎉 Pipeline complete! All stages finished.",
                f"📊 Review results and export data as needed",
                f"🔄 Or run the full pipeline again: {get_notebook_links()['full_pipeline']}"
            ]
    except ValueError:
        return [
            f"❓ Unknown stage: {current_stage}",
            f"📋 Available stages: {', '.join(stage_order)}",
            f"🔗 Full pipeline: {get_notebook_links()['full_pipeline']}"
        ]


def show_state_files_info(project_root: Path) -> str:
    """
    Show information about saved state files.
    
    Args:
        project_root: Path to the project root
        
    Returns:
        HTML string with state file information
    """
    from datetime import datetime
    
    state_dir = project_root / 'pipeline_state'
    
    if not state_dir.exists():
        return """
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4>📁 State Files</h4>
            <p>No state files found. State files will be created as you run the pipeline stages.</p>
        </div>
        """
    
    state_files = list(state_dir.glob('*_state.json'))
    
    if not state_files:
        return """
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4>📁 State Files</h4>
            <p>No state files found. State files will be created as you run the pipeline stages.</p>
        </div>
        """
    
    # Get file info
    file_info = []
    for state_file in sorted(state_files):
        try:
            stat = state_file.stat()
            size_kb = stat.st_size / 1024
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            file_info.append({
                'name': state_file.name,
                'size': f"{size_kb:.1f} KB",
                'modified': modified
            })
        except Exception:
            file_info.append({
                'name': state_file.name,
                'size': 'Unknown',
                'modified': 'Unknown'
            })
    
    # Create HTML table
    table_rows = []
    for info in file_info:
        table_rows.append(f"""
        <tr>
            <td><code>{info['name']}</code></td>
            <td>{info['size']}</td>
            <td>{info['modified']}</td>
        </tr>
        """)
    
    return f"""
    <div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h4>📁 Saved State Files</h4>
        <p>Pipeline state files are saved in: <code>{state_dir}</code></p>
        <p>These files contain the results of each pipeline stage and can be inspected or used for analysis.</p>
        
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="background-color: #bee5eb;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">File</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Size</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Last Modified</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        
        <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
            💡 <strong>Tip:</strong> You can open these JSON files in any text editor to inspect the detailed results, 
            or use them programmatically in your own analysis scripts.
        </p>
    </div>
    """
