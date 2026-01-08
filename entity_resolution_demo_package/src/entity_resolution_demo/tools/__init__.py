"""
Tools module for the entity_resolution_demo package.

This module contains utility tools for the entity resolution system.
"""

from entity_resolution_demo.tools.generate_tutorial_notebook import generate_tutorial_notebook
from entity_resolution_demo.tools.notebook_helpers import (
    setup_environment_variables,
    clean_pipeline_state,
    load_and_preview_data,
    create_interactive_widgets,
    update_config_from_widgets,
    analyze_enrichment_results,
    extract_entities_to_dataframe,
    create_entity_visualizations
)
from entity_resolution_demo.tools.state_manager import (
    StateManager,
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
from entity_resolution_demo.tools.enhanced_notebook_helpers import (
    # Entity Preparation Helpers
    create_entity_watch_list_display,
    analyze_enrichment_results as analyze_enrichment_results_enhanced,
    create_index_verification_report,
    demonstrate_semantic_search,
    export_entity_preparation_results,
    
    # Article Processing Helpers
    create_article_preview_display,
    analyze_extraction_results,
    demonstrate_hybrid_extraction,
    create_entity_type_visualization,
    compare_extraction_methods,
    export_article_processing_results,
    
    # Entity Matching Helpers
    demonstrate_search_strategies,
    analyze_match_quality,
    demonstrate_llm_judgment,
    create_match_type_visualization,
    compare_matching_strategies,
    demonstrate_hybrid_search,
    analyze_confidence_factors,
    export_entity_matching_results,
    
        # Navigation Helpers
        get_notebook_links,
        create_status_dashboard,
        generate_next_steps,
        show_state_files_info
)

__all__ = [
    'generate_tutorial_notebook',
    # Notebook Helpers
    'setup_environment_variables',
    'clean_pipeline_state',
    'load_and_preview_data',
    'create_interactive_widgets',
    'update_config_from_widgets',
    'analyze_enrichment_results',
    'extract_entities_to_dataframe',
    'create_entity_visualizations',
    # State Management
    'StateManager',
    'get_state_manager',
    'check_pipeline_readiness',
    'create_state_summary',
    'save_entity_preparation_state',
    'load_entity_preparation_state',
    'save_article_processing_state',
    'load_article_processing_state',
    'save_entity_matching_state',
    'load_entity_matching_state',
    # Enhanced Helpers
    'create_entity_watch_list_display',
    'analyze_enrichment_results_enhanced',
    'create_index_verification_report',
    'demonstrate_semantic_search',
    'export_entity_preparation_results',
    'create_article_preview_display',
    'analyze_extraction_results',
    'demonstrate_hybrid_extraction',
    'create_entity_type_visualization',
    'compare_extraction_methods',
    'export_article_processing_results',
    'demonstrate_search_strategies',
    'analyze_match_quality',
    'demonstrate_llm_judgment',
    'create_match_type_visualization',
    'compare_matching_strategies',
    'demonstrate_hybrid_search',
    'analyze_confidence_factors',
    'export_entity_matching_results',
        'get_notebook_links',
        'create_status_dashboard',
        'generate_next_steps',
        'show_state_files_info'
]
