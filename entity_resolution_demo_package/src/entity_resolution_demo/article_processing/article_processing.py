#!/usr/bin/env python3
"""
Article Processing Module

Processes articles and extracts entities using HybridNERExtractor.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Import local modules
from entity_resolution_demo.pipeline_runner.config import load_config
from entity_resolution_demo.article_processing.article_data import load_articles, get_articles, get_ner_config
from entity_resolution_demo.pipeline_runner.utils import (
    print_header, print_subheader, print_success, print_warning, 
    print_error, print_info, time_function,
    save_state, load_state
)

# Import project modules
from entity_resolution_demo.article_processing.article_processor import ArticleProcessor, Article, ProcessedArticle
from entity_resolution_demo.search.elastic_client import ElasticClient
from entity_resolution_demo.article_processing.hybrid_ner_extractor import HybridNERExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@time_function
def create_article_processor(config: Dict[str, Any]) -> ArticleProcessor:
    """
    Create an article processor
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ArticleProcessor: Configured article processor
    """
    print_subheader("Creating Article Processor")
    
    # Create Elasticsearch client
    elastic_client = ElasticClient(config)
    
    # Create article processor with HybridNERExtractor
    processor = ArticleProcessor(config, elastic_client)
    
    # Check if HybridNERExtractor is being used
    if isinstance(processor.name_extractor, HybridNERExtractor):
        print_success("Using HybridNERExtractor for entity extraction")
    else:
        print_warning("Not using HybridNERExtractor - compound entities may not be detected")
    
    return processor

@time_function
def create_articles(config: Dict[str, Any]) -> List[Article]:
    """
    Create articles for processing
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List[Article]: List of articles
    """
    print_subheader("Creating Articles")
    
    # Load articles from file or use default
    data_file = config["article_processing"].get("data_file")
    use_default = config["article_processing"].get("use_default_data", True)
    
    # Load articles from the appropriate source
    articles_data, _ = load_articles(data_file) if data_file else (get_articles(), get_ner_config())
    
    print_info(f"Loaded {len(articles_data)} articles from {'data file' if data_file else 'default data'}")
    
    # Create articles
    articles = []
    for article_data in articles_data:
        article = Article(
            id=article_data["id"],
            title=article_data["title"],
            content=article_data["content"],
            source=article_data["source"],
            language=article_data.get("language", "en")
        )
        articles.append(article)
        print_info(f"Created article: {article.id} - '{article.title}'")
    
    print_success(f"Created {len(articles)} articles")
    return articles

@time_function
def process_articles(processor: ArticleProcessor, articles: List[Article]) -> List[ProcessedArticle]:
    """
    Process articles to extract entities
    
    Args:
        processor: Article processor
        articles: List of articles
        
    Returns:
        List[ProcessedArticle]: List of processed articles
    """
    print_subheader("Processing Articles")
    
    # Process articles
    processed_articles = []
    for article in articles:
        print_info(f"Processing article: {article.id} - '{article.title}'")
        
        try:
            # Process article
            processed_article = processor.process_article(article)
            
            # Print extraction results
            print_success(f"Processed {article.id} - found {processed_article.total_entities_found} entities in {processed_article.processing_time:.2f}s")
            
            # Print extracted entities
            if processed_article.extracted_entities:
                print_info(f"Extracted entities from {article.id}:")
                for entity in processed_article.extracted_entities:
                    entity_type_emoji = "👤" if entity.entity_type == "PERSON" else "🏢" if entity.entity_type == "ORGANIZATION" else "📍" if entity.entity_type == "LOCATION" else "📌"
                    print_info(f"  {entity_type_emoji} {entity.name} ({entity.entity_type}) - Confidence: {entity.confidence:.2f}")
                    print_info(f"     Context: ...{entity.context[:50]}..." if len(entity.context) > 50 else f"     Context: {entity.context}")
            
            processed_articles.append(processed_article)
        except Exception as e:
            print_error(f"Error processing article {article.id}: {e}")
    
    # Print statistics
    stats = processor.get_stats()
    print_success(f"Processed {stats['articles_processed']} articles")
    print_info(f"  Total entities extracted: {stats['total_entities_extracted']}")
    print_info(f"  Average processing time: {stats['average_processing_time']:.2f}s")
    
    return processed_articles

@time_function
def index_processed_articles(processed_articles: List[ProcessedArticle], config: Dict[str, Any]) -> bool:
    """
    Index processed articles in Elasticsearch
    
    Args:
        processed_articles: List of processed articles
        config: Configuration dictionary
        
    Returns:
        bool: True if indexing was successful, False otherwise
    """
    print_subheader("Indexing Processed Articles")
    
    # Create Elasticsearch client
    elastic_client = ElasticClient(config)
    
    # Get index name from config
    articles_index = config["elasticsearch"].get("processed_articles_index", "processed_articles")
    print_info(f"Using index: {articles_index}")
    
    # Create index if it doesn't exist
    if not elastic_client.es.indices.exists(index=articles_index):
        print_info(f"Creating index: {articles_index}")
        
        # Define index mapping for processed articles
        mapping = {
            "mappings": {
                "properties": {
                    # Article information
                    "article_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "source": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "published_date": {"type": "date"},
                    "url": {"type": "keyword"},
                    "processing_date": {"type": "date"},
                    
                    # Processing information
                    "processing_time": {"type": "float"},
                    "total_entities_found": {"type": "integer"},
                    
                    # Extracted entities
                    "extracted_entities": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "text"},
                            "entity_type": {"type": "keyword"},
                            "confidence": {"type": "float"},
                            "context": {"type": "text"},
                            "position": {"type": "integer"},
                            "extraction_method": {"type": "keyword"}
                        }
                    },
                    
                    # For semantic search
                    "content_semantic": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        # Create index
        elastic_client.es.indices.create(index=articles_index, body=mapping)
        print_success(f"Created index: {articles_index}")
    
    # Index processed articles
    indexed_count = 0
    for processed_article in processed_articles:
        # Get article information
        article = processed_article.article
        
        # Create document
        doc = {
            "article_id": article.id,
            "title": article.title,
            "content": article.content,
            "source": article.source,
            "language": article.language,
            "published_date": article.published_date.isoformat() if article.published_date else None,
            "url": article.url,
            "processing_date": datetime.now().isoformat(),  # Use current time as processing date
            "processing_time": processed_article.processing_time,
            "total_entities_found": processed_article.total_entities_found,
            "extracted_entities": []
        }
        
        # Add extracted entities
        for entity in processed_article.extracted_entities:
            entity_doc = {
                "name": entity.name,
                "entity_type": entity.entity_type,
                "confidence": entity.confidence,
                "context": entity.context,
                "position": entity.position,
                "extraction_method": entity.extraction_method
            }
            doc["extracted_entities"].append(entity_doc)
        
        # Index document
        try:
            response = elastic_client.es.index(index=articles_index, id=article.id, body=doc)
            indexed_count += 1
            print_success(f"Indexed article: {article.id} - '{article.title}' with {len(processed_article.extracted_entities)} entities")
        except Exception as e:
            print_error(f"Error indexing article {article.id}: {e}")
    
    # Refresh index
    elastic_client.es.indices.refresh(index=articles_index)
    
    print_success(f"Indexed {indexed_count} articles to {articles_index}")
    return True

@time_function
def verify_article_processing(processed_articles: List[ProcessedArticle], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that articles were processed correctly
    
    Args:
        processed_articles: List of processed articles
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Verification results
    """
    print_subheader("Verifying Article Processing")
    
    # Check if any articles were processed
    if not processed_articles:
        print_error("No articles were processed")
        return {"success": False, "details": "No articles were processed"}
    
    # Check if entities were extracted
    total_entities = sum(article.total_entities_found for article in processed_articles)
    if total_entities == 0:
        print_error("No entities were extracted from any article")
        return {"success": False, "details": "No entities were extracted"}
    
    print_success(f"Extracted {total_entities} entities from {len(processed_articles)} articles")
    
    # Check for specific entity types
    entity_types = set()
    for article in processed_articles:
        for entity in article.extracted_entities:
            entity_types.add(entity.entity_type)
    
    print_info(f"Found entity types: {', '.join(entity_types)}")
    
    # Check for compound entities
    compound_entities = []
    for article in processed_articles:
        for entity in article.extracted_entities:
            if " " in entity.name and not entity.entity_type == "PERSON":
                compound_entities.append(entity)
    
    if compound_entities:
        print_success(f"Found {len(compound_entities)} compound entities")
        for entity in compound_entities[:3]:  # Show first 3
            print_info(f"  {entity.name} ({entity.entity_type}) - Confidence: {entity.confidence:.2f}")
    else:
        print_warning("No compound entities found - HybridNERExtractor may not be working correctly")
    
    # Check for multilingual entities
    languages = set()
    for article in processed_articles:
        if article.article.language:
            languages.add(article.article.language)
    
    if len(languages) > 1:
        print_success(f"Processed articles in multiple languages: {', '.join(languages)}")
    else:
        print_warning("Only processed articles in one language")
    
    return {
        "success": True,
        "details": {
            "total_articles": len(processed_articles),
            "total_entities": total_entities,
            "entity_types": list(entity_types),
            "compound_entities": len(compound_entities),
            "languages": list(languages)
        }
    }

def run_article_processing(config: Dict[str, Any], verify: bool = True, load_previous: bool = True, state_dir: str = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Run the article processing pipeline
    
    Args:
        config: Configuration dictionary
        verify: Whether to verify the results
        load_previous: Whether to load previous state
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Success flag and state dictionary
    """
    print_header("ARTICLE PROCESSING")
    
    # 1. Load previous state if requested
    previous_state = None
    if load_previous:
        previous_state = load_state("entity_preparation", state_dir=state_dir)
        if previous_state:
            print_success("Loaded previous state from entity_preparation")
        else:
            print_warning(f"No previous state found from entity_preparation in {state_dir if state_dir else 'default location'}")
    
    # 2. Create article processor
    processor = create_article_processor(config)
    
    # 3. Create articles
    articles = create_articles(config)
    
    # 4. Process articles
    processed_articles = process_articles(processor, articles)
    
    # 5. Verification step
    if verify:
        verification_results = verify_article_processing(processed_articles, config)
        if not verification_results["success"]:
            print_error("Article processing verification failed")
            print_info(f"Details: {verification_results['details']}")
            return False, {}
        print_success("Article processing verification successful")
    
    # 6. Index processed articles to Elasticsearch
    index_success = index_processed_articles(processed_articles, config)
    if not index_success:
        print_error("Failed to index processed articles")
    
    # 7. Save state for next module
    state = {
        "processed_articles": processed_articles,
        "processor_stats": processor.get_stats(),
        "indexed_articles_count": len(processed_articles) if index_success else 0,
        "article_index_name": config["elasticsearch"].get("processed_articles_index", "processed_articles")
    }
    save_state("article_processing", state, state_dir=state_dir)
    
    return True, state

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Run article processing
    success, state = run_article_processing(config)
    
    # Print result
    if success:
        print_success("Article processing completed successfully")
    else:
        print_error("Article processing failed")
        sys.exit(1)
