"""
Article Processor for Real-Time Entity Resolution Alerts

Handles incoming articles and processes them for entity extraction and matching.
Uses Elasticsearch-native multilingual NER with XLM-RoBERTa for superior accuracy and performance.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import hashlib
import uuid

# Import package components
from entity_resolution_demo.search.elastic_client import ElasticClient
from entity_resolution_demo.article_processing.hybrid_ner_extractor import HybridNERExtractor


@dataclass
class Article:
    """Represents an article for processing"""
    id: str
    title: str
    content: str
    source: str
    published_date: Optional[str] = None
    url: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processed_date: Optional[str] = None
    
    def __post_init__(self):
        if self.processed_date is None:
            self.processed_date = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
        if self.id is None:
            # Generate ID from content hash if not provided
            content_hash = hashlib.md5(f"{self.title}{self.content}".encode()).hexdigest()
            self.id = f"article_{content_hash[:12]}"


@dataclass
class ExtractedEntity:
    """Represents an entity extracted from an article"""
    name: str
    entity_type: str
    confidence: float
    context: str  # Surrounding text where entity was found
    position: int  # Character position in article
    extraction_method: str  # How it was extracted (spacy, regex, etc.)


@dataclass
class ProcessedArticle:
    """Represents an article after entity extraction
    
    Note: Single-character entities are automatically filtered out from the unique_entities set
    to improve matching efficiency, as they are rarely meaningful for entity resolution.
    """
    article: Article
    extracted_entities: List[ExtractedEntity]
    processing_time: float
    total_entities_found: int
    unique_entities: Set[str]
    
    def __post_init__(self):
        # Filter out single-character entities from unique_entities
        all_entities = set(entity.name for entity in self.extracted_entities)
        filtered_entities = set(entity.name for entity in self.extracted_entities if len(entity.name) > 1)
        
        # Log if any entities were filtered out
        single_char_entities = all_entities - filtered_entities
        if single_char_entities:
            logging.getLogger(__name__).info(f"Filtered out {len(single_char_entities)} single-character entities: {single_char_entities}")
            
        self.unique_entities = filtered_entities
        self.total_entities_found = len(self.extracted_entities)


class ArticleProcessor:
    """Processes articles for real-time entity extraction and matching"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, elasticsearch_client: Optional[ElasticClient] = None, 
                 allow_local_fallback: bool = True):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize Elasticsearch client and NER extractor
        try:
            # Use provided elasticsearch_client if available, otherwise create a new one
            if elasticsearch_client:
                self.elastic_client = elasticsearch_client
            else:
                # Create a new ElasticClient instance
                self.elastic_client = ElasticClient(self.config, allow_local_fallback)
            
            self.name_extractor = HybridNERExtractor(elastic_client=self.elastic_client)
            self.logger.info("ArticleProcessor initialized with Hybrid NER (Elasticsearch NER + pattern-based title extraction)")
        except Exception as e:
            self.logger.error(f"Failed to initialize Elasticsearch NER: {e}")
            raise
        
        # Processing statistics
        self.stats = {
            'articles_processed': 0,
            'total_entities_extracted': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }
    
    def process_article(self, article: Article) -> ProcessedArticle:
        """Process a single article to extract entities"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing article: {article.id} - '{article.title[:50]}...'")
            
            # Combine title and content for entity extraction
            full_text = f"{article.title}\n\n{article.content}"
            
            # Extract entities using the multilingual name extractor
            extracted_entities = self._extract_entities_from_text(full_text, article)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create processed article result
            processed_article = ProcessedArticle(
                article=article,
                extracted_entities=extracted_entities,
                processing_time=processing_time,
                total_entities_found=len(extracted_entities),
                unique_entities=set()  # Will be populated in __post_init__ with filtering
            )
            
            # Update statistics
            self._update_stats(processed_article)
            
            self.logger.info(f"Processed article {article.id}: found {len(extracted_entities)} entities in {processing_time:.2f}s")
            
            return processed_article
            
        except Exception as e:
            self.stats['processing_errors'] += 1
            self.logger.error(f"Error processing article {article.id}: {e}")
            
            # Return empty result on error
            return ProcessedArticle(
                article=article,
                extracted_entities=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                total_entities_found=0,
                unique_entities=set()  # Already empty, no need to filter
            )
    
    def _extract_entities_from_text(self, text: str, article: Article) -> List[ExtractedEntity]:
        """Extract entities from text using Hybrid NER (Elasticsearch NER + pattern-based title extraction)"""
        extracted_entities = []
        
        try:
            # Use Hybrid NER extraction (combines Elasticsearch NER with pattern-based title extraction)
            hybrid_entities = self.name_extractor.extract_entities_hybrid(text)
            
            # Convert hybrid entity dictionaries to ExtractedEntity objects
            for entity_dict in hybrid_entities:
                # Extract entity name and position
                name = entity_dict.get('name', '')
                position = entity_dict.get('position', entity_dict.get('start_pos', 0))
                
                # Map entity type - HybridNERExtractor uses different entity type fields
                entity_type = entity_dict.get('type', entity_dict.get('entity_type', 'UNKNOWN'))
                
                # Get confidence score
                confidence = entity_dict.get('confidence', 0.0)
                
                # Get context (already provided by HybridNERExtractor)
                context = entity_dict.get('context', name)
                
                # Get extraction method
                extraction_method = entity_dict.get('extraction_method', 'hybrid_ner')
                
                entity = ExtractedEntity(
                    name=name,
                    entity_type=entity_type,
                    confidence=confidence,
                    context=context,
                    position=position,
                    extraction_method=extraction_method
                )
                
                extracted_entities.append(entity)
            
        except Exception as e:
            self.logger.error(f"Error in Hybrid NER extraction: {e}")
        
        return extracted_entities
    
    def process_batch(self, articles: List[Article]) -> List[ProcessedArticle]:
        """Process multiple articles in batch"""
        self.logger.info(f"Processing batch of {len(articles)} articles")
        
        processed_articles = []
        for article in articles:
            processed_article = self.process_article(article)
            processed_articles.append(processed_article)
        
        self.logger.info(f"Batch processing complete: {len(processed_articles)} articles processed")
        return processed_articles
    
    def _update_stats(self, processed_article: ProcessedArticle):
        """Update processing statistics"""
        self.stats['articles_processed'] += 1
        self.stats['total_entities_extracted'] += processed_article.total_entities_found
        
        # Update average processing time
        current_avg = self.stats['average_processing_time']
        count = self.stats['articles_processed']
        new_avg = ((current_avg * (count - 1)) + processed_article.processing_time) / count
        self.stats['average_processing_time'] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'articles_processed': 0,
            'total_entities_extracted': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }
        self.logger.info("Processing statistics reset")
    
    def create_article_from_text(self, 
                                title: str, 
                                content: str, 
                                source: str,
                                url: Optional[str] = None,
                                language: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> Article:
        """Convenience method to create an Article object"""
        return Article(
            id=str(uuid.uuid4()),
            title=title,
            content=content,
            source=source,
            url=url,
            language=language,
            metadata=metadata or {}
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    processor = ArticleProcessor()
    
    # Create sample articles
    sample_articles = [
        processor.create_article_from_text(
            title="Meeting with Russian President",
            content="Vladimir Putin met with Chinese leader Xi Jinping yesterday to discuss trade relations. The meeting was held in Moscow and covered various bilateral issues.",
            source="news_agency_1",
            language="en"
        ),
        processor.create_article_from_text(
            title="Tech Company Announcement",
            content="Apple Inc announced new products today. CEO Tim Cook presented the latest innovations at the company headquarters in Cupertino.",
            source="tech_news",
            language="en"
        ),
        processor.create_article_from_text(
            title="Multilingual Article",
            content="The meeting included 习近平 (Xi Jinping) and Владимир Путин (Vladimir Putin). Both leaders discussed important matters.",
            source="international_news",
            language="mixed"
        )
    ]
    
    # Process articles
    print("🔄 Processing sample articles...")
    processed_articles = processor.process_batch(sample_articles)
    
    # Display results
    print(f"\n📊 Processing Results:")
    for i, processed in enumerate(processed_articles, 1):
        print(f"\nArticle {i}: {processed.article.title}")
        print(f"  Entities found: {processed.total_entities_found}")
        print(f"  Processing time: {processed.processing_time:.2f}s")
        print(f"  Unique entities: {', '.join(processed.unique_entities)}")
        
        if processed.extracted_entities:
            print("  Detailed entities:")
            for entity in processed.extracted_entities[:3]:  # Show first 3
                print(f"    - {entity.name} ({entity.entity_type}) - Confidence: {entity.confidence:.2f}")
                print(f"      Context: ...{entity.context}...")
    
    # Show statistics
    stats = processor.get_stats()
    print(f"\n📈 Processing Statistics:")
    print(f"  Articles processed: {stats['articles_processed']}")
    print(f"  Total entities extracted: {stats['total_entities_extracted']}")
    print(f"  Average processing time: {stats['average_processing_time']:.2f}s")
    print(f"  Processing errors: {stats['processing_errors']}")
