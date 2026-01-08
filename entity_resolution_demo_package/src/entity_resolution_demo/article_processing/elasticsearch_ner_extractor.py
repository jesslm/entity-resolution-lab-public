#!/usr/bin/env python3
"""
Elasticsearch-based Named Entity Recognition Extractor

Replaces spaCy-based NER with Elasticsearch's built-in NER capabilities using
the inference API and trained models. Provides superior accuracy and performance
while maintaining compatibility with existing interfaces.

Based on: https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-ner-example
"""

import logging
import time
from typing import List, Dict, Set, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

# Import Elasticsearch client
try:
    from ..search.elastic_client import ElasticClient
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from search.elastic_client import ElasticClient


@dataclass
class ElasticsearchNEREntity:
    """Represents an entity extracted by Elasticsearch NER"""
    entity: str
    class_name: str  # PERSON, ORG, LOC, MISC
    class_probability: float
    start_pos: int
    end_pos: int
    context: str = ""
    extraction_method: str = "elasticsearch_ner"


class ElasticsearchNERExtractor:
    """Named Entity Recognition using Elasticsearch's built-in NER capabilities"""
    
    # Default NER model - now using multilingual XLM-RoBERTa (100+ languages)
    DEFAULT_NER_MODEL = "facebookai__xlm-roberta-large-finetuned-conll03-english"
    
    # Mapping from Elasticsearch NER classes to our standard format
    CLASS_MAPPING = {
        'PER': 'PERSON',
        'PERSON': 'PERSON',
        'ORG': 'ORGANIZATION', 
        'ORGANIZATION': 'ORGANIZATION',
        'LOC': 'LOCATION',
        'LOCATION': 'LOCATION',
        'MISC': 'MISCELLANEOUS',
        'MISCELLANEOUS': 'MISCELLANEOUS'
    }
    
    def __init__(self, elastic_client: Optional[ElasticClient] = None, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize Elasticsearch client
        if elastic_client:
            self.elastic_client = elastic_client
        else:
            try:
                self.elastic_client = ElasticClient(self.config)
                self.logger.info("Initialized Elasticsearch client for NER")
            except Exception as e:
                self.logger.error(f"Failed to initialize Elasticsearch client: {e}")
                raise
        
        # NER model configuration
        self.ner_model_id = self.config.get('ner_model_id', self.DEFAULT_NER_MODEL)
        self.input_field = self.config.get('ner_input_field', 'text_field')
        self.use_ml_api = False  # Flag to determine which API to use
        
        # Verify NER model is available
        self._verify_ner_model()
        
        # Performance tracking
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_entities_found': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }
        
        self.logger.info(f"ElasticsearchNERExtractor initialized with model: {self.ner_model_id}")
    
    def _verify_ner_model(self) -> bool:
        """Verify that the NER model is deployed and available via Inference API or ML API"""
        # First try modern Inference API
        try:
            response = self.elastic_client.es.inference.get(inference_id=self.ner_model_id)
            if response:
                self.logger.info(f"NER model {self.ner_model_id} is available via Inference API")
                self.use_ml_api = False
                return True
        except Exception as e:
            self.logger.debug(f"Model not found via Inference API: {e}")
        
        # Fallback to ML Trained Models API (legacy but functional)
        try:
            response = self.elastic_client.es.ml.get_trained_models(model_id=self.ner_model_id)
            if response and 'trained_model_configs' in response:
                # Check if model is deployed/started
                stats_response = self.elastic_client.es.ml.get_trained_models_stats(model_id=self.ner_model_id)
                if stats_response and 'trained_model_stats' in stats_response:
                    for stats in stats_response['trained_model_stats']:
                        deployment_stats = stats.get('deployment_stats', {})
                        state = deployment_stats.get('state', 'unknown')
                        if state == 'started':
                            self.logger.info(f"NER model {self.ner_model_id} is available via ML API (deployed and started)")
                            self.use_ml_api = True
                            return True
                        else:
                            self.logger.warning(f"NER model {self.ner_model_id} found but not started (state: {state})")
                            return False
        except Exception as e:
            self.logger.debug(f"Model not found via ML API: {e}")
        
        # Model not found via either API
        self.logger.error(f"NER model {self.ner_model_id} not found via Inference API or ML API")
        self.logger.info("To deploy a NER model, run:")
        self.logger.info(f"  docker run -it --rm docker.elastic.co/eland/eland \\")
        self.logger.info(f"    eland_import_hub_model \\")
        self.logger.info(f"    --cloud-id $CLOUD_ID \\")
        self.logger.info(f"    -u <username> -p <password> \\")
        self.logger.info(f"    --hub-model-id {self.ner_model_id.replace('__', '/')} \\")
        self.logger.info(f"    --task-type ner \\")
        self.logger.info(f"    --start")
        return False
    
    def _deploy_ner_model(self) -> bool:
        """Attempt to deploy the NER model"""
        try:
            # Start model deployment
            self.elastic_client.es.ml.start_trained_model_deployment(
                model_id=self.ner_model_id,
                wait_for="started"
            )
            self.logger.info(f"Successfully deployed NER model: {self.ner_model_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to deploy NER model: {e}")
            return False
    
    def extract_entities_from_text(self, text: str, context_window: int = 50) -> List[ElasticsearchNEREntity]:
        """
        Extract named entities from text using Elasticsearch NER
        
        Args:
            text: Input text to process
            context_window: Number of characters around entity for context
            
        Returns:
            List of extracted entities with metadata
        """
        start_time = time.time()
        
        try:
            # Prepare inference request
            inference_request = {
                "docs": [
                    {
                        self.input_field: text
                    }
                ]
            }
            
            # Call Elasticsearch inference API
            response = self.elastic_client.es.ml.infer_trained_model(
                model_id=self.ner_model_id,
                body=inference_request
            )
            
            # Process results
            entities = []
            if 'inference_results' in response and response['inference_results']:
                inference_result = response['inference_results'][0]
                
                if 'entities' in inference_result:
                    for entity_data in inference_result['entities']:
                        # Extract context around the entity
                        start_pos = entity_data['start_pos']
                        end_pos = entity_data['end_pos']
                        context_start = max(0, start_pos - context_window)
                        context_end = min(len(text), end_pos + context_window)
                        context = text[context_start:context_end]
                        
                        # Map class name to our standard format
                        class_name = self.CLASS_MAPPING.get(
                            entity_data['class_name'].upper(), 
                            entity_data['class_name']
                        )
                        
                        entity = ElasticsearchNEREntity(
                            entity=entity_data['entity'],
                            class_name=class_name,
                            class_probability=entity_data['class_probability'],
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context,
                            extraction_method=f"elasticsearch_ner_{self.ner_model_id}"
                        )
                        entities.append(entity)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_extractions'] += 1
            self.stats['successful_extractions'] += 1
            self.stats['total_entities_found'] += len(entities)
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (self.stats['total_extractions'] - 1) + processing_time) /
                self.stats['total_extractions']
            )
            
            self.logger.debug(f"Extracted {len(entities)} entities from text in {processing_time:.3f}s")
            return entities
            
        except Exception as e:
            self.stats['total_extractions'] += 1
            self.stats['failed_extractions'] += 1
            self.stats['processing_errors'] += 1
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_names_with_elasticsearch(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract names using Elasticsearch NER - compatible with existing interface
        
        This method maintains compatibility with the existing spaCy-based interface
        while using Elasticsearch's superior NER capabilities.
        """
        entities = self.extract_entities_from_text(text)
        
        # Convert to format compatible with existing code
        names = []
        for entity in entities:
            # Include all entity types: PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS
            # This ensures we capture all relevant entities including locations like "Peking"
            if entity.class_name in ['PERSON', 'ORGANIZATION', 'LOCATION', 'MISCELLANEOUS']:
                name_data = {
                    'name': entity.entity,
                    'type': entity.class_name,
                    'confidence': entity.class_probability,
                    'context': entity.context,
                    'position': entity.start_pos,
                    'extraction_method': entity.extraction_method,
                    'start_pos': entity.start_pos,
                    'end_pos': entity.end_pos
                }
                names.append(name_data)
        
        return names
    
    def create_ner_ingest_pipeline(self, pipeline_name: str = "elasticsearch_ner_pipeline") -> bool:
        """
        Create an Elasticsearch ingest pipeline for automatic NER processing
        
        This pipeline can be used to automatically extract entities during document indexing.
        """
        try:
            pipeline_config = {
                "description": "Elasticsearch NER pipeline for automatic entity extraction",
                "processors": [
                    {
                        "inference": {
                            "model_id": self.ner_model_id,
                            "target_field": "ml.ner",
                            "field_map": {
                                "content": self.input_field  # Map document content to model input
                            }
                        }
                    },
                    {
                        "script": {
                            "lang": "painless",
                            "if": "return ctx['ml']['ner'].containsKey('entities')",
                            "source": """
                                Map tags = new HashMap();
                                for (item in ctx['ml']['ner']['entities']) {
                                    if (!tags.containsKey(item.class_name)) {
                                        tags[item.class_name] = new HashSet();
                                    }
                                    tags[item.class_name].add(item.entity);
                                }
                                ctx['extracted_entities'] = tags;
                            """
                        }
                    }
                ],
                "on_failure": [
                    {
                        "set": {
                            "description": "Index document to 'failed-<index>'",
                            "field": "_index",
                            "value": "failed-{{{ _index }}}"
                        }
                    },
                    {
                        "set": {
                            "description": "Set error message",
                            "field": "ingest.failure",
                            "value": "{{_ingest.on_failure_message}}"
                        }
                    }
                ]
            }
            
            # Create the pipeline
            self.elastic_client.es.ingest.put_pipeline(
                id=pipeline_name,
                body=pipeline_config
            )
            
            self.logger.info(f"Created NER ingest pipeline: {pipeline_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create NER ingest pipeline: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'texts_processed': 0,
            'entities_extracted': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0
        }


# Compatibility function for existing code
def create_elasticsearch_ner_extractor(config: Optional[Dict[str, Any]] = None) -> ElasticsearchNERExtractor:
    """Factory function to create Elasticsearch NER extractor"""
    return ElasticsearchNERExtractor(config=config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test text with various entity types
    test_text = """
    Elastic is headquartered in Mountain View, California. The company was founded by 
    Shay Banon and is now led by CEO Ash Kulkarni. Vladimir Putin, the Russian President,
    recently met with Xi Jinping in Beijing to discuss trade relations.
    """
    
    try:
        # Initialize extractor
        extractor = ElasticsearchNERExtractor()
        
        # Extract entities
        entities = extractor.extract_entities_from_text(test_text)
        
        print(f"\n🔍 Extracted {len(entities)} entities using Elasticsearch NER:")
        for entity in entities:
            print(f"  • {entity.entity} ({entity.class_name}) - confidence: {entity.class_probability:.3f}")
        
        # Test compatibility interface
        names = extractor.extract_names_with_elasticsearch(test_text)
        print(f"\n👥 Extracted {len(names)} names (PERSON/ORG only):")
        for name in names:
            print(f"  • {name['name']} ({name['type']}) - confidence: {name['confidence']:.3f}")
        
        # Show statistics
        stats = extractor.get_statistics()
        print(f"\n📊 Processing Statistics:")
        print(f"  • Texts processed: {stats['texts_processed']}")
        print(f"  • Entities extracted: {stats['entities_extracted']}")
        print(f"  • Average processing time: {stats['average_processing_time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Error testing Elasticsearch NER: {e}")
        print("Make sure Elasticsearch is running and NER model is deployed.")
