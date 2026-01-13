"""
Elastic Cloud Client

Handles connection to Elastic Cloud and provides vector search capabilities
for name resolution.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np

# Optional dependency: only needed for the DEVELOPMENT fallback embeddings mode.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

class ElasticClient:
    """Advanced Elasticsearch client with latest semantic search capabilities
    
    EMBEDDING APPROACH:
    - PRIMARY: Elasticsearch native embeddings via semantic_text fields (PRODUCTION)
    - FALLBACK: External SentenceTransformers (DEVELOPMENT ONLY - NOT for production!)
    
    Features:
    - semantic_text field type for automatic embeddings
    - Inference API integration for model management
    - Hybrid search combining semantic and lexical matching
    - Advanced vector similarity with multiple algorithms
    - Multilingual semantic understanding
    
    WARNING: The SentenceTransformers fallback is for development flexibility only.
    Production systems should always use Elasticsearch native embeddings.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, allow_local_fallback: bool = True):
        """
        Initialize the ElasticClient with configuration.
        
        Args:
            config: Configuration dictionary with Elasticsearch settings
            allow_local_fallback: If True, allows fallback to local Elasticsearch
                                 when cloud credentials are missing
        """
        self.logger = logging.getLogger(__name__)
        self.allow_local_fallback = allow_local_fallback
        
        # Load configuration
        self.config = config or {}
        
        # Initialize Elasticsearch client
        self.es = self._create_elasticsearch_client()
        
        # Get index name from config
        self.index_name = self.config['elasticsearch'].get('index_name', 'entity_resolution')
        self.embedding_dimension = self.config['elasticsearch'].get('embedding_dimension', 384)
        self.similarity_threshold = self.config['elasticsearch'].get('similarity_threshold', 0.7)
        self.max_results = self.config['elasticsearch'].get('max_results', 10)
        
        # Modern semantic search configuration
        self.inference_id = self.config['elasticsearch'].get('inference_id', '.multilingual-e5-small-elasticsearch')
        self.model_id = self.config['elasticsearch'].get('model_id', '.multilingual-e5-small-elasticsearch')
        self.use_semantic_text = self.config['elasticsearch'].get('use_semantic_text', True)
        self.hybrid_search_enabled = self.config['elasticsearch'].get('hybrid_search', True)
        
        # Initialize embeddings based on configuration
        self.local_embeddings_available = False
        self.embedding_model = None
        
        if self.use_semantic_text:
            # Setup Elasticsearch native embeddings via semantic_text fields
            self.logger.info("🚀 Using Elasticsearch native embeddings via semantic_text fields")
            self._setup_inference_endpoint()
        else:
            # Initialize sentence transformer for DEVELOPMENT FALLBACK ONLY
            # WARNING: This fallback should NOT be used in production!
            # Production systems should always use Elasticsearch native embeddings via semantic_text fields
            try:
                if SentenceTransformer is None:
                    raise ImportError(
                        "sentence-transformers is not installed. "
                        "Install it to use the local embeddings fallback: pip install sentence-transformers"
                    )
                # Use multilingual E5 model for consistency with Elasticsearch inference endpoint
                self.encoder = SentenceTransformer('intfloat/multilingual-e5-small')
                self.embedding_model = self.encoder  # Add alias for compatibility
                self.local_embeddings_available = True
                self.logger.warning("⚠️  Using SentenceTransformers fallback - FOR DEVELOPMENT ONLY!")
                self.logger.warning("⚠️  Production should use Elasticsearch native embeddings via semantic_text fields")
            except Exception as e:
                self.logger.warning(f"Local embeddings unavailable: {e}")
                self.local_embeddings_available = False
                self.embedding_model = None
        
    def _create_elasticsearch_client(self) -> Elasticsearch:
        """Create and return an Elasticsearch client based on configuration
        
        Returns:
            Elasticsearch: Configured Elasticsearch client
        """
        es_cfg = (self.config or {}).get("elasticsearch", {}) if isinstance(self.config, dict) else {}

        # Check for credentials (env vars take precedence over config)
        endpoint = os.environ.get("ELASTIC_ENDPOINT") or es_cfg.get("endpoint")
        cloud_id = os.environ.get("ELASTIC_CLOUD_ID") or es_cfg.get("cloud_id")
        api_key = os.environ.get("ELASTIC_API_KEY") or es_cfg.get("api_key")

        # Prefer endpoint-based config if present (common in Searchlabs content)
        if endpoint and api_key:
            self.logger.info("Connecting to Elasticsearch via ELASTIC_ENDPOINT...")
            return Elasticsearch(
                hosts=[endpoint],
                api_key=api_key,
                request_timeout=60,  # Increase timeout for operations
                max_retries=3,       # Allow retries for transient errors
                retry_on_timeout=True  # Retry on timeout errors
            )
        
        # Try to connect to Elastic Cloud
        if cloud_id and api_key:
            self.logger.info("Connecting to Elastic Cloud...")
            return Elasticsearch(
                cloud_id=cloud_id,
                api_key=api_key,
                request_timeout=60,  # Increase timeout for operations
                max_retries=3,       # Allow retries for transient errors
                retry_on_timeout=True  # Retry on timeout errors
            )
        
        # Fallback to local Elasticsearch if allowed
        if self.allow_local_fallback:
            self.logger.warning("⚠️ Cloud credentials not found, falling back to local Elasticsearch")
            return Elasticsearch(
                "http://localhost:9200",
                request_timeout=60,  # Increase timeout for operations
                max_retries=3,       # Allow retries for transient errors
                retry_on_timeout=True  # Retry on timeout errors
            )
        else:
            self.logger.error("❌ Elastic Cloud credentials required but not found in environment")
            raise ValueError("Elastic Cloud credentials required but not found in environment")
            
    def check_connection(self) -> bool:
        """Check if the Elasticsearch connection is working
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            # Try to ping the Elasticsearch server
            info = self.es.info()
            self.logger.info(f"Connected to Elasticsearch cluster: {info.get('cluster_name', 'unknown')}")
            self.logger.info(f"Elasticsearch version: {info.get('version', {}).get('number', 'unknown')}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False
    
    def _setup_inference_endpoint(self):
        """Verify existing inference endpoint for semantic search"""
        try:
            # Check if the inference endpoint exists
            response = self.es.inference.get(inference_id=self.inference_id)
            if response:
                self.logger.info(f"Using existing inference endpoint: {self.inference_id}")
                self.logger.info(f"Endpoint task type: {response.get('task_type', 'unknown')}")
                self.logger.info(f"Endpoint service: {response.get('service', 'unknown')}")
                return True
        except Exception as e:
            self.logger.warning(f"Inference endpoint {self.inference_id} not found or error: {str(e)}")
            
        # If we reach here, the endpoint doesn't exist or there was an error
        # For .multilingual-e5-small-elasticsearch, it should exist in Elastic Cloud
        # Log a warning but don't fail - we can fall back to local embeddings
        self.logger.warning(f"Inference endpoint {self.inference_id} not available. Using local embeddings as fallback.")
        return False
    
    def setup_index(self) -> bool:
        """Create the index with proper mapping if it doesn't exist"""
        try:
            if self.es.indices.exists(index=self.index_name):
                self.logger.info(f"Index {self.index_name} already exists")
                return True
            
            # Define modern index mapping with semantic_text field
            mapping = {
                "mappings": {
                    "properties": {
                        "name": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {
                                    "type": "keyword"
                                },
                                "semantic": {
                                    "type": "semantic_text",
                                    "inference_id": self.inference_id
                                } if self.use_semantic_text else {"type": "text"}
                            }
                        },
                        "name_lower": {
                            "type": "keyword"
                        },
                        "name_semantic": {
                            "type": "semantic_text",
                            "inference_id": self.inference_id
                        } if self.use_semantic_text else {
                            "type": "dense_vector",
                            "dims": self.embedding_dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dimension,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,
                                "ef_construction": 100
                            }
                        },
                        "source": {
                            "type": "keyword"
                        },
                        "frequency": {
                            "type": "integer"
                        },
                        "context": {
                            "type": "text",
                            "fields": {
                                "semantic": {
                                    "type": "semantic_text",
                                    "inference_id": self.inference_id
                                } if self.use_semantic_text else {"type": "text"}
                            }
                        },
                        "metadata": {
                            "type": "object",
                            "enabled": True
                        },
                        "created_at": {
                            "type": "date"
                        },
                        "updated_at": {
                            "type": "date"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "multilingual_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase",
                                    "asciifolding",
                                    "stop"
                                ]
                            }
                        }
                    },
                    "index": {
                        # semantic_text fields work without custom pipelines
                    }
                }
            }
            
            # Create index
            self.es.indices.create(index=self.index_name, body=mapping)
            self.logger.info(f"Created index {self.index_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up index: {str(e)}")
            return False
    
    def index_names(self, names: List[str], source: str = "synthetic", context: str = "") -> bool:
        """Index names using modern semantic_text approach with automatic embeddings"""
        try:
            from datetime import datetime
            current_time = datetime.utcnow().isoformat()
            
            # Prepare documents for bulk indexing
            docs = []
            
            # Generate embeddings if not using semantic_text or as fallback
            # WARNING: This fallback should only be used for development!
            embeddings = None
            if not self.use_semantic_text or not self.local_embeddings_available:
                if self.local_embeddings_available and hasattr(self, 'encoder'):
                    self.logger.warning("⚠️  DEVELOPMENT FALLBACK: Using external SentenceTransformers instead of Elasticsearch native embeddings")
                    self.logger.warning("⚠️  For production, ensure semantic_text fields are properly configured")
                    embeddings = self.encoder.encode(names)
            
            for i, name in enumerate(names):
                doc_source = {
                    "name": name,
                    "name_lower": name.lower(),
                    "source": source,
                    "frequency": 1,
                    "context": context,
                    "created_at": current_time,
                    "updated_at": current_time,
                    "metadata": {
                        "indexed_with_semantic_text": self.use_semantic_text,
                        "inference_id": self.inference_id if self.use_semantic_text else None
                    }
                }
                
                # Add semantic_text field if using modern approach
                if self.use_semantic_text:
                    doc_source["name_semantic"] = name
                    if context:
                        doc_source["context"] = {"text": context, "semantic": context}
                
                # Add traditional embedding as fallback
                if embeddings is not None:
                    doc_source["embedding"] = embeddings[i].tolist()
                
                doc = {
                    "_index": self.index_name,
                    "_source": doc_source
                }
                docs.append(doc)
            
            # Bulk index with modern settings
            success, failed = bulk(
                self.es, 
                docs, 
                chunk_size=500,  # Smaller chunks for semantic processing
                request_timeout=60,  # Longer timeout for inference
                max_retries=3
            )
            
            self.logger.info(f"Indexed {success} names using {'semantic_text' if self.use_semantic_text else 'traditional'} approach, {len(failed)} failed")
            
            return len(failed) == 0
            
        except Exception as e:
            self.logger.error(f"Error indexing names with modern approach: {str(e)}")
            return False
    
    def modern_semantic_search(self, query: str, size: int = None, min_score: float = None) -> List[Dict]:
        """Modern semantic search using semantic_text and latest Elasticsearch capabilities
        
        Args:
            query: Search query text
            size: Number of results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of search results with enhanced semantic scoring
        """
        try:
            size = size or self.max_results
            min_score = min_score or self.similarity_threshold
            
            if self.use_semantic_text:
                # Use modern semantic_text approach
                search_body = {
                    "query": {
                        "bool": {
                            "should": [
                                # Semantic search on semantic_text field
                                {
                                    "semantic": {
                                        "field": "name_semantic",
                                        "query": query
                                    }
                                },
                                # Traditional text search for exact matches
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["name^3", "name.keyword^2", "context"],
                                        "type": "best_fields",
                                        "fuzziness": "AUTO"
                                    }
                                },
                                # Context semantic search if available
                                {
                                    "semantic": {
                                        "field": "context.semantic",
                                        "query": query
                                    }
                                }
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "size": size,
                    "min_score": min_score,
                    "_source": ["name", "source", "frequency", "context", "metadata", "created_at"],
                    "highlight": {
                        "fields": {
                            "name": {},
                            "context": {}
                        }
                    }
                }
            else:
                # Fallback to traditional vector + text search
                query_embedding = self.encoder.encode([query])[0].tolist() if (self.local_embeddings_available and hasattr(self, 'encoder')) else None
                
                search_body = {
                    "query": {
                        "bool": {
                            "should": []
                        }
                    },
                    "size": size,
                    "min_score": min_score
                }
                
                # Add vector search if embeddings available
                if query_embedding:
                    search_body["query"]["bool"]["should"].append({
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    })
                
                # Add text search
                search_body["query"]["bool"]["should"].append({
                    "multi_match": {
                        "query": query,
                        "fields": ["name^2", "context"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                })
            
            # Execute search
            response = self.es.search(index=self.index_name, body=search_body)
            
            # Process results with enhanced scoring
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'name': hit['_source']['name'],
                    'score': hit['_score'],
                    'source': hit['_source'].get('source', 'unknown'),
                    'frequency': hit['_source'].get('frequency', 1),
                    'context': hit['_source'].get('context', ''),
                    'metadata': hit['_source'].get('metadata', {}),
                    'highlight': hit.get('highlight', {}),
                    'search_type': 'semantic_text' if self.use_semantic_text else 'hybrid_fallback'
                }
                
                # Add semantic similarity score if using modern approach
                if self.use_semantic_text and 'semantic' in str(hit.get('matched_queries', [])):
                    result['semantic_match'] = True
                    result['confidence'] = min(hit['_score'] / 10.0, 1.0)  # Normalize semantic scores
                else:
                    result['semantic_match'] = False
                    result['confidence'] = min(hit['_score'] / 5.0, 1.0)  # Normalize text scores
                
                results.append(result)
            
            self.logger.info(f"Modern semantic search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in modern semantic search: {str(e)}")
            return []
    
    def search_names(self, query_names: List[str], use_vector: bool = True, search_mode: str = 'hybrid') -> List[Dict]:
        """Enhanced semantic search with multiple search modes
        
        Args:
            query_names: List of names to search for
            use_vector: Whether to use vector similarity search
            search_mode: 'hybrid', 'semantic', 'text', or 'fuzzy'
        """
        try:
            all_results = []
            
            for query_name in query_names:
                if search_mode == 'hybrid':
                    # Hybrid search combining vector and text with intelligent weighting
                    results = self._hybrid_semantic_search(query_name, use_vector)
                elif search_mode == 'semantic':
                    # Pure semantic search using vector embeddings
                    results = self._enhanced_vector_search(query_name)
                elif search_mode == 'fuzzy':
                    # Fuzzy matching for typos and variations
                    results = self._fuzzy_semantic_search(query_name)
                else:
                    # Traditional text search
                    results = self._text_search(query_name)
                
                all_results.extend(results)
            
            # Advanced deduplication and semantic ranking
            unique_results = self._semantic_deduplicate_and_rank(all_results, query_names)
            
            return unique_results[:self.max_results]
            
        except Exception as e:
            self.logger.error(f"Error in enhanced semantic search: {str(e)}")
            return []
    
    def _hybrid_semantic_search(self, query_name: str, use_vector: bool = True) -> List[Dict]:
        """Advanced hybrid search combining vector similarity and text matching with semantic scoring"""
        try:
            all_results = []
            
            # Vector search with enhanced scoring
            if use_vector:
                vector_results = self._enhanced_vector_search(query_name)
                all_results.extend(vector_results)
            
            # Text search with semantic boosting
            text_results = self._enhanced_text_search(query_name)
            all_results.extend(text_results)
            
            # Fuzzy search for variations
            fuzzy_results = self._fuzzy_semantic_search(query_name)
            all_results.extend(fuzzy_results)
            
            # Apply semantic similarity scoring
            scored_results = self._apply_semantic_scoring(all_results, query_name)
            
            return scored_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid semantic search: {str(e)}")
            return []
    
    def _enhanced_vector_search(self, query_name: str) -> List[Dict]:
        """Enhanced vector search with semantic similarity scoring"""
        try:
            # Check if local embeddings are available
            if not self.local_embeddings_available or not hasattr(self, 'encoder'):
                self.logger.warning("Enhanced vector search requires local embeddings - falling back to text search")
                return self._enhanced_text_search(query_name)
            
            # Generate embedding for query
            query_embedding = self.encoder.encode([query_name])[0]
            
            # Enhanced vector search with semantic filtering
            search_body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding.tolist(),
                    "k": self.max_results * 2,  # Get more candidates for better filtering
                    "num_candidates": self.max_results * 4,
                    "filter": {
                        "range": {
                            "frequency": {"gte": 1}  # Only include names with valid frequency
                        }
                    }
                },
                "_source": ["name", "source", "frequency", "metadata", "context"]
            }
            
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                # Calculate semantic similarity score
                semantic_score = self._calculate_semantic_similarity(query_name, hit['_source']['name'])
                
                result = {
                    'name': hit['_source']['name'],
                    'score': hit['_score'],
                    'semantic_score': semantic_score,
                    'combined_score': (hit['_score'] * 0.7) + (semantic_score * 0.3),  # Weighted combination
                    'source': hit['_source']['source'],
                    'frequency': hit['_source'].get('frequency', 1),
                    'search_type': 'enhanced_vector',
                    'metadata': hit['_source'].get('metadata', {}),
                    'context': hit['_source'].get('context', '')
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced vector search: {str(e)}")
            return []
    
    def _enhanced_text_search(self, query_name: str) -> List[Dict]:
        """Enhanced text search with semantic boosting and multilingual support"""
        try:
            # Prepare query variations for better matching
            query_variations = self._generate_query_variations(query_name)
            
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Exact match with highest boost
                            {
                                "term": {
                                    "name.keyword": {
                                        "value": query_name,
                                        "boost": 5.0
                                    }
                                }
                            },
                            # Fuzzy match for typos
                            {
                                "fuzzy": {
                                    "name": {
                                        "value": query_name,
                                        "fuzziness": "AUTO",
                                        "boost": 3.0
                                    }
                                }
                            },
                            # Match phrase for partial matches
                            {
                                "match_phrase": {
                                    "name": {
                                        "query": query_name,
                                        "boost": 4.0
                                    }
                                }
                            },
                            # Wildcard for partial matching
                            {
                                "wildcard": {
                                    "name_lower": {
                                        "value": f"*{query_name.lower()}*",
                                        "boost": 2.0
                                    }
                                }
                            }
                        ] + [
                            # Add variations with lower boost
                            {
                                "match": {
                                    "name": {
                                        "query": variation,
                                        "boost": 1.5
                                    }
                                }
                            } for variation in query_variations
                        ]
                    }
                },
                "_source": ["name", "source", "frequency", "metadata", "context"],
                "size": self.max_results * 2
            }
            
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                # Calculate semantic similarity for text results too
                semantic_score = self._calculate_semantic_similarity(query_name, hit['_source']['name'])
                
                result = {
                    'name': hit['_source']['name'],
                    'score': hit['_score'],
                    'semantic_score': semantic_score,
                    'combined_score': (hit['_score'] * 0.8) + (semantic_score * 0.2),  # Text search weighted higher
                    'source': hit['_source']['source'],
                    'frequency': hit['_source'].get('frequency', 1),
                    'search_type': 'enhanced_text',
                    'metadata': hit['_source'].get('metadata', {}),
                    'context': hit['_source'].get('context', '')
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced text search: {str(e)}")
            return []
    
    def _fuzzy_semantic_search(self, query_name: str) -> List[Dict]:
        """Fuzzy search optimized for typos, OCR errors, and name variations"""
        try:
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Multi-match with fuzziness
                            {
                                "multi_match": {
                                    "query": query_name,
                                    "fields": ["name^3", "name_lower^2"],
                                    "fuzziness": "AUTO",
                                    "prefix_length": 1,
                                    "max_expansions": 50,
                                    "boost": 2.0
                                }
                            },
                            # Phonetic matching (if available)
                            {
                                "match": {
                                    "name": {
                                        "query": query_name,
                                        "analyzer": "standard",
                                        "fuzziness": 2,
                                        "boost": 1.5
                                    }
                                }
                            },
                            # N-gram matching for partial similarities
                            {
                                "match": {
                                    "name": {
                                        "query": query_name,
                                        "minimum_should_match": "60%",
                                        "boost": 1.0
                                    }
                                }
                            }
                        ]
                    }
                },
                "_source": ["name", "source", "frequency", "metadata", "context"],
                "size": self.max_results
            }
            
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                # Calculate edit distance for fuzzy matching quality
                edit_distance = self._calculate_edit_distance(query_name.lower(), hit['_source']['name'].lower())
                fuzzy_score = max(0, 1.0 - (edit_distance / max(len(query_name), len(hit['_source']['name']))))
                
                result = {
                    'name': hit['_source']['name'],
                    'score': hit['_score'],
                    'fuzzy_score': fuzzy_score,
                    'edit_distance': edit_distance,
                    'combined_score': (hit['_score'] * 0.6) + (fuzzy_score * 0.4),
                    'source': hit['_source']['source'],
                    'frequency': hit['_source'].get('frequency', 1),
                    'search_type': 'fuzzy_semantic',
                    'metadata': hit['_source'].get('metadata', {}),
                    'context': hit['_source'].get('context', '')
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fuzzy semantic search: {str(e)}")
            return []
    
    def _vector_search(self, query_name: str) -> List[Dict]:
        """Perform vector similarity search"""
        try:
            # Check if local embeddings are available
            if not self.local_embeddings_available or not hasattr(self, 'encoder'):
                self.logger.warning("Vector search requires local embeddings - falling back to text search")
                return self._text_search(query_name)
            
            # Generate embedding for query
            query_embedding = self.encoder.encode([query_name])[0]
            
            # Vector search query
            search_body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding.tolist(),
                    "k": self.max_results,
                    "num_candidates": self.max_results * 2
                },
                "_source": ["name", "source", "frequency", "metadata"]
            }
            
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'name': hit['_source']['name'],
                    'score': hit['_score'],
                    'source': hit['_source']['source'],
                    'frequency': hit['_source'].get('frequency', 1),
                    'search_type': 'vector',
                    'metadata': hit['_source'].get('metadata', {})
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def _text_search(self, query_name: str) -> List[Dict]:
        """Perform text-based search"""
        try:
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "name": {
                                        "query": query_name,
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "term": {
                                    "name.keyword": {
                                        "value": query_name,
                                        "boost": 3.0
                                    }
                                }
                            },
                            {
                                "wildcard": {
                                    "name_lower": {
                                        "value": f"*{query_name.lower()}*",
                                        "boost": 1.5
                                    }
                                }
                            }
                        ]
                    }
                },
                "_source": ["name", "source", "frequency", "metadata"],
                "size": self.max_results
            }
            
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'name': hit['_source']['name'],
                    'score': hit['_score'],
                    'source': hit['_source']['source'],
                    'frequency': hit['_source'].get('frequency', 1),
                    'search_type': 'text',
                    'metadata': hit['_source'].get('metadata', {})
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in text search: {str(e)}")
            return []
    
    def _calculate_semantic_similarity(self, query_name: str, candidate_name: str) -> float:
        """Calculate semantic similarity between query and candidate names using embeddings"""
        try:
            # Check if local embeddings are available
            if not self.local_embeddings_available or not hasattr(self, 'encoder'):
                # Fallback to simple string similarity when encoder not available
                # Use normalized edit distance as a proxy for semantic similarity
                max_len = max(len(query_name), len(candidate_name))
                if max_len == 0:
                    return 1.0
                edit_dist = self._calculate_edit_distance(query_name.lower(), candidate_name.lower())
                return max(0.0, 1.0 - (edit_dist / max_len))
            
            # Generate embeddings for both names
            embeddings = self.encoder.encode([query_name, candidate_name])
            query_embedding = embeddings[0]
            candidate_embedding = embeddings[1]
            
            # Calculate cosine similarity
            dot_product = np.dot(query_embedding, candidate_embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_candidate = np.linalg.norm(candidate_embedding)
            
            if norm_query == 0 or norm_candidate == 0:
                return 0.0
            
            similarity = dot_product / (norm_query * norm_candidate)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def _generate_query_variations(self, query_name: str) -> List[str]:
        """Generate variations of the query name for better matching"""
        variations = []
        
        # Basic variations
        variations.append(query_name.lower())
        variations.append(query_name.upper())
        variations.append(query_name.title())
        
        # Remove common prefixes/suffixes
        prefixes = ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sir', 'lady']
        suffixes = ['jr.', 'sr.', 'ii', 'iii', 'iv']
        
        name_parts = query_name.lower().split()
        filtered_parts = []
        
        for part in name_parts:
            if part not in prefixes and part not in suffixes:
                filtered_parts.append(part)
        
        if filtered_parts and len(filtered_parts) != len(name_parts):
            variations.append(' '.join(filtered_parts))
        
        # First name + last name variations
        if len(name_parts) >= 2:
            variations.append(f"{name_parts[0]} {name_parts[-1]}")  # First + Last
            variations.append(name_parts[0])  # First only
            variations.append(name_parts[-1])  # Last only
        
        # Remove duplicates while preserving order
        unique_variations = []
        seen = set()
        for var in variations:
            if var not in seen and var != query_name:
                unique_variations.append(var)
                seen.add(var)
        
        return unique_variations[:5]  # Limit to avoid too many variations
    
    def _calculate_edit_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(str1) < len(str2):
            return self._calculate_edit_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _apply_semantic_scoring(self, results: List[Dict], query_name: str) -> List[Dict]:
        """Apply advanced semantic scoring to search results"""
        scored_results = []
        
        for result in results:
            # Calculate additional semantic features
            name_length_ratio = min(len(query_name), len(result['name'])) / max(len(query_name), len(result['name']))
            
            # Boost exact matches
            exact_match_boost = 2.0 if query_name.lower() == result['name'].lower() else 1.0
            
            # Boost partial matches intelligently
            partial_match_boost = 1.0
            if query_name.lower() in result['name'].lower() or result['name'].lower() in query_name.lower():
                partial_match_boost = 1.5
            
            # Frequency-based scoring (rare names get slight boost)
            frequency_score = 1.0 / (1.0 + np.log(result.get('frequency', 1)))
            
            # Calculate final semantic score
            base_score = result.get('combined_score', result.get('score', 0))
            semantic_boost = exact_match_boost * partial_match_boost * (1.0 + frequency_score * 0.1)
            
            result['final_score'] = base_score * semantic_boost * name_length_ratio
            result['semantic_features'] = {
                'name_length_ratio': name_length_ratio,
                'exact_match_boost': exact_match_boost,
                'partial_match_boost': partial_match_boost,
                'frequency_score': frequency_score,
                'semantic_boost': semantic_boost
            }
            
            scored_results.append(result)
        
        return scored_results
    
    def _semantic_deduplicate_and_rank(self, results: List[Dict], query_names: List[str]) -> List[Dict]:
        """Advanced deduplication and ranking with semantic understanding"""
        # Group similar results
        grouped_results = {}
        
        for result in results:
            name_key = result['name'].lower().strip()
            
            # Check for semantic duplicates
            found_group = None
            for existing_key in grouped_results.keys():
                similarity = self._calculate_semantic_similarity(name_key, existing_key)
                if similarity > 0.95:  # Very high similarity threshold for deduplication
                    found_group = existing_key
                    break
            
            group_key = found_group if found_group else name_key
            
            if group_key not in grouped_results:
                grouped_results[group_key] = []
            grouped_results[group_key].append(result)
        
        # Select best result from each group
        deduplicated_results = []
        for group in grouped_results.values():
            # Sort by final_score if available, otherwise by combined_score or score
            best_result = max(group, key=lambda x: x.get('final_score', x.get('combined_score', x.get('score', 0))))
            
            # Add group information
            best_result['duplicate_count'] = len(group)
            best_result['search_methods'] = list(set([r.get('search_type', 'unknown') for r in group]))
            
            deduplicated_results.append(best_result)
        
        # Final ranking by relevance to all query names
        for result in deduplicated_results:
            # Calculate relevance to all query names
            total_relevance = 0
            for query_name in query_names:
                relevance = self._calculate_semantic_similarity(query_name, result['name'])
                total_relevance += relevance
            
            result['multi_query_relevance'] = total_relevance / len(query_names)
            result['final_ranking_score'] = (
                result.get('final_score', result.get('combined_score', result.get('score', 0))) * 0.7 +
                result['multi_query_relevance'] * 0.3
            )
        
        # Sort by final ranking score
        return sorted(deduplicated_results, key=lambda x: x['final_ranking_score'], reverse=True)
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Legacy deduplication method - kept for backward compatibility"""
        seen_names = {}
        
        for result in results:
            name_lower = result['name'].lower()
            if name_lower not in seen_names or result['score'] > seen_names[name_lower]['score']:
                seen_names[name_lower] = result
        
        return list(seen_names.values())
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            count_response = self.es.count(index=self.index_name)
            
            return {
                'document_count': count_response['count'],
                'index_size': stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                'shards': stats['indices'][self.index_name]['total']['docs']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def delete_index(self) -> bool:
        """Delete the index (use with caution)"""
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                self.logger.info(f"Deleted index {self.index_name}")
                return True
            else:
                self.logger.info(f"Index {self.index_name} does not exist")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting index: {str(e)}")
            return False
