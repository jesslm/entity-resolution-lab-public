#!/usr/bin/env python3
"""
Notebook Generator for Entity Resolution Tutorial

This module generates a Jupyter notebook that demonstrates the entity resolution system
using the minimal datasets and following the structure of the run_pipeline.py script.
"""

import os
import sys
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path


class TutorialNotebookGenerator:
    """
    Generator for the Entity Resolution Tutorial Notebook.
    """
    
    def __init__(self, output_path=None):
        """
        Initialize the notebook generator.
        
        Args:
            output_path: Path where the notebook will be saved. If None, saves to current directory.
                If output_path is a directory, the notebook will be saved in that directory with the default filename.
                If output_path is a file path, the notebook will be saved with that filename.
        """
        # Process output path
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix == '.ipynb':  # It's a file path
                self.output_dir = output_path.parent
                self.filename = output_path.name
            else:  # It's a directory
                self.output_dir = output_path
                self.filename = "entity_resolution_tutorial.ipynb"
        else:
            self.output_dir = Path(os.getcwd())
            self.filename = "entity_resolution_tutorial.ipynb"
            
        self.notebook = new_notebook()
        
    def generate(self):
        """
        Generate the complete tutorial notebook.
        """
        self._add_title_and_introduction()
        self._add_setup_section()
        self._add_entity_preparation_section()
        self._add_article_processing_section()
        self._add_entity_matching_section()
        self._add_visualization_section()
        self._add_conclusion()
        
        return self.notebook
    
    def save(self, force=False):
        """
        Save the generated notebook to a file.
        
        Args:
            force: If True, overwrite existing file if it exists.
        
        Returns:
            str: Path to the saved notebook file.
            
        Raises:
            FileExistsError: If the file exists and force is False.
        """
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the output file path
        output_file = self.output_dir / self.filename
        
        # Check if file exists and force is False
        if output_file.exists() and not force:
            raise FileExistsError(f"File {output_file} already exists. Use --force to overwrite.")
        
        # Save the notebook
        with open(output_file, 'w', encoding='utf-8') as f:
            nbf.write(self.notebook, f)
            
        print(f"Tutorial notebook saved to: {output_file}")
        return str(output_file)
    
    def _add_title_and_introduction(self):
        """Add title and introduction to the notebook."""
        self.notebook.cells.append(new_markdown_cell("""
# Intelligent Entity Resolution Tutorial

Welcome to this interactive tutorial on the Entity Resolution system! This notebook will guide you through the process of resolving entities using Elasticsearch and LLM technologies.

## What is Entity Resolution?

Entity resolution is the process of identifying and linking mentions of the same entity across different data sources. For example, recognizing that "Tim Cook", "Apple CEO", and "Tim Apple" all refer to the same person.

## What You'll Learn

In this tutorial, you'll learn how to:

1. **Prepare Entities**: Create and enrich a watch list of entities
2. **Process Articles**: Extract entities from articles using named entity recognition and patterns
3. **Match Entities**: Match extracted entities against your watch list
4. **Visualize Results**: Visualize the matching results

## Pipeline Overview

The entity resolution pipeline consists of three main stages:

1. **Entity Preparation**: Creating and enriching a watch list of entities
2. **Article Processing**: Processing articles to extract entities
3. **Entity Matching**: Matching extracted entities against the watch list

Let's get started!
"""))

    def _add_setup_section(self):
        """Add setup section to the notebook."""
        self.notebook.cells.append(new_markdown_cell("""
## Installation and Setup

First, let's make sure the entity_resolution_demo package is installed and configured correctly.
"""))

        self.notebook.cells.append(new_code_cell("""
# Ensure the package is in the Python path
import sys
import os
from pathlib import Path

# Define potential package paths
project_paths = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/Users/jmoszko/CascadeProjects/name-resolution-llm")
]

# Add all potential package paths to sys.path
for path in project_paths:
    # Try the src directory structure
    src_path = path / "entity_resolution_demo_package" / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")
    
    # Try direct package directory
    pkg_path = path / "entity_resolution_demo"
    if pkg_path.exists() and str(pkg_path) not in sys.path:
        sys.path.insert(0, str(pkg_path))
        print(f"✅ Added {pkg_path} to Python path")

# Print the current Python path for debugging
print("Current Python path:")
for p in sys.path[:5]:  # Show first 5 paths
    print(f"  {p}")

# Check if the package is installed
try:
    import entity_resolution_demo
    print(f"✅ Successfully imported entity_resolution_demo package")
    print(f"Package location: {entity_resolution_demo.__file__}")
except ImportError as e:
    print(f"❌ Could not import entity_resolution_demo: {e}")
    print("Please make sure the package is installed or in the Python path.")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### Environment Configuration

The entity resolution system requires several environment variables to be set:

- `ELASTIC_CLOUD_ID`: Your Elasticsearch Cloud ID
- `ELASTIC_API_KEY`: Your Elasticsearch API key
- `OPENAI_API_KEY`: Your OpenAI API key (for LLM explanations)

Let's set these up:
"""))

        self.notebook.cells.append(new_code_cell("""
# Import required libraries
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Try to load from .env file
load_dotenv()

# Check if environment variables are set
elastic_cloud_id = os.environ.get("ELASTIC_CLOUD_ID")
elastic_api_key = os.environ.get("ELASTIC_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Display status
print(f"ELASTIC_CLOUD_ID: {'✅ Set' if elastic_cloud_id else '❌ Not set'}")
print(f"ELASTIC_API_KEY: {'✅ Set' if elastic_api_key else '❌ Not set'}")
print(f"OPENAI_API_KEY: {'✅ Set' if openai_api_key else '❌ Not set'}")

# Simple helper for setting environment variables if needed
if not all([elastic_cloud_id, elastic_api_key]):
    print("")
    print("⚠️ Note: This tutorial requires Elasticsearch connection.")
    print("  Please set ELASTIC_CLOUD_ID and ELASTIC_API_KEY environment variables.")
    
    # Allow setting them here for convenience
    from getpass import getpass
    if not elastic_cloud_id:
        elastic_cloud_id = getpass("Enter your ELASTIC_CLOUD_ID: ")
        os.environ["ELASTIC_CLOUD_ID"] = elastic_cloud_id
        
    if not elastic_api_key:
        elastic_api_key = getpass("Enter your ELASTIC_API_KEY: ")
        os.environ["ELASTIC_API_KEY"] = elastic_api_key
        
    print("Environment variables set for this session.")
"""))

        # Add a separate cell for imports with error handling
        self.notebook.cells.append(new_markdown_cell("""
### Import Components

Now let's import the entity resolution components:
"""))

        self.notebook.cells.append(new_code_cell("""
# Try to import entity resolution components with error handling
try:
    # Import pipeline modules
    from entity_resolution_demo.pipeline_runner.config import load_config
    from entity_resolution_demo.pipeline_runner.utils import print_header, print_subheader, print_success, print_warning, print_error, print_info
    
    # Import components directly
    from entity_resolution_demo.search.elastic_client import ElasticClient
    from entity_resolution_demo.entity_preparation.entity_watch_list import EntityWatchList, WatchedEntity
    from entity_resolution_demo.entity_preparation.entity_enricher import EntityEnricher
    from entity_resolution_demo.entity_preparation.entity_indexer import EntityIndexer
    from entity_resolution_demo.article_processing.article_processor import Article, ProcessedArticle, ExtractedEntity, ArticleProcessor
    from entity_resolution_demo.entity_matching.real_time_entity_matcher import RealTimeEntityMatcher
    from entity_resolution_demo.entity_matching.elasticsearch_entity_matcher import ElasticsearchEntityMatcher
    from entity_resolution_demo.entity_matching.entity_match import EntityMatch, MatchingResult, PotentialMatch
    from entity_resolution_demo.entity_matching.enhanced_batch_match_judge import EnhancedBatchMatchJudge
    
    print("✅ Successfully imported all components")
    
    # Load configuration
    config = load_config()
    
    # Create Elasticsearch client
    elastic_client = ElasticClient(config)
    print(f"Elasticsearch connection: {'✅ Connected' if elastic_client.es else '❌ Not connected'}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️ Please make sure the entity_resolution_demo package is installed correctly.")
    print("  You may need to run: pip install -e /path/to/entity_resolution_demo_package")
    print("  or add the package directory to your PYTHONPATH.")
    
    # Define placeholder variables to allow the notebook to continue
    config = {}
    elastic_client = None
"""))

        self.notebook.cells.append(new_markdown_cell("""
### Load Minimal Datasets

For this tutorial, we'll use minimal datasets that demonstrate various entity resolution challenges.
"""))

        self.notebook.cells.append(new_code_cell("""
# Load minimal entity data
import os
from pathlib import Path

# Try different paths to find the minimal datasets
try:
    # Search in multiple potential locations
    potential_paths = [
        # Current directory
        Path("minimal_entities.json"),
        Path("minimal_articles.json"),
        # Notebooks directory
        Path("notebooks/minimal_entities.json"),
        Path("notebooks/minimal_articles.json"),
        # Package data directory
        Path("entity_resolution_demo/data/minimal_entities.json"),
        Path("entity_resolution_demo/data/minimal_articles.json"),
        # Development environment paths
        Path("/Users/jmoszko/CascadeProjects/name-resolution-llm/minimal_entities.json"),
        Path("/Users/jmoszko/CascadeProjects/name-resolution-llm/minimal_articles.json"),
        # Package installation paths
        Path(sys.prefix) / "share/entity_resolution_demo/data/minimal_entities.json",
        Path(sys.prefix) / "share/entity_resolution_demo/data/minimal_articles.json"
    ]
    
    # Find the first existing entities file
    entities_path = None
    for path in potential_paths:
        if "entities" in str(path) and path.exists():
            entities_path = path
            break
    
    # Find the first existing articles file
    articles_path = None
    for path in potential_paths:
        if "articles" in str(path) and path.exists():
            articles_path = path
            break
            
    # Check if we found the files
    if not entities_path or not articles_path:
        raise FileNotFoundError("Could not find minimal_entities.json or minimal_articles.json")

    
    # Display paths for debugging
    print(f"Using entities file: {entities_path}")
    print(f"Using articles file: {articles_path}")
    
    # Load entity data
    with open(entities_path, "r") as f:
        entity_data = json.load(f)
    
    entities = entity_data["entities"]
    enrichment_config = entity_data["enrichment"]
    
    # Load article data
    with open(articles_path, "r") as f:
        article_data = json.load(f)
    
    articles = article_data["articles"]
    ner_config = article_data["ner"]
    
    # Display summary
    print(f"Loaded {len(entities)} entities and {len(articles)} articles")
    
    # Display a sample entity
    print("Sample entity:")
    sample_entity = entities[0]
    for key, value in sample_entity.items():
        print(f"  {key}: {value}")

    # Display a sample article
    print("Sample article:")
    sample_article = articles[0]
    for key, value in sample_article.items():
        print(f"  {key}: {value}")
except FileNotFoundError as e:
    print(f"Warning: Could not find the minimal datasets: {e}")
    print("Creating minimal example datasets for demonstration purposes...")
    
    # Create minimal entities
    entities = [
        {
            "name": "Tim Cook",
            "entity_type": "person",
            "description": "CEO of Apple Inc.",
            "aliases": ["Apple CEO", "Tim Apple"],
            "explicit_context": "Timothy Donald Cook (born November 1, 1960) is an American business executive who has been the chief executive officer of Apple Inc. since 2011."
        },
        {
            "name": "Linus Torvalds",
            "entity_type": "person",
            "description": "CEO of Tesla and Git",
            "aliases": ["Linux creator", "Git CEO", "エロン・マスク"],
            "explicit_context": "Linus Benedict Torvalds (born June 28, 1971) is a business magnate and investor. He is the founder, chairman, CEO, and CTO of Git; angel investor, CEO, product architect and former chairman of Tesla, Inc."
        },
        {
            "name": "Joe Biden",
            "entity_type": "person",
            "description": "46th President of the United States",
            "aliases": ["President Biden", "POTUS"],
            "explicit_context": "Joseph Robinette Biden Jr. (born November 20, 1942) is an American politician who is the 46th and current president of the United States."
        }
    ]
    
    # Create minimal articles
    articles = [
        {
            "id": "article1",
            "title": "Tech CEO News",
            "content": "Tim Cook announced new Apple products today. The Apple CEO spoke about innovation and the future of technology.",
            "source": "Tech News",
            "language": "en"
        },
        {
            "id": "article2",
            "title": "Space Exploration Update",
            "content": "Linus Torvalds's Git launched another rocket today. The Linux creator continues to push boundaries in space technology.",
            "source": "Science Daily",
            "language": "en"
        },
        {
            "id": "article3",
            "title": "White House Briefing",
            "content": "President Biden addressed the nation today on important policy matters. POTUS emphasized the need for unity.",
            "source": "Politics Today",
            "language": "en"
        }
    ]
    
    # Create sample entity and article for display
    sample_entity = entities[0]
    sample_article = articles[0]
    
    # Create enrichment and NER config
    enrichment_config = {"use_wikipedia": True, "languages": ["en"]}
    ner_config = {"use_elasticsearch": True}
    
    print("✅ Created minimal example datasets")
    
    # Optionally save the datasets for future use
    try:
        # Save entities
        entities_path = Path("minimal_entities.json")
        with open(entities_path, "w") as f:
            json.dump({"entities": entities, "enrichment": enrichment_config}, f, indent=2)
            
        # Save articles
        articles_path = Path("minimal_articles.json")
        with open(articles_path, "w") as f:
            json.dump({"articles": articles, "ner": ner_config}, f, indent=2)
            
        print(f"✅ Saved minimal datasets to {entities_path} and {articles_path}")
    except Exception as save_error:
        print(f"Note: Could not save minimal datasets: {save_error}")
        print("Continuing with in-memory datasets...")
        entities_path = Path("minimal_entities.json")
        articles_path = Path("minimal_articles.json")
"""))

    def _add_entity_preparation_section(self):
        """Add entity preparation section to the notebook."""
        self.notebook.cells.append(new_markdown_cell("""
## Entity Preparation

The first stage of the pipeline is entity preparation. This involves:

1. **Creating a Watch List**: Creating a list of entities to watch for
2. **Enriching Entities**: Adding context to entities using Wikipedia or explicit context
3. **Indexing Entities**: Indexing entities in Elasticsearch for efficient searching

Let's go through each step.
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 1. Creating a Watch List

We'll create a watch list of entities from our minimal dataset.
"""))

        self.notebook.cells.append(new_code_cell("""
# Import the EntityWatchList class
from entity_resolution_demo.entity_preparation.entity_watch_list import EntityWatchList

# Create a watch list
watch_list = EntityWatchList()

# Add entities to the watch list
for entity in entities:
    # Extract entity information
    name = entity.get("name")
    entity_type = entity.get("entity_type", "person")
    description = entity.get("description", "")
    aliases = entity.get("aliases", [])
    
    # Add entity to watch list with aliases
    entity_id = watch_list.add_entity(
        name=name,
        entity_type=entity_type,
        description=description,
        priority="high",
        aliases=aliases
    )

# Display watch list summary
print(f"Created watch list with {len(watch_list.get_all_entities())} entities")

# Display a few entities from the watch list
print("Entities in watch list:")
for i, entity in enumerate(watch_list.get_all_entities()[:5]):
    print(f"  {i+1}. {entity.name} ({entity.entity_type}) - {entity.description}")
    if hasattr(entity, 'aliases') and entity.aliases:
        print(f"     Aliases: {', '.join(entity.aliases)}")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 2. Enriching Entities

Next, we'll enrich the entities with additional context. This can come from:

- **Wikipedia**: Automatically fetched based on entity name and description
- **Explicit Context**: Provided directly in the entity data

Enrichment helps improve matching accuracy by providing more context for semantic matching.
"""))

        self.notebook.cells.append(new_code_cell("""
# Import the EntityEnricher class
from entity_resolution_demo.entity_preparation.entity_enricher import EntityEnricher

# Create entity enricher
enricher = EntityEnricher(elasticsearch_client=elastic_client)

# Enrich entities
enriched_entities = []
for entity in entities:
    # Extract entity information
    name = entity.get("name")
    entity_type = entity.get("entity_type", "person")
    description = entity.get("description", "")
    explicit_context = entity.get("explicit_context", None)
    
    # Combine description and explicit context if available
    source_context = description
    if explicit_context:
        source_context = f"{description}. {explicit_context}" if description else explicit_context
    
    # Enrich entity
    enriched_entity = enricher.enrich_entity(
        name=name,
        source_context=source_context
    )
    
    # Add to list of enriched entities
    enriched_entities.append(enriched_entity)

# Display enrichment summary
print(f"Enriched {len(enriched_entities)} entities")

# Display a few enriched entities
print("Enriched entities:")
for i, entity in enumerate(enriched_entities[:3]):
    print(f"{i+1}. {entity.name}")
    print(f"   Confidence: {entity.confidence_score:.2f}")
    print(f"   Source: {entity.enrichment_source}")
    print(f"   Context: {entity.entity_context[:150]}...")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 3. Indexing Entities

Finally, we'll index the enriched entities in Elasticsearch. This creates a searchable index with:

- **Text Fields**: For exact and fuzzy matching
- **Semantic Text Fields**: For semantic matching using embeddings

Indexing enables efficient searching and matching of entities.
"""))

        self.notebook.cells.append(new_code_cell("""
# Import the EntityIndexer class
from entity_resolution_demo.entity_preparation.entity_indexer import EntityIndexer

# Create entity indexer
entity_indexer = EntityIndexer(elastic_client, config)
index_name = entity_indexer.entity_index
print(f"Working with Elasticsearch index: {index_name}")

# Create a fresh index
if elastic_client.es.indices.exists(index=index_name):
    print(f"Index {index_name} already exists, deleting it for a fresh start...")
    elastic_client.es.indices.delete(index=index_name)

# Create the index with proper mappings
entity_indexer.create_indices()
print(f"✅ Created index: {index_name}")

# Index our enriched entities
indexed_count = 0
for entity in enriched_entities:
    success = entity_indexer.index_entity(entity)
    if success:
        indexed_count += 1

# Refresh the index to make entities immediately searchable
entity_indexer.refresh_indices()
print(f"✅ Successfully indexed {indexed_count} entities in {index_name}")

# Test a simple search to verify indexing
search_result = elastic_client.es.search(
    index=index_name,
    body={
        "query": {
            "match": {
                "entity_name": "Putin"
            }
        }
    }
)

# Display search results
hits = search_result["hits"]["hits"]
print(f"Found {len(hits)} matches for 'Putin':")
for hit in hits:
    print(f"  - {hit['_source']['entity_name']} (Score: {hit['_score']})")
    print(f"    Context: {hit['_source']['context'][:100]}...")
"""))

    def _add_article_processing_section(self):
        """Add article processing section to the notebook."""
        self.notebook.cells.append(new_markdown_cell("""
## Article Processing

The second stage of the pipeline is article processing. This involves:

1. **Creating Articles**: Creating article objects from raw data
2. **Processing Articles**: Extracting entities from articles using NER
3. **Indexing Articles**: Indexing processed articles in Elasticsearch

Let's go through each step.
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 1. Creating Articles

First, we'll create article objects from our minimal dataset.
"""))

        self.notebook.cells.append(new_code_cell("""
# Import the Article class
from entity_resolution_demo.article_processing.article_processor import Article

# Create articles
article_objects = []
for article_data in articles:
    # Create article object
    article = Article(
        id=article_data["id"],
        title=article_data["title"],
        content=article_data["content"],
        source=article_data["source"],
        language=article_data.get("language", "en")
    )
    
    # Add to list of articles
    article_objects.append(article)

# Display article summary
print(f"Created {len(article_objects)} articles")

# Display a few articles
print("Articles:")
for i, article in enumerate(article_objects[:3]):
    print(f"{i+1}. {article.title} (ID: {article.id})")
    print(f"   Source: {article.source}")
    print(f"   Content: {article.content[:100]}...")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 2. Processing Articles

Next, we'll process the articles to extract entities using Named Entity Recognition (NER).

The article processor uses a HybridNERExtractor that combines:
- **Elasticsearch NER**: For extracting named entities (PERSON, ORGANIZATION, LOCATION)
- **Pattern-Based Extraction**: For extracting titles, roles, and other entity references

This hybrid approach provides better coverage of entity mentions.
"""))

        self.notebook.cells.append(new_code_cell("""
# Import the ArticleProcessor class
from entity_resolution_demo.article_processing.article_processor import ArticleProcessor

# Create article processor
article_processor = ArticleProcessor(config=config, elasticsearch_client=elastic_client)

# Process articles
processed_articles = []
for article in article_objects:
    # Process article
    processed_article = article_processor.process_article(article)
    
    # Add to list of processed articles
    processed_articles.append(processed_article)

# Display processing summary
print(f"Processed {len(processed_articles)} articles")

# Display extracted entities
print("Extracted entities:")
total_entities = 0
for i, article in enumerate(processed_articles[:3]):
    entities = article.extracted_entities
    total_entities += len(entities)
    print(f"{i+1}. {article.article.title} - {len(entities)} entities:")
    for j, entity in enumerate(entities[:5]):
        print(f"   {j+1}. {entity.name} ({entity.entity_type})")
        if hasattr(entity, 'context') and entity.context:
            print(f"      Context: {entity.context[:100]}...")

print(f"Total extracted entities across all articles: {sum(len(a.extracted_entities) for a in processed_articles)}")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 3. Indexing Articles

Finally, we'll index the processed articles in Elasticsearch. This creates a searchable index with:

- **Article Content**: For searching article text
- **Extracted Entities**: For searching by entity
- **Metadata**: For filtering by source, language, etc.

Indexing enables efficient searching and analysis of articles.
"""))

        self.notebook.cells.append(new_code_cell("""
# Import time for unique index name
import time

# Create article index
article_index = f"demo_article_index_{int(time.time())}"
print(f"Working with Elasticsearch index: {article_index}")

# Create mapping for article index
mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text"},
            "content": {"type": "text"},
            "source": {"type": "keyword"},
            "language": {"type": "keyword"},
            "extracted_entities": {
                "type": "nested",
                "properties": {
                    "name": {"type": "text"},
                    "entity_type": {"type": "keyword"},
                    "context": {"type": "text"}
                }
            }
        }
    }
}

# Create index
if elastic_client.es.indices.exists(index=article_index):
    print(f"Index {article_index} already exists, deleting it for a fresh start...")
    elastic_client.es.indices.delete(index=article_index)

elastic_client.es.indices.create(index=article_index, body=mapping)
print(f"✅ Created index: {article_index}")

# Index articles
indexed_count = 0
for article in processed_articles:
    # Create document
    doc = {
        "id": article.article.id,
        "title": article.article.title,
        "content": article.article.content,
        "source": article.article.source,
        "language": article.article.language,
        "extracted_entities": [
            {
                "name": entity.name,
                "entity_type": entity.entity_type,
                "context": entity.context if entity.context else ""
            }
            for entity in article.extracted_entities
        ]
    }
    
    # Index document
    elastic_client.es.index(index=article_index, id=article.article.id, body=doc)
    indexed_count += 1

# Refresh the index to make articles immediately searchable
elastic_client.es.indices.refresh(index=article_index)
print(f"✅ Successfully indexed {indexed_count} articles in {article_index}")

# Test a simple search to verify indexing
search_result = elastic_client.es.search(
    index=article_index,
    body={
        "query": {
            "match": {
                "content": "Putin"
            }
        }
    }
)

# Display search results
hits = search_result["hits"]["hits"]
print(f"Found {len(hits)} articles mentioning 'Putin':")
for hit in hits:
    print(f"  - {hit['_source']['title']} (Score: {hit['_score']})")
    print(f"    Content: {hit['_source']['content'][:100]}...")
"""))

    def _add_entity_matching_section(self):
        """Add entity matching section to the notebook."""
        self.notebook.cells.append(new_markdown_cell("""
## Entity Matching

The third and final stage of the pipeline is entity matching. This involves:

1. **Matcher Initialization**: Setting up the RealTimeEntityMatcher
2. **Matching Loop**: Matching extracted entities against the watch list
3. **Match Judging**: Using LLMs to explain and validate matches

This is where the real magic of entity resolution happens!
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 1. Matcher Initialization

First, we'll initialize the RealTimeEntityMatcher, which is the core component responsible for matching entities.

The matcher uses several components:
- **EntityWatchList**: The list of entities to match against
- **ElasticClient**: For performing searches against the entity index
- **Configuration**: For controlling matching behavior and thresholds
"""))

        self.notebook.cells.append(new_code_cell("""
# Import time for timing measurements
import time

# Import the RealTimeEntityMatcher class
from entity_resolution_demo.entity_matching.real_time_entity_matcher import RealTimeEntityMatcher

# Configure matching parameters
matching_config = {
    "match_threshold": 0.1,  # Lower threshold to catch more potential matches
    "semantic_similarity_threshold": 0.1,  # Lower threshold for semantic matching
    "hybrid_search_enabled": True,  # Enable hybrid search (lexical + semantic)
    "matching": {
        "use_llm_explanations": True  # Enable LLM explanations
    },
    "llm": {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 500,
        "batch_size": 5
    }
}

# Update config with matching parameters
for key, value in matching_config.items():
    config[key] = value

# Create the RealTimeEntityMatcher
matcher = RealTimeEntityMatcher(
    watch_list=watch_list,
    elastic_client=elastic_client,
    config=config
)

# Display matcher configuration
print("RealTimeEntityMatcher initialized with:")
print(f"  - Watch list: {len(watch_list.get_all_entities())} entities")
print(f"  - Match threshold: {matcher.es_matcher.match_threshold}")
print(f"  - Semantic similarity threshold: {matcher.es_matcher.semantic_similarity_threshold}")
print(f"  - Hybrid search: {'Enabled' if matcher.es_matcher.hybrid_search_enabled else 'Disabled'}")
print(f"  - LLM explanations: {'Enabled' if matcher.use_llm_explanations else 'Disabled'}")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 2. Matching Loop

Next, we'll match each processed article against the watch list. This is the core of the entity resolution process.

The matching process follows these steps for each article:
1. Extract entities from the article (already done in the article processing stage)
2. For each extracted entity, search for potential matches in the watch list
3. Score and rank the potential matches
4. Apply LLM reasoning to validate and explain matches

Let's see it in action!
"""))

        self.notebook.cells.append(new_code_cell("""
# Match articles against watch list
matching_results = []
start_time = time.time()

for article in processed_articles:
    # Match article against watch list
    print(f"Matching article: {article.article.title}")
    try:
        result = matcher.match_article(article)
    except Exception as e:
        print(f"Error matching article: {e}")
        continue
    
    # Add to list of matching results
    matching_results.append(result)
    
    # Display matches found
    if result.matches_found:
        print(f"  ✅ Found {len(result.matches_found)} matches:")
        for i, match in enumerate(result.matches_found[:3]):  # Show up to 3 matches
            print(f"    {i+1}. {match.extracted_entity.name} → {match.watched_entity.name} ({match.match_type}, confidence: {match.confidence:.2f})")
    else:
        print("  ❌ No matches found")

end_time = time.time()
print(f"Matched {len(processed_articles)} articles in {end_time - start_time:.2f} seconds")

# Display matching summary
total_matches = sum(len(result.matches_found) for result in matching_results)
print(f"Found {total_matches} matches across {len(matching_results)} articles")

# Display match types distribution
match_types = {}
for result in matching_results:
    for match in result.matches_found:
        match_type = match.match_type
        match_types[match_type] = match_types.get(match_type, 0) + 1

print("Match types distribution:")
for match_type, count in sorted(match_types.items()):
    print(f"  - {match_type}: {count}")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 3. Match Judging with LLMs

One of the most powerful features of our entity resolution system is the use of LLMs to explain and validate matches.

The EnhancedBatchMatchJudge component:
1. Takes potential matches from the matcher
2. Sends them to an LLM (like GPT-3.5 or GPT-4)
3. Gets back detailed explanations and confidence scores
4. Structures the explanations for easy consumption

Let's examine some of the LLM-powered explanations:
"""))

        self.notebook.cells.append(new_code_cell("""
# Create a DataFrame to display matches with explanations
import pandas as pd

# Collect match data
match_data = []
for result in matching_results:
    try:
        for match in result.matches_found:
            # Extract match information
            explanation = match.reasoning if hasattr(match, 'reasoning') and match.reasoning else ""
            match_data.append({
                "Article": result.article_title,
                "Extracted Entity": match.extracted_entity.name,
                "Watched Entity": match.watched_entity.name,
                "Match Type": match.match_type,
                "Confidence": match.confidence,
                "Explanation": explanation[:150] + "..." if explanation and len(explanation) > 150 else explanation
            })
    except Exception as e:
        print(f"Error collecting match data: {e}")
        continue

# Create DataFrame
if match_data:
    matches_df = pd.DataFrame(match_data)
    
    # Display high confidence matches
    print("High confidence matches (confidence >= 0.8):")
    high_conf = matches_df[matches_df["Confidence"] >= 0.8].sort_values("Confidence", ascending=False)
    if not high_conf.empty:
        display(high_conf)
    else:
        print("No high confidence matches found.")
    
    # Display medium confidence matches
    print("\nMedium confidence matches (0.5 <= confidence < 0.8):")
    med_conf = matches_df[(matches_df["Confidence"] >= 0.5) & (matches_df["Confidence"] < 0.8)].sort_values("Confidence", ascending=False)
    if not med_conf.empty:
        display(med_conf)
    else:
        print("No medium confidence matches found.")
    
    # Display low confidence matches
    print("\nLow confidence matches (confidence < 0.5):")
    low_conf = matches_df[matches_df["Confidence"] < 0.5].sort_values("Confidence", ascending=False).head(5)  # Show only top 5
    if not low_conf.empty:
        display(low_conf)
    else:
        print("No low confidence matches found.")
else:
    print("No match data available to display.")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 4. Deep Dive into Match Types

Let's examine the different types of matches our system found:

1. **Exact Matches**: Direct string matches (e.g., "Leo Tolstoy" → "Leo Tolstoy")
2. **Alias Matches**: Matches against entity aliases (e.g., "Putin" → "Leo Tolstoy")
3. **Semantic Matches**: Matches based on semantic similarity (e.g., "Russian President" → "Leo Tolstoy")
4. **Partial Matches**: Matches on part of the name (e.g., "Diaz" → "Carlos Alfonzo Diaz")
5. **Nickname Matches**: Matches on nicknames (e.g., "Bill Johnson" → "William Johnson")

Let's look at examples of each type:
"""))

        self.notebook.cells.append(new_code_cell("""
# Function to find examples of each match type
def find_match_examples(match_type, matches_df, n=1):
    try:
        if matches_df is not None and not matches_df.empty:
            examples = matches_df[matches_df["Match Type"] == match_type].sort_values("Confidence", ascending=False).head(n)
            if len(examples) == 0:
                return pd.DataFrame()
            return examples
        return pd.DataFrame()
    except Exception as e:
        print(f"Error finding examples for {match_type}: {e}")
        return pd.DataFrame()

# Display examples of each match type
if 'matches_df' in locals() and not matches_df.empty:
    match_types = ["exact", "alias", "semantic", "hybrid", "direct"]
    
    for match_type in match_types:
        try:
            examples = find_match_examples(match_type, matches_df)
            if not examples.empty:
                print(f"\n{match_type.upper()} MATCH EXAMPLE:")
                display(examples)
        except Exception as e:
            print(f"Error displaying examples for {match_type}: {e}")
else:
    print("\nNo match data available to display examples.")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 5. Indexing Match Results

Finally, we'll index the match results in Elasticsearch for future analysis and visualization.

This creates a searchable index with:
- **Match Information**: Extracted entity, watched entity, confidence, etc.
- **Article Information**: Article ID, title, source, etc.
- **LLM Explanations**: Reasoning, confidence factors, key evidence, etc.

Indexing enables efficient searching and analysis of match results.
"""))

        self.notebook.cells.append(new_code_cell("""
# Create match results index
match_results_index = f"demo_match_results_{int(time.time())}"
print(f"Working with Elasticsearch index: {match_results_index}")

# Create mapping for match results index
mapping = {
    "mappings": {
        "properties": {
            "match_id": {"type": "keyword"},
            "match_timestamp": {"type": "date"},
            "extracted_entity": {"type": "text"},
            "extracted_entity_type": {"type": "keyword"},
            "extracted_entity_context": {"type": "text"},
            "watched_entity": {"type": "text"},
            "watched_entity_type": {"type": "keyword"},
            "watched_entity_priority": {"type": "keyword"},
            "match_type": {"type": "keyword"},
            "is_match": {"type": "boolean"},
            "confidence": {"type": "float"},
            "article_id": {"type": "keyword"},
            "article_title": {"type": "text"},
            "article_source": {"type": "keyword"},
            "reasoning": {"type": "text"},
            "explanation": {"type": "text"},
            "explanation_full": {"type": "text"},
            "confidence_factors": {"type": "object"},
            "key_evidence": {"type": "text"},
            "risk_factors": {"type": "text"}
        }
    }
}

# Create index
if elastic_client.es.indices.exists(index=match_results_index):
    print(f"Index {match_results_index} already exists, deleting it for a fresh start...")
    elastic_client.es.indices.delete(index=match_results_index)

elastic_client.es.indices.create(index=match_results_index, body=mapping)
print(f"✅ Created index: {match_results_index}")

# Index match results
indexed_count = 0
for result in matching_results:
    for match in result.matches_found:
        # Create document
        doc = {
            "match_id": f"match_{int(time.time())}_{indexed_count:03d}",
            "match_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "extracted_entity": match.extracted_entity.name,
            "extracted_entity_type": match.extracted_entity.entity_type,
            "extracted_entity_context": match.extracted_entity.context if hasattr(match.extracted_entity, 'context') else "",
            "watched_entity": match.watched_entity.name,
            "watched_entity_type": match.watched_entity.entity_type,
            "watched_entity_priority": match.watched_entity.priority if hasattr(match.watched_entity, 'priority') else "",
            "match_type": match.match_type,
            "is_match": match.is_match if hasattr(match, 'is_match') else (match.confidence >= 0.3),
            "confidence": match.confidence,
            "article_id": result.article_id,
            "article_title": result.article_title,
            "article_source": result.article.source if hasattr(result, 'article') else "",
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
        elastic_client.es.index(index=match_results_index, body=doc)
        indexed_count += 1

# Refresh the index to make match results immediately searchable
elastic_client.es.indices.refresh(index=match_results_index)
print(f"✅ Successfully indexed {indexed_count} matches in {match_results_index}")

# Test a simple search to verify indexing
search_result = elastic_client.es.search(
    index=match_results_index,
    body={
        "query": {
            "match": {
                "watched_entity": "Putin"
            }
        }
    }
)

# Display search results
hits = search_result["hits"]["hits"]
print(f"Found {len(hits)} matches for 'Putin':")
for hit in hits:
    print(f"  - {hit['_source']['extracted_entity']} → {hit['_source']['watched_entity']} (Confidence: {hit['_source']['confidence']})")
    print(f"    Reasoning: {hit['_source']['reasoning'][:100]}...")
"""))

    def _add_visualization_section(self):
        """Add visualization section to the notebook."""
        self.notebook.cells.append(new_markdown_cell("""
## Visualization and Analysis

Now that we have completed the entity resolution pipeline, let's visualize and analyze the results.

We'll create several visualizations to help understand the entity resolution results:

1. **Match Distribution**: Distribution of matches by type and confidence
2. **Entity Network**: Network graph of entities and their matches
3. **Match Timeline**: Timeline of matches across articles

These visualizations help identify patterns and insights in the entity resolution results.
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 1. Match Distribution

First, let's visualize the distribution of matches by type and confidence.
"""))

        self.notebook.cells.append(new_code_cell("""
# Create visualizations of match distribution
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure size
plt.figure(figsize=(15, 10))

# 1. Match types distribution
plt.subplot(2, 2, 1)
match_type_counts = matches_df['Match Type'].value_counts()
plt.pie(match_type_counts, labels=match_type_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Match Types Distribution')

# 2. Confidence distribution histogram
plt.subplot(2, 2, 2)
sns.histplot(matches_df['Confidence'], bins=10, kde=True)
plt.title('Confidence Score Distribution')
plt.xlabel('Confidence Score')
plt.ylabel('Count')

# 3. Match types by confidence boxplot
plt.subplot(2, 2, 3)
sns.boxplot(x='Match Type', y='Confidence', data=matches_df)
plt.title('Confidence by Match Type')
plt.xticks(rotation=45)

# 4. Top entities bar chart
plt.subplot(2, 2, 4)
top_entities = matches_df['Watched Entity'].value_counts().head(10)
sns.barplot(x=top_entities.values, y=top_entities.index)
plt.title('Top 10 Matched Entities')
plt.xlabel('Count')

plt.tight_layout()
plt.show()
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 2. Entity Network

Next, let's create a network graph of entities and their matches.
"""))

        self.notebook.cells.append(new_code_cell("""
# Create a network graph of entities and their matches
try:
    import networkx as nx
    from pyvis.network import Network
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes for watched entities
    for entity in watch_list.get_all_entities():
        G.add_node(entity.name, type='watched', size=20, color='#6baed6')  # Blue for watched entities
    
    # Add nodes for extracted entities and edges for matches
    for result in matching_results:
        for match in result.matches_found:
            extracted_entity = match.extracted_entity.name
            watched_entity = match.watched_entity.name
            
            # Add node if it doesn't exist
            if not G.has_node(extracted_entity):
                G.add_node(extracted_entity, type='extracted', size=10, color='#fd8d3c')  # Orange for extracted entities
            
            # Add edge with confidence as weight
            G.add_edge(extracted_entity, watched_entity, weight=match.confidence, title=f"Confidence: {match.confidence:.2f}")
    
    # Create a pyvis network
    net = Network(notebook=True, height='500px', width='100%')
    
    # Add nodes and edges from networkx graph
    for node, attrs in G.nodes(data=True):
        net.add_node(node, title=node, size=attrs['size'], color=attrs['color'])
    
    for u, v, attrs in G.edges(data=True):
        width = attrs['weight'] * 5  # Scale width by confidence
        net.add_edge(u, v, width=width, title=attrs['title'])
    
    # Set physics layout
    net.barnes_hut(spring_length=200)
    
    # Display the graph
    net.show('entity_network.html')
    
    print("Network graph created successfully! Open 'entity_network.html' to view it.")
    print(f"Graph contains {len(G.nodes())} nodes and {len(G.edges())} edges.")
    
    # Show a preview of the graph structure
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=[G.nodes[n]['color'] for n in G.nodes], 
            node_size=[G.nodes[n]['size']*10 for n in G.nodes], font_size=8, width=[G[u][v]['weight']*2 for u, v in G.edges])
    plt.title('Entity Network Preview')
    plt.show()
    
except ImportError:
    print("To create network graphs, install networkx and pyvis:")
    print("pip install networkx pyvis")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### 3. Match Timeline

Finally, let's create a timeline of matches across articles.
"""))

        self.notebook.cells.append(new_code_cell("""
# Create a timeline of matches across articles

# Add article index to match data
for i, result in enumerate(matching_results):
    for match in result.matches_found:
        # Find the corresponding row in the DataFrame
        mask = (matches_df['Article'] == result.article_title) & \
               (matches_df['Extracted Entity'] == match.extracted_entity.name) & \
               (matches_df['Watched Entity'] == match.watched_entity.name)
        
        # Add article index
        matches_df.loc[mask, 'Article Index'] = i

# Convert to numeric
matches_df['Article Index'] = pd.to_numeric(matches_df['Article Index'])

# Create the timeline plot
plt.figure(figsize=(15, 8))

# Plot matches by article index
sns.scatterplot(x='Article Index', y='Confidence', hue='Match Type', size='Confidence',
                data=matches_df, palette='viridis', sizes=(20, 200))

# Add article titles as x-tick labels
plt.xticks(range(len(matching_results)), 
           [result.article_title[:20] + '...' if len(result.article_title) > 20 else result.article_title 
            for result in matching_results], 
           rotation=45, ha='right')

plt.title('Match Timeline Across Articles')
plt.xlabel('Article')
plt.ylabel('Confidence Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
"""))

    def _add_conclusion(self):
        """Add conclusion to the notebook."""
        self.notebook.cells.append(new_markdown_cell("""
## Conclusion

In this tutorial, you've learned how to use the entity resolution system to:

1. **Prepare Entities**: Create and enrich a watch list of entities
2. **Process Articles**: Extract entities from articles using NER
3. **Match Entities**: Match extracted entities against your watch list
4. **Visualize Results**: Create insightful visualizations of the matching results

The entity resolution system demonstrates several key capabilities:

- **Hybrid Matching**: Combining exact, alias, semantic, and fuzzy matching
- **LLM-Powered Explanations**: Using LLMs to explain and validate matches
- **Multilingual Support**: Working with entities and articles in multiple languages
- **Scalable Architecture**: Built on Elasticsearch for scalability and performance

### Next Steps

To further explore the entity resolution system, you can:

1. **Add More Entities**: Expand your watch list with more entities
2. **Process More Articles**: Test with a larger corpus of articles
3. **Tune Parameters**: Adjust matching thresholds and configurations
4. **Integrate with Applications**: Use the API to integrate with your applications

### Resources

- **Documentation**: [Entity Resolution Demo Package Documentation](https://github.com/yourusername/entity_resolution_demo_package)
- **Source Code**: [GitHub Repository](https://github.com/yourusername/entity_resolution_demo_package)
- **Issues and Feedback**: [GitHub Issues](https://github.com/yourusername/entity_resolution_demo_package/issues)

Thank you for exploring the entity resolution system!
"""))


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a tutorial notebook for the Entity Resolution system")
    parser.add_argument("--output", "-o", help="Output path for the notebook")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite if file exists")
    args = parser.parse_args()
    
    # Generate and save the notebook
    generator = TutorialNotebookGenerator(output_path=args.output)
    generator.generate()
    generator.save(force=args.force)