#!/usr/bin/env python3
"""
Notebook Generator for Entity Resolution Tutorial

This module generates a Jupyter notebook that demonstrates the entity resolution system.
It creates a comprehensive tutorial with explanations and code examples.
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
        self._add_installation_section()
        self._add_system_overview()
        self._add_entity_preparation_section()
        self._add_article_processing_section()
        self._add_entity_matching_section()
        self._add_visualization_section()
        self._add_interactive_section()
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

Entity resolution is the process of identifying and linking mentions of the same entity across different data sources. For example, recognizing that "Joe Biden", "President Biden", and "POTUS" all refer to the same person.

## What You'll Learn

In this tutorial, you will:

1. Set up the entity resolution system
2. Prepare a watch list of entities with rich contextual information
3. Process articles to extract entities using hybrid NER
4. Match extracted entities against your watch list
5. Understand the LLM-powered explanations for entity matches
6. Visualize the results

Let's get started!
"""))

    def _add_installation_section(self):
        """Add installation and setup instructions."""
        self.notebook.cells.append(new_markdown_cell("""
## Installation and Setup

First, let's make sure the entity_resolution_demo package is installed and configured correctly.
"""))

        self.notebook.cells.append(new_code_cell("""
# Check if the package is installed
try:
    import entity_resolution_demo
    print(f"✅ Successfully imported entity_resolution_demo package")
except ImportError:
    print("❌ Package not found. Installing...")
    !pip install -e .
    import entity_resolution_demo
    print(f"✅ Successfully installed and imported entity_resolution_demo package")
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
import os
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

    def _add_system_overview(self):
        """Add system overview section."""
        self.notebook.cells.append(new_markdown_cell("""
## System Overview

The entity resolution system consists of three main components:

1. **Entity Preparation**: Creates and enriches a watch list of entities with contextual information
2. **Article Processing**: Extracts entities from articles using hybrid NER (Named Entity Recognition)
3. **Entity Matching**: Matches extracted entities against the watch list using multiple strategies

Let's visualize this pipeline:
"""))

        self.notebook.cells.append(new_code_cell("""
try:
    from IPython.display import Image, display
    from graphviz import Digraph
    import os
    
    # Create a simple pipeline diagram
    dot = Digraph(comment='Entity Resolution Pipeline')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Add nodes
    dot.node('A', 'Entity Preparation\\n(Watch List)')
    dot.node('B', 'Article Processing\\n(Entity Extraction)')
    dot.node('C', 'Entity Matching\\n(Resolution)')
    dot.node('D', 'LLM Explanations')
    
    # Add edges
    dot.edge('A', 'C')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    
    # Render the diagram
    dot.render('entity_resolution_pipeline', format='png', cleanup=True)
    display(Image(filename='entity_resolution_pipeline.png'))
    os.remove('entity_resolution_pipeline.png')  # Clean up
    print("✅ Pipeline visualization created successfully")
    
except ModuleNotFoundError as e:
    print(f"❌ Error: {e}")
    print("To visualize the pipeline, please install the required packages:")
    print("")
    print("    pip install graphviz")
    print("")
    print("Note: This requires the Graphviz software to be installed on your system as well.")
    print("- For macOS: brew install graphviz")
    print("- For Ubuntu/Debian: sudo apt-get install graphviz")
    print("- For Windows: Download from https://graphviz.org/download/")
    print("")
    print("Alternatively, here's a text representation of the pipeline:")
    print("")
    print("Entity Preparation (Watch List) ------+")
    print("                                      |---> Entity Matching ---> LLM Explanations")
    print("Article Processing (Entity Extraction) +")
    
    # Create a simple ASCII diagram as fallback
    from IPython.display import HTML
    display(HTML(
        '<div style="font-family: monospace; white-space: pre; line-height: 1.5">'
        'Entity Preparation (Watch List) ------+<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--&gt; Entity Matching --&gt; LLM Explanations<br>'
        'Article Processing (Entity Extraction) +<br>'
        '</div>'
    ))
except Exception as e:
    print(f"❌ Error creating visualization: {e}")
    
    # Provide more specific guidance for common Graphviz errors
    if "failed to execute" in str(e) and "dot" in str(e):
        print("This error indicates that the Graphviz executable is not installed or not in your PATH.")
        print("The Python package 'graphviz' is just a wrapper and requires the actual Graphviz software.")
        print("")
        print("To install Graphviz:")
        print("- For macOS: brew install graphviz")
        print("- For Ubuntu/Debian: sudo apt-get install graphviz")
        print("- For Windows: Download from https://graphviz.org/download/ and add to PATH")
        print("")
        print("After installation, restart your Jupyter notebook or Python kernel.")
    
    print("")
    print("Continuing with the tutorial without visualization...")

"""))

    def _add_entity_preparation_section(self):
        """Add entity preparation section."""
        self.notebook.cells.append(new_markdown_cell("""
## Entity Preparation

The first step in the entity resolution process is to prepare a watch list of entities that you want to monitor. These entities are enriched with contextual information to improve matching accuracy.

Let's create a simple watch list with a few entities:
"""))

        self.notebook.cells.append(new_code_cell("""
from entity_resolution_demo.entity_preparation.entity_watch_list import EntityWatchList
from entity_resolution_demo.entity_preparation.entity_enricher import EntityEnricher
from entity_resolution_demo.entity_preparation.entity_indexer import EntityIndexer
from entity_resolution_demo.search.elastic_client import ElasticClient
from entity_resolution_demo.pipeline_runner.config import load_config

# Load configuration
config = load_config()

# Create Elasticsearch client
try:
    elastic_client = ElasticClient(config)
    print("✅ Successfully connected to Elasticsearch")
    
    # Test connection
    if hasattr(elastic_client, 'check_connection') and elastic_client.check_connection():
        print("✅ Elasticsearch connection verified")
    else:
        print("⚠️ Could not verify Elasticsearch connection")
        print("")
        print("❌ This tutorial requires a working Elasticsearch connection to proceed.")
        print("Please ensure that:")
        print("1. You have set the ELASTIC_CLOUD_ID and ELASTIC_API_KEY environment variables")
        print("2. Your Elasticsearch cluster is running and accessible")
        print("3. You have the necessary permissions to create indices and index documents")
        print("")
        print("Once you have fixed these issues, please restart the notebook.")
        import sys
        sys.exit("Exiting tutorial due to Elasticsearch connection issues.")
except Exception as e:
    print(f"❌ Error connecting to Elasticsearch: {e}")
    print("")
    print("❌ This tutorial requires a working Elasticsearch connection to proceed.")
    print("Please ensure that:")
    print("1. You have set the ELASTIC_CLOUD_ID and ELASTIC_API_KEY environment variables")
    print("2. Your Elasticsearch cluster is running and accessible")
    print("3. You have the necessary permissions to create indices and index documents")
    print("")
    print("Once you have fixed these issues, please restart the notebook.")
    import sys
    sys.exit("Exiting tutorial due to Elasticsearch connection issues.")

# Create entity watch list
watch_list = EntityWatchList()

# Add some entities to the watch list
entities = [
    {"name": "Joe Biden", "entity_type": "PERSON", "description": "President of the United States", "priority": "high"},
    {"name": "Elon Musk", "entity_type": "PERSON", "description": "CEO of Tesla and SpaceX", "priority": "high"},
    {"name": "Taylor Swift", "entity_type": "PERSON", "description": "American singer-songwriter", "priority": "medium"},
    {"name": "Microsoft", "entity_type": "ORGANIZATION", "description": "Technology company", "priority": "high"}
]

for entity in entities:
    watch_list.add_entity(
        name=entity["name"],
        entity_type=entity["entity_type"],
        description=entity["description"],
        priority=entity["priority"]
    )

print(f"Created watch list with {len(watch_list.get_all_entities())} entities")

# Display the entities
for entity in watch_list.get_all_entities():
    print(f"- {entity.name} ({entity.entity_type}): {entity.description}")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### Entity Enrichment

Now, let's enrich these entities with contextual information from Wikipedia. This will help improve matching accuracy, especially for semantic and contextual matches.
"""))

        self.notebook.cells.append(new_code_cell("""
# Create entity enricher
try:
    enricher = EntityEnricher(config)
    print("✅ Successfully created EntityEnricher")
except Exception as e:
    print(f"❌ Error creating EntityEnricher: {e}")
    print("Using fallback enrichment method with static descriptions")
    # Define a fallback enricher function
    class FallbackEnricher:
        def enrich_entity(self, name, description):
            from entity_resolution_demo.entity_preparation.entity_enricher import EnrichedEntity
            context = f"{description}. This entity is important for entity resolution demonstrations."
            return EnrichedEntity(
                name=name,
                entity_context=context,
                confidence_score=0.7,
                enrichment_source="Fallback Enricher",
                alternative_contexts=[]
            )
    enricher = FallbackEnricher()

# Enrich entities
enriched_entities = []
for entity in watch_list.get_all_entities():
    print(f"Enriching entity: {entity.name}")
    try:
        # Enrich entity with Wikipedia context
        enriched_entity = enricher.enrich_entity(entity.name, entity.description)
        
        # Generate a consistent ID
        base_name = entity.name.lower().replace(' ', '_').replace('-', '_')
        enriched_entity.id = f"{base_name}_000"
        
        # Add to the list of enriched entities
        enriched_entities.append(enriched_entity)
        
        # Print a sample of the context
        context_preview = enriched_entity.entity_context[:100] + "..." if len(enriched_entity.entity_context) > 100 else enriched_entity.entity_context
        print(f"  Context: {context_preview}")
        
    except Exception as e:
        print(f"  ❌ Error enriching entity {entity.name}: {e}")
        print(f"  Creating basic entity with just the description")
        
        # Create a basic entity with just the description
        from entity_resolution_demo.entity_preparation.entity_enricher import EnrichedEntity
        fallback_entity = EnrichedEntity(
            name=entity.name,
            entity_context=entity.description or f"This is {entity.name}, a {entity.entity_type.lower()}.",
            confidence_score=0.5,
            enrichment_source="Fallback",
            alternative_contexts=[]
        )
        fallback_entity.id = entity.name.lower().replace(' ', '_').replace('-', '_') + "_000"
        enriched_entities.append(fallback_entity)
        print(f"  ✅ Created fallback entity for {entity.name}")

if enriched_entities:
    print(f"✅ Enriched {len(enriched_entities)} entities")
else:
    print("❌ No entities were enriched. Please check your configuration and try again.")
"""))

        self.notebook.cells.append(new_markdown_cell("""
### Entity Indexing

Now, let's index these enriched entities in Elasticsearch so we can search for them later.
"""))

        self.notebook.cells.append(new_code_cell("""
# Create entity indexer
from entity_resolution_demo.entity_preparation.entity_indexer import EntityIndexer

# Initialize the entity indexer
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

"""))

    def _add_article_processing_section(self):
        """Add article processing section."""
        self.notebook.cells.append(new_markdown_cell("""
## Article Processing

The second step in the entity resolution process is to process articles and extract entities from them. We'll use the HybridNERExtractor, which combines Elasticsearch NER with pattern-based extraction for better coverage.

Let's create a sample article and process it:
"""))

        self.notebook.cells.append(new_code_cell("""
from entity_resolution_demo.article_processing.article_processor import ArticleProcessor, Article
from entity_resolution_demo.article_processing.hybrid_ner_extractor import HybridNERExtractor

# Create a sample article
sample_article = Article(
    id="article1",
    title="Tech Leaders Meet at Global Summit",
    content="Tech industry leaders gathered at the Global Technology Summit in San Francisco yesterday. "
           "Microsoft CEO Satya Nadella discussed the company's AI initiatives, while Tesla CEO Elon Musk "
           "presented his vision for sustainable energy. The event was also attended by Apple representatives "
           "and several government officials including US President Joe Biden, who emphasized the importance "
           "of technological innovation for economic growth. "
           "Taylor Swift made a surprise appearance at the gala dinner, performing her latest hits for the attendees.",
    source="Tech News Daily",
    language="en"
)

# Create article processor with HybridNERExtractor
processor = ArticleProcessor(config, elastic_client)

# Check if we're using HybridNERExtractor
from entity_resolution_demo.article_processing.hybrid_ner_extractor import HybridNERExtractor
if isinstance(processor.name_extractor, HybridNERExtractor):
    print("✅ Using HybridNERExtractor for better entity extraction")
else:
    print("⚠️ Using basic NER extractor - entity extraction may be limited")

# Process the article
processed_article = processor.process_article(sample_article)
print(f"✅ Successfully processed article: {sample_article.title}")


# Print extraction results
print(f"Processed article: {sample_article.id} - '{sample_article.title}'")
print(f"Found {processed_article.total_entities_found} entities in {processed_article.processing_time:.2f}s")

# Print extracted entities
if processed_article.extracted_entities:
    print("Extracted entities:")
    for entity in processed_article.extracted_entities:
        entity_type_emoji = "👤" if entity.entity_type == "PERSON" else "🏢" if entity.entity_type == "ORGANIZATION" else "📍" if entity.entity_type == "LOCATION" else "📌"
        print(f"  {entity_type_emoji} {entity.name} ({entity.entity_type}) - Confidence: {entity.confidence:.2f}")
        print(f"     Context: ...{entity.context[:50]}..." if len(entity.context) > 50 else f"     Context: {entity.context}")
else:
    print("⚠️ No entities were extracted from the article.")
"""))

    def _add_entity_matching_section(self):
        """Add entity matching section."""
        self.notebook.cells.append(new_markdown_cell("""
## Entity Matching

The third step in the entity resolution process is to match the extracted entities against our watch list. This is where the real power of the system comes into play, with multiple matching strategies:

1. **Exact Matching**: Direct string comparison
2. **Lexical Matching**: BM25 text similarity
3. **Semantic Matching**: Vector embeddings similarity
4. **Hybrid Matching**: Combination of lexical and semantic approaches
5. **Role-Based Matching**: Matching based on roles/titles (e.g., "US President" → "Joe Biden")

Let's match the entities we extracted from our sample article:
from entity_resolution_demo.entity_matching.real_time_entity_matcher import RealTimeEntityMatcher

# Create entity matcher
matcher = RealTimeEntityMatcher(
    watch_list=watch_list,
    elastic_client=elastic_client,
    config=config
)

# Match entities
result = matcher.match_article(processed_article)

# Print matches
if result.matches_found:
    print(f"Found {len(result.matches_found)} matches:")
    for i, match in enumerate(result.matches_found):
        print(f"Match {i+1}: {match.extracted_entity.name} → {match.watched_entity.name}")
        print(f"  Confidence: {match.confidence:.2f}")
        print(f"  Match Type: {match.match_type}")
        if hasattr(match, 'reasoning') and match.reasoning:
            explanation = match.reasoning[:150] + "..." if len(match.reasoning) > 150 else match.reasoning
            print(f"  Explanation: {explanation}")
        print()
else:
    print("No matches found")
One of the most powerful features of our entity resolution system is the ability to generate detailed explanations for matches using LLMs. Let's enhance our matches with LLM explanations:
"""))

        self.notebook.cells.append(new_code_cell("""
from entity_resolution_demo.entity_matching.enhanced_batch_match_judge import EnhancedBatchMatchJudge

# First, let's make sure we have the result from the previous section
try:
    # Check if result is defined and has matches_found attribute
    if not hasattr(result, 'matches_found'):
        print("⚠️ No matching results found from previous section.")
        result.matches_found = []
except NameError:
    # If result is not defined, create a minimal result object
    from entity_resolution_demo.entity_matching.entity_match import MatchingResult
    result = MatchingResult(
        article_id="demo_article",
        article_title="Demo Article",
        total_entities_extracted=0
    )
    # result.matches_found is already an empty list by default
    print("⚠️ No matching results available. Creating empty result object.")

# Check if OpenAI API key is set
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️ OPENAI_API_KEY is not set. Using mock explanations for demonstration.")
    
    # Create simple mock explanations
    if result.matches_found:
        print("")
        print("Mock LLM Explanations:")
        for i, match in enumerate(result.matches_found):
            print(f"Match: {match.extracted_entity.name} → {match.watched_entity.name}")
            print(f"Confidence: {match.confidence:.2f}")
            print(f"Match Type: {match.match_type}")
            print(f"Reasoning: This appears to be a valid match based on name similarity.")
    else:
        print("No matches to explain")
else:
    # Create batch match judge with LLM integration
    batch_judge = EnhancedBatchMatchJudge(
        config=config,
        batch_size=5,
        es_client=elastic_client.es
    )

    # Prepare batch for processing
    batch = []
    for i, match in enumerate(result.matches_found):
        batch.append({
            'pair_index': i,
            'query_name': match.extracted_entity.name,
            'candidate_name': match.watched_entity.name,
            'context': match.extracted_entity.context or ""
        })

    # Process batch with LLM
    if batch:
        processed_batch = batch_judge.batch_judge_matches(batch)
        
        # Print enhanced explanations
        print("")
        print("Enhanced LLM Explanations:")
        for i, processed_match in enumerate(processed_batch):
            print(f"Match: {processed_match.extracted_entity.name} → {processed_match.watched_entity.name}")
            print(f"Confidence: {processed_match.confidence:.2f}")
            print(f"Match Type: {processed_match.match_type}")
            
            if hasattr(processed_match, 'reasoning') and processed_match.reasoning:
                print(f"Reasoning: {processed_match.reasoning}")
    else:
        print("No matches to process with LLM")
"""))

    def _add_visualization_section(self):
        """Add visualization section."""
        self.notebook.cells.append(new_markdown_cell("""
## Visualizing Entity Matches

Let's visualize the entity matches we found to better understand the relationships:
"""))

        self.notebook.cells.append(new_code_cell("""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a DataFrame from the matches
if result.matches_found:
    match_data = []
    for match in result.matches_found:
        match_data.append({
            'Extracted Entity': match.extracted_entity.name,
            'Watched Entity': match.watched_entity.name,
            'Confidence': match.confidence,
            'Match Type': match.match_type
        })
    
    df = pd.DataFrame(match_data)
    
    # Create a bar chart of confidence scores
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Extracted Entity'] + ' → ' + df['Watched Entity'], df['Confidence'], color='skyblue')
    
    # Add confidence values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Entity Match')
    plt.ylabel('Confidence Score')
    plt.title('Entity Match Confidence Scores')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate the text
    plt.tight_layout()
    plt.show()
    
    # Create a table of match details
    from IPython.display import display
    display(df)
else:
    print("No matches to visualize")
"""))

    def _add_interactive_section(self):
        """Add interactive section."""
        self.notebook.cells.append(new_markdown_cell("""
## Try It Yourself

Now it's your turn! Use the form below to enter your own article and see what entities are extracted and matched:
"""))

        self.notebook.cells.append(new_code_cell("""
import ipywidgets as widgets
from IPython.display import display, clear_output

# Create text area for article input
article_text = widgets.Textarea(
    value='',
    placeholder='Enter article text here...',
    description='Article:',
    disabled=False,
    layout=widgets.Layout(width='100%', height='200px')
)

# Create text field for article title
article_title = widgets.Text(
    value='',
    placeholder='Enter article title here...',
    description='Title:',
    disabled=False,
    layout=widgets.Layout(width='50%')
)

# Create button to process article
process_button = widgets.Button(
    description='Process Article',
    disabled=False,
    button_style='primary',
    tooltip='Click to process the article',
    icon='check'
)

# Create output area for results
output = widgets.Output()

# Define button click handler
def on_button_clicked(b):
    with output:
        clear_output()
        
        if not article_text.value:
            print("Please enter some article text.")
            return
            
        title = article_title.value or "Custom Article"
        
        print(f"Processing article: {title}")
        print("=" * 50)
        
        # Create article
        custom_article = Article(
            id="custom_article",
            title=title,
            content=article_text.value,
            source="Custom Input",
            language="en"
        )
        
        # Process article
        processed = processor.process_article(custom_article)
        
        # Print extracted entities
        print(f"Found {processed.total_entities_found} entities:")
        for entity in processed.extracted_entities:
            entity_type_emoji = "👤" if entity.entity_type == "PERSON" else "🏢" if entity.entity_type == "ORGANIZATION" else "📍" if entity.entity_type == "LOCATION" else "📌"
            print(f"  {entity_type_emoji} {entity.name} ({entity.entity_type}) - Confidence: {entity.confidence:.2f}")
        
        print("\\nMatching entities against watch list...")
        # Match entities
        match_result = matcher.match_article(processed)
        
        # Print matches
        if match_result.matches_found:
            print(f"Found {len(match_result.matches_found)} matches:")
            for i, match in enumerate(match_result.matches_found):
                print(f"Match {i+1}: {match.extracted_entity.name} → {match.watched_entity.name}")
                print(f"  Confidence: {match.confidence:.2f}")
                print(f"  Match Type: {match.match_type}")
        else:
            print("No matches found in watch list")

# Connect button to handler
process_button.on_click(on_button_clicked)

# Display widgets
display(article_title)
display(article_text)
display(process_button)
display(output)
"""))

    def _add_conclusion(self):
        """Add conclusion section."""
        self.notebook.cells.append(new_markdown_cell("""
## Conclusion

In this tutorial, you've learned how to use the entity resolution system to:

1. Create and enrich a watch list of entities
2. Process articles to extract entities using hybrid NER
3. Match extracted entities against your watch list
4. Generate LLM-powered explanations for matches
5. Visualize the results

This system can be used in a variety of applications, including:

- News monitoring and alerting
- Compliance and risk management
- Competitive intelligence
- Customer relationship management
- Research and analysis

### Next Steps

To learn more about the entity resolution system, check out the following resources:

- [GitHub Repository](https://github.com/your-username/entity_resolution_demo_package)
- [Documentation](https://your-username.github.io/entity_resolution_demo_package)
- [API Reference](https://your-username.github.io/entity_resolution_demo_package/api)

### Feedback

If you have any feedback or questions, please open an issue on the GitHub repository or contact the author directly.

Thank you for using the entity resolution system!
"""))


def generate_tutorial_notebook(output_path=None, force=False):
    """
    Generate the entity resolution tutorial notebook.
    
    Args:
        output_path: Path where the notebook will be saved. If None, saves to current directory.
        force: If True, overwrite existing file if it exists.
        
    Returns:
        str: Path to the generated notebook file.
        
    Raises:
        FileExistsError: If the output file exists and force is False.
    """
    generator = TutorialNotebookGenerator(output_path)
    generator.generate()
    return generator.save(force=force)


if __name__ == "__main__":
    # If run directly, generate the notebook in the current directory
    generate_tutorial_notebook()
