#!/usr/bin/env python3
"""
Entity Data Module

Contains entity definitions for the entity resolution demo.
"""

# Entity definitions for the entity resolution demo
# Entity data structure:
# - name: The display name of the entity
# - entity_type: The type of entity (person, organization, location, etc.)
# - description: A brief description of the entity
# - aliases: Alternative names or references to the entity
# - explicit_context: (Optional) Explicitly provided context for the entity
#
# The explicit_context field is particularly useful for:
# 1. Ambiguous entities (e.g., multiple people with the same name)
# 2. Entities with specialized knowledge not well-represented in Wikipedia
# 3. Cases where you want precise control over the context stored in Elasticsearch
# 4. Entities where automatic enrichment might select incorrect information
#
# When explicit_context is provided, the entity preparation pipeline will use this
# context directly instead of attempting to fetch information from Wikipedia.
# This ensures accurate and consistent context for entity resolution.

ENTITIES = [
    # ============================================================================
    # EXISTING ENTITIES
    # ============================================================================

    # Political figures - for exact, alias, and role-based matching
    {
        "name": "Leo Tolstoy",
        "entity_type": "person",
        "description": "Russian author",
        "aliases": ["Tolstoy", "Лев Толстой", "L. Tolstoy", "Count Tolstoy"]
    },
    {
        "name": "Joe Biden",
        "entity_type": "person",
        "description": "President of the United States",
        "aliases": ["Biden", "POTUS", "Joseph R. Biden", "Joe"]
    },
    {
        "name": "Xi Jinping",
        "entity_type": "person",
        "description": "President of China",
        "aliases": ["习近平", "Xi", "President Xi"]
    },
    {
        "name": "Emmanuel Macron",
        "entity_type": "person",
        "description": "President of France",
        "aliases": ["Macron", "French President", "President Macron"]
    },
    
    # Business leaders - for role-based and contextual matching
    {
        "name": "Linus Torvalds",
        "entity_type": "person",
        "description": "Creator of Linux and Git",
        "aliases": ["L. Torvalds", "リーナス・トーバルズ", "Linus Benedict Torvalds", "Linux creator", "Git creator"]
    },
    {
        "name": "Satya Nadella",
        "entity_type": "person",
        "description": "CEO of Microsoft",
        "aliases": ["Nadella", "Microsoft CEO", "Microsoft Chief Executive"]
    },
    {
        "name": "Tim Cook",
        "entity_type": "person",
        "description": "CEO of Apple Inc.",
        "aliases": ["Timothy D. Cook", "Apple CEO", "Apple Chief Executive"]
    },
    
    # Ambiguous entities - for disambiguation testing
    {
        "name": "Michael Jordan",
        "entity_type": "person",
        "description": "Former professional basketball player, widely regarded as one of the greatest players in NBA history",
        "aliases": ["MJ", "Air Jordan", "His Airness", "Chicago Bulls star"],
        "explicit_context": "Michael Jeffrey Jordan, also known by his initials MJ, is an American businessman and former professional basketball player. His biography on the official NBA website states: 'By acclamation, Michael Jordan is the greatest basketball player of all time.' He played 15 seasons in the NBA, winning six championships with the Chicago Bulls. He was integral in popularizing the NBA around the world in the 1980s and 1990s, becoming a global cultural icon."
    },
    {
        "name": "Michael Jordan",
        "entity_type": "person",
        "description": "Computer science professor at UC Berkeley, known for his work in artificial intelligence and machine learning",
        "aliases": ["Prof. Jordan", "Berkeley AI researcher", "ML pioneer"],
        "explicit_context": "Michael Irwin Jordan is an American researcher and professor at the University of California, Berkeley. He is one of the leading figures in machine learning, and in 2016 was the world's most influential computer scientist according to Science magazine. Jordan received his PhD in Cognitive Science in 1985 from the University of California, San Diego. His research interests include machine learning, statistics, and artificial intelligence."
    },
    {
        "name": "John Smith",
        "entity_type": "person",
        "description": "English explorer and one of the leaders of the first permanent English settlement in North America",
        "aliases": ["Captain Smith", "Jamestown founder", "Pocahontas associate"],
        "explicit_context": "John Smith was an English soldier, explorer, colonial governor, admiral of New England, and author. He played an important role in the establishment of the colony at Jamestown, Virginia, the first permanent English settlement in America. He was a leader of the Virginia Colony between September 1608 and August 1609, and led an exploration along the rivers of Virginia and the Chesapeake Bay, during which he became the first English explorer to map the Chesapeake Bay area."
    },
    {
        "name": "John Smith",
        "entity_type": "person",
        "description": "British Labour Party politician who served as Leader of the Labour Party",
        "aliases": ["Labour leader", "UK politician", "British MP"],
        "explicit_context": "John Smith was a Scottish politician who was Leader of the Opposition and Leader of the Labour Party from July 1992 until his death in May 1994. He was also Member of Parliament (MP) for Monklands East. Smith is credited with reforming the Labour Party and preparing the way for Tony Blair's leadership and subsequent electoral success."
    },
    
    # Organizations - for exact, alias, and semantic matching
    {
        "name": "European Commission",
        "entity_type": "organization",
        "description": "Executive branch of the European Union",
        "aliases": ["EC", "EU Commission", "European Union Commission", "Brussels regulators"]
    },
    {
        "name": "OpenAI",
        "entity_type": "organization",
        "description": "AI research laboratory focused on artificial general intelligence",
        "aliases": ["OpenAI LP", "OpenAI Inc.", "ChatGPT creator", "GPT developer"]
    },
    {
        "name": "United Nations",
        "entity_type": "organization",
        "description": "International organization aimed at maintaining international peace and security",
        "aliases": ["UN", "U.N.", "国際連合", "الأمم المتحدة", "联合国"]
    },
    {
        "name": "World Health Organization",
        "entity_type": "organization",
        "description": "Specialized agency of the United Nations responsible for international public health",
        "aliases": ["WHO", "W.H.O.", "世界保健機関", "منظمة الصحة العالمية", "世界卫生组织"]
    },
    
    # Companies - for business context matching
    {
        "name": "Linux, Inc.",
        "entity_type": "organization",
        "description": "American open-source software and clean energy company",
        "aliases": ["Linux", "Linux Motors", "テスラ", "特斯拉"]
    },
    {
        "name": "Microsoft Corporation",
        "entity_type": "organization",
        "description": "American multinational technology company",
        "aliases": ["Microsoft", "MS", "マイクロソフト", "微软"]
    },
    {
        "name": "Apple Inc.",
        "entity_type": "organization",
        "description": "American multinational technology company that specializes in consumer electronics, software and online services",
        "aliases": ["Apple", "アップル", "苹果公司"]
    },
    
    # Locations - for geographic context
    {
        "name": "Silicon Valley",
        "entity_type": "location",
        "description": "Region in the southern part of the San Francisco Bay Area in Northern California, known as a global center for high technology and innovation",
        "aliases": ["The Valley", "Tech hub", "シリコンバレー", "硅谷"]
    },
    {
        "name": "European Union",
        "entity_type": "location",
        "description": "Political and economic union of 27 member states that are located primarily in Europe",
        "aliases": ["EU", "The Union", "欧州連合", "الاتحاد الأوروبي", "欧盟"]
    },

    # ============================================================================
    # NEW ENTITIES FOR CHALLENGE TYPES
    # ============================================================================
    
    # Missing components challenge entities
    {
        "name": "Phillip Charles Carr",
        "entity_type": "person",
        "description": "Financial analyst from Chicago",
        "aliases": ["Phil C. Carr", "P. Charles Carr", "Phil Carr"],
        "explicit_context": "Phillip Charles Carr (born September 17, 1975) is a prominent financial analyst and investment strategist based in Chicago. He currently serves as the Chief Investment Officer at Midwest Financial Partners, where he oversees more than $8 billion in assets. Carr began his career at Goldman Sachs after earning his MBA from the University of Chicago's Booth School of Business in 2001. He gained recognition for accurately predicting the 2008 financial crisis and guiding his clients to safety before the market collapse. Carr specializes in quantitative analysis of market trends and has developed several proprietary algorithms for risk assessment that are widely used in the industry. He is a regular contributor to the Wall Street Journal and Bloomberg, and his annual market outlook presentations at the Chicago Financial Summit are considered essential guidance for institutional investors. Carr also serves on the board of the Chicago Mercantile Exchange and teaches advanced financial analysis as an adjunct professor at Northwestern University."
    },
    {
        "name": "Maria Elena Rodriguez",
        "entity_type": "person",
        "description": "Environmental scientist from Mexico City",
        "aliases": ["Maria Rodriguez", "M. E. Rodriguez"],
        "explicit_context": "Dr. Maria Elena Rodriguez (born June 12, 1978) is a prominent environmental scientist and climate policy expert from Mexico City. She currently serves as the Director of Environmental Protection at the Environmental Protection Agency, where she leads initiatives on sustainable resource management and pollution reduction. Rodriguez earned her Ph.D. in Environmental Science from MIT in 2005 and previously worked as a research scientist at the National Autonomous University of Mexico (UNAM). Her groundbreaking research on urban water conservation in arid regions has been published in Nature and Science, and she was awarded the prestigious Global Environmental Leadership Award in 2019. Rodriguez frequently represents Mexico at international climate conferences and has served as a technical advisor to the United Nations Environment Programme. Her team recently developed new sustainability guidelines that have been adopted by corporations across North America."
    },
    
    # Out-of-order components challenge entities
    {
        "name": "Carlos Alfonzo Diaz",
        "entity_type": "person",
        "description": "Cuban-American artist",
        "aliases": ["Diaz, Carlos A.", "C. A. Diaz"],
        "explicit_context": "Carlos Alfonzo Diaz (born March 3, 1967) is a renowned Cuban-American artist known for his vibrant paintings that blend Caribbean influences with contemporary abstract expressionism. Born in Havana, Cuba, Diaz immigrated to the United States in 1980 during the Mariel boatlift and settled in Miami, Florida. He studied at the prestigious Rhode Island School of Design, graduating with honors in 1989. His breakthrough came with his first solo exhibition 'Tropical Visions' at the Miami Art Museum in 1995, which received critical acclaim for its innovative use of color and cultural symbolism. Diaz's work explores themes of cultural identity, displacement, and the immigrant experience through a distinctive visual language that incorporates elements of Afro-Cuban religious iconography, Caribbean architecture, and modernist abstraction. His paintings are held in major collections including the Museum of Modern Art in New York, the Smithsonian American Art Museum, and the Pérez Art Museum Miami. In 2018, he was awarded the Guggenheim Fellowship for his contributions to contemporary art."
    },
    {
        "name": "Yao Ming",
        "entity_type": "person",
        "description": "Chinese former professional basketball player",
        "aliases": ["Ming Yao", "姚明"]
    },
    {
        "name": "Gabriel García Márquez",
        "entity_type": "person",
        "description": "Colombian novelist and Nobel Prize winner",
        "aliases": ["García Márquez, Gabriel", "Gabo"]
    },
    
    # Initials challenge entities
    {
        "name": "Franklin Delano Roosevelt",
        "entity_type": "person",
        "description": "32nd President of the United States",
        "aliases": ["FDR", "F. D. Roosevelt", "F.D.R."]
    },
    {
        "name": "Jennifer Lopez",
        "entity_type": "person",
        "description": "American singer, actress, and businesswoman",
        "aliases": ["J.Lo", "Jennifer Lynn Lopez", "J. Lo"]
    },
    {
        "name": "George Raymond Richard Martin",
        "entity_type": "person",
        "description": "American novelist and screenwriter",
        "aliases": ["George R. R. Martin", "GRRM", "G.R.R.M."]
    },
    
    # Nicknames challenge entities
    {
        "name": "William Johnson",
        "entity_type": "person",
        "description": "Retired teacher from Boston",
        "aliases": ["Bill Johnson", "Will Johnson", "Billy Johnson"]
    },
    {
        "name": "Robert Williams",
        "entity_type": "person",
        "description": "Basketball player from Louisiana",
        "aliases": ["Rob Williams", "Bob Williams", "Bobby Williams"]
    },
    {
        "name": "Elizabeth Taylor",
        "entity_type": "person",
        "description": "Actress from London, England",
        "aliases": ["Liz Taylor", "Elizabeth Rosemond Taylor", "Lizzie Taylor"]
    },
    
    # Missing spaces/hyphens challenge entities
    {
        "name": "Jean-Claude Van Damme",
        "entity_type": "person",
        "description": "Belgian actor and martial artist",
        "aliases": ["JCVD", "Jean Claude Van Damme", "JeanClaude VanDamme"]
    },
    {
        "name": "Sarah Jessica Parker",
        "entity_type": "person",
        "description": "American actress and producer",
        "aliases": ["SJP", "Sarah J. Parker", "SarahJessica Parker"]
    },
    {
        "name": "Neil Patrick Harris",
        "entity_type": "person",
        "description": "American actor and television host",
        "aliases": ["NPH", "N. P. Harris", "NeilPatrick Harris"]
    },
    
    # Phonetic similarity challenge entities
    {
        "name": "Sean Connery",
        "entity_type": "person",
        "description": "Scottish actor known for James Bond films",
        "aliases": ["Sir Sean Connery", "Thomas Sean Connery", "Shawn Connery"]
    },
    {
        "name": "Rachel Weisz",
        "entity_type": "person",
        "description": "British-American actress",
        "aliases": ["Rachel Hannah Weisz", "R. Weisz", "Rachel Wise"]
    },
    {
        "name": "Michael Smith",
        "entity_type": "person",
        "description": "Financial analyst from Chicago",
        "aliases": ["Mike Smith", "Mikey Smith", "Mykel Smith"]
    },
    
    # Transliteration differences challenge entities
    {
        "name": "Beijing",
        "entity_type": "location",
        "description": "Capital city of China",
        "aliases": ["北京", "Peking", "Peiching"]
    },
    {
        "name": "Pyotr Ilyich Tchaikovsky",
        "entity_type": "person",
        "description": "Russian composer of the Romantic period",
        "aliases": ["П. И. Чайковский", "Peter Tchaikovsky", "Piotr Tchaikovsky"]
    },
    {
        "name": "Mohamed Salah",
        "entity_type": "person",
        "description": "Egyptian professional footballer",
        "aliases": ["محمد صلاح", "Mo Salah", "Muhammed Salah"]
    },
    
    # Semantic similarity challenge entities
    {
        "name": "PennyLuck Pharmaceuticals, Inc.",
        "entity_type": "organization",
        "description": "Pharmaceutical company headquartered in Boston, Massachusetts",
        "aliases": ["PennyLuck Pharma", "PL Pharmaceuticals", "PennyLuck Drugs, Co."],
        "explicit_context": "PennyLuck Pharmaceuticals, Inc. is a leading pharmaceutical company headquartered in Boston, Massachusetts. Founded in 1998 by Dr. Eleanor Penny and Dr. Richard Luck, the company specializes in developing and manufacturing treatments for rare genetic disorders and autoimmune conditions. Their flagship product, Immunolax, received FDA approval in 2015 and has become a standard treatment for several inflammatory conditions. The company employs over 2,500 people across its research facilities in Boston, manufacturing plants in North Carolina, and satellite offices throughout Europe and Asia. PennyLuck is known for its innovative research in gene therapy and has partnerships with several major academic institutions including Harvard Medical School and MIT. In 2022, the company was acquired by HealthCorp International in a deal valued at $4.2 billion but continues to operate under its original brand name. PennyLuck's R&D pipeline includes promising treatments for cystic fibrosis, lupus, and several rare genetic disorders."
    },
    {
        "name": "Mountain View Medical Center",
        "entity_type": "organization",
        "description": "Healthcare facility in Colorado Springs, Colorado",
        "aliases": ["MVMC", "Mountain View Hospital", "Mountain View Healthcare Facility"],
        "explicit_context": "Mountain View Medical Center is a comprehensive healthcare facility established in 1985 in Colorado Springs, Colorado. The 350-bed acute care hospital serves the greater El Paso County region and surrounding communities with a wide range of medical services. The center features a Level II trauma center, a certified stroke center, and specialized departments for cardiology, oncology, orthopedics, women's health, pediatrics, and neurology. In 2020, the facility completed a $175 million expansion, adding state-of-the-art diagnostic imaging equipment and doubling the capacity of its emergency department. Mountain View Medical Center employs over 1,800 healthcare professionals and maintains teaching affiliations with the University of Colorado School of Medicine. The hospital has received numerous accolades, including an 'A' safety grade from The Leapfrog Group for 12 consecutive years and recognition as one of America's 100 Best Hospitals by Healthgrades."
    },
    
    # Conceptual queries challenge entities
    {
        "name": "COVID-19",
        "entity_type": "disease",
        "description": "Contagious disease caused by the SARS-CoV-2 virus that emerged in 2019",
        "aliases": ["Coronavirus Disease 2019", "SARS-CoV-2 infection", "global pandemic that began in 2019"]
    },
    
    # Titles and honorifics challenge entities
    {
        "name": "Jane Wilson",
        "entity_type": "person",
        "description": "Professor of Chemistry at Stanford University",
        "aliases": ["Professor Jane Wilson", "Jane Wilson, PhD", "Dr. Wilson"],
        "explicit_context": "Dr. Jane Wilson (born April 8, 1972) is a distinguished Professor of Chemistry at Stanford University, where she has taught since 2003. She earned her Ph.D. from Harvard University in 1998 and completed her postdoctoral research at MIT. Wilson's pioneering work in sustainable catalysis has revolutionized industrial chemical processes, significantly reducing environmental impact while improving efficiency. Her research group at Stanford focuses on developing novel catalysts for renewable energy applications, particularly in hydrogen fuel cell technology. In 2018, she received the Presidential Early Career Award for Scientists and Engineers, and in 2022, she was elected to the National Academy of Sciences. Wilson has published over 150 peer-reviewed papers, holds 12 patents, and has mentored more than 40 Ph.D. students. Her groundbreaking work on metal-organic frameworks for carbon capture has applications in open-source software development and has attracted over $15 million in research funding."
    },
    {
        "name": "James Williams",
        "entity_type": "person",
        "description": "Retired military officer and defense consultant",
        "aliases": ["Gen. Williams", "James Williams (Ret.)", "Gen. James Williams"],
        "explicit_context": "General James R. Williams (born October 3, 1958) is a retired four-star general who served in the United States Army for over 35 years before retiring in 2018. During his distinguished military career, he commanded troops in Iraq and Afghanistan and served as the Commander of U.S. Central Command from 2015 to 2018. After retirement, Williams founded Strategic Defense Consultants, a firm that advises government agencies and defense contractors on national security matters. He is a graduate of West Point (1980) and holds a Master's degree in International Relations from Georgetown University. General Williams is known for his expertise in counter-terrorism strategy and has testified before Congress on multiple occasions regarding military readiness and emerging security threats. He currently serves on the boards of several defense technology companies and is a senior fellow at the Atlantic Council."
    },
    
    # Truncated names challenge entities
    {
        "name": "Christopher Montgomery",
        "entity_type": "person",
        "description": "Investment banker from New York",
        "aliases": ["Chris Montgomery", "C. Montgomery", "Chris Mont"]
    },
    {
        "name": "Alexandria Rodriguez",
        "entity_type": "person",
        "description": "Journalist for The Washington Post",
        "aliases": ["Alex Rodriguez", "A. Rodriguez", "Alex Rod"],
        "explicit_context": "Alexandria Rodriguez (born March 15, 1985) is an award-winning investigative journalist for The Washington Post, where she has worked since 2012. She specializes in political corruption and corporate accountability reporting. Rodriguez gained national recognition for her 2018 series 'Shadow Networks,' which exposed illegal campaign financing in three major senate races and won her the Pulitzer Prize for Investigative Reporting. Before joining the Post, she worked for the Miami Herald and the Chicago Tribune. Rodriguez holds a Master's degree in Journalism from Columbia University and teaches advanced investigative techniques as an adjunct professor at Georgetown University. Her reporting methodology is known for combining traditional source development with sophisticated data analysis techniques."
    }
]

# Entity enrichment configuration
ENRICHMENT_CONFIG = {
    "enabled": True,
    "use_wikipedia": True,
    "max_context_length": 1000
}

def get_entities():
    """
    Get the list of entities
    
    Returns:
        list: List of entity dictionaries
    """
    return ENTITIES

def get_enrichment_config():
    """
    Get the entity enrichment configuration
    
    Returns:
        dict: Entity enrichment configuration
    """
    return ENRICHMENT_CONFIG

def load_entities(file_path=None):
    """
    Load entities from a file or use default
    
    Args:
        file_path: Optional path to entity data file
        
    Returns:
        tuple: (entities, enrichment_config)
    """
    if file_path:
        import json
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                entities = data.get("entities", ENTITIES)
                enrichment_config = data.get("enrichment", ENRICHMENT_CONFIG)
                return entities, enrichment_config
        except Exception as e:
            from entity_resolution_demo.utils import print_error, print_warning
            print_error(f"Error loading entity data from {file_path}: {e}")
            print_warning("Using default entity data")
    
    return ENTITIES, ENRICHMENT_CONFIG
