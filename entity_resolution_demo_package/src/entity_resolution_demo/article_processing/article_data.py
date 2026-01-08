#!/usr/bin/env python3
"""
Article Data Module

Contains article definitions for the entity resolution demo.
"""

# Sample articles for processing
SAMPLE_ARTICLES = [
    # ============================================================================
    # EXISTING ARTICLES
    # ============================================================================

    # Articles with exact matches
    {
        "id": "article1",
        "title": "Global Leaders Summit in Geneva",
        "content": "Vladimir Putin met with Chinese leader Xi Jinping yesterday to discuss trade relations. The meeting was held in Moscow and covered various bilateral issues. The Russian President emphasized the importance of cooperation between the two nations. Joe Biden also attended the summit, representing the United States.",
        "source": "global_news",
        "language": "en"
    },
    
    # Articles with alias matches
    {
        "id": "article2",
        "title": "International Relations Update",
        "content": "POTUS and President Xi discussed climate initiatives during their meeting last week. The talks were described as productive by White House officials. Meanwhile, VVP was notably absent from the discussions, having declined the invitation citing scheduling conflicts.",
        "source": "diplomatic_affairs",
        "language": "en"
    },
    
    # Articles with role-based references (semantic matches)
    {
        "id": "article3",
        "title": "Global Leadership Dynamics",
        "content": "The American President outlined a new foreign policy approach yesterday, focusing on multilateral cooperation. Meanwhile, the Russian leader has taken a more assertive stance in Eastern Europe. The Chinese President's economic initiatives continue to expand global influence through infrastructure investments.",
        "source": "political_analysis",
        "language": "en"
    },
    
    # Articles with ambiguous entity references
    {
        "id": "article4",
        "title": "Michael Jordan's Impact on Modern Science",
        "content": "Michael Jordan has revolutionized the field with his groundbreaking research on machine learning algorithms. His work at Berkeley has influenced countless AI researchers and established new paradigms in statistical approaches to artificial intelligence. Jordan's mathematical frameworks have become standard tools in the field.",
        "source": "science_journal",
        "language": "en"
    },
    {
        "id": "article5",
        "title": "Basketball Legends and Their Legacy",
        "content": "Michael Jordan's impact on basketball cannot be overstated. His six championships with the Chicago Bulls defined an era of the sport. The player known as 'Air Jordan' transformed not just the game, but sports marketing and athlete branding forever. His Airness continues to influence basketball culture decades after his retirement.",
        "source": "sports_illustrated",
        "language": "en"
    },
    {
        "id": "article6",
        "title": "Historical Explorers and Their Discoveries",
        "content": "John Smith played a crucial role in the early Jamestown settlement, establishing relations with local Native American tribes. Captain Smith's leadership was instrumental during the colony's difficult early years. His interactions with Pocahontas have become legendary, though historians debate their actual nature.",
        "source": "history_magazine",
        "language": "en"
    },
    {
        "id": "article7",
        "title": "British Political History",
        "content": "John Smith's leadership of the Labour Party marked a significant turning point in British politics. The UK politician's sudden death in 1994 altered the course of the party, eventually leading to Tony Blair's rise and the creation of 'New Labour'. Many British MPs still reference Smith's principles in parliamentary debates.",
        "source": "uk_politics",
        "language": "en"
    },
    
    # Multilingual articles
    {
        "id": "article8",
        "title": "国連安全保障理事会の会議",
        "content": "国際連合の安全保障理事会は昨日、重要な会議を開催しました。会議では世界保健機関の代表者も参加し、グローバルな健康危機について議論しました。イーロン・マスクのテスラ社も持続可能なエネルギーソリューションについて発表を行いました。",
        "source": "japanese_international_news",
        "language": "ja"
    },
    {
        "id": "article9",
        "title": "تطورات الشرق الأوسط والعلاقات الدولية",
        "content": "أعلنت الأمم المتحدة عن مبادرة جديدة للسلام في المنطقة. وقال المتحدث باسم منظمة الصحة العالمية إن الوضع الصحي يتحسن تدريجياً. وفي سياق آخر، التقى الرئيس الروسي فلاديمير بوتين مع نظيره الصيني لمناقشة العلاقات الثنائية.",
        "source": "arabic_international_news",
        "language": "ar"
    },
    {
        "id": "article10",
        "title": "欧盟与中国贸易关系",
        "content": "欧盟委员会今日宣布了新的贸易政策。欧盟与中国的贸易关系日益紧张，习近平主席表示愿意进行建设性对话。与此同时，联合国秘书长呼吁各方保持冷静，通过外交途径解决分歧。",
        "source": "chinese_business_news",
        "language": "zh"
    },
    
    # Business and technology articles
    {
        "id": "article11",
        "title": "Tech Industry Leadership Changes",
        "content": "The Microsoft CEO announced a major reorganization of the company's cloud division yesterday. Satya Nadella emphasized the importance of AI integration across all product lines. Meanwhile, Apple Chief Executive revealed plans for new augmented reality devices during the annual developer conference in Cupertino.",
        "source": "tech_industry_news",
        "language": "en"
    },
    {
        "id": "article12",
        "title": "Electric Vehicle Market Disruption",
        "content": "Tesla Motors continues to dominate the electric vehicle market despite increasing competition. The company's innovative battery technology remains years ahead of competitors. The Tesla CEO recently tweeted about plans for a more affordable model that could dramatically increase market penetration in developing countries.",
        "source": "automotive_news",
        "language": "en"
    },
    {
        "id": "article13",
        "title": "AI Research Breakthroughs",
        "content": "OpenAI has released a new version of their language model with significantly improved capabilities. The ChatGPT creator claims the system now has enhanced reasoning abilities and reduced hallucinations. The GPT developer is working closely with Microsoft to integrate the technology into productivity tools.",
        "source": "ai_research_journal",
        "language": "en"
    },
    
    # Regulatory and international organization articles
    {
        "id": "article14",
        "title": "EU Regulatory Framework Updates",
        "content": "The European Commission has proposed new regulations for tech companies operating in the digital market. Brussels regulators aim to ensure fair competition and protect consumer privacy. The EC will begin enforcement next year, with significant penalties for non-compliance.",
        "source": "eu_policy_news",
        "language": "en"
    },
    {
        "id": "article15",
        "title": "Global Health Initiatives",
        "content": "The WHO has announced a new program to improve vaccine distribution in developing countries. The W.H.O. Director-General emphasized the importance of equitable access to healthcare. The United Nations will provide logistical support through its humanitarian agencies.",
        "source": "global_health_news",
        "language": "en"
    },
    
    # Geographic and contextual articles
    {
        "id": "article16",
        "title": "Innovation Hubs Around the World",
        "content": "Silicon Valley remains the global center for technology innovation, but other regions are quickly catching up. The Valley's concentration of venture capital and talent continues to attract entrepreneurs from around the world. Meanwhile, the European Union has launched initiatives to boost its own technology sector.",
        "source": "innovation_trends",
        "language": "en"
    },
    {
        "id": "article17",
        "title": "French Economic Policy Shifts",
        "content": "The French President announced new economic measures aimed at boosting growth and reducing unemployment. President Macron's plan includes tax incentives for businesses and increased funding for education. The announcement was met with mixed reactions from various political parties in France.",
        "source": "european_economics",
        "language": "en"
    },
    
    # Articles with multiple entity types
    {
        "id": "article18",
        "title": "Global Technology and Politics",
        "content": "Microsoft Corporation has signed a new agreement with the United Nations to provide cloud services for humanitarian operations. The Tech hub in Silicon Valley is increasingly engaging with international organizations. Meanwhile, POTUS has called for stronger regulations on AI development, specifically mentioning concerns about systems developed by OpenAI.",
        "source": "global_tech_politics",
        "language": "en"
    },
    {
        "id": "article19",
        "title": "International Relations and Business",
        "content": "The Russian President's visit to Beijing included meetings with executives from major Chinese technology companies. Vladimir Putin and President Xi discussed potential collaborations in semiconductor development. Apple Inc. and Tesla have expressed concerns about the geopolitical implications for their supply chains.",
        "source": "international_business",
        "language": "en"
    },
    {
        "id": "article20",
        "title": "Global Leadership Summit",
        "content": "The UN Secretary-General opened the annual leadership summit with a call for greater international cooperation. Joe Biden, Xi Jinping, and Emmanuel Macron delivered keynote addresses highlighting their visions for addressing climate change. The European Commission presented a comprehensive plan for carbon neutrality by 2050.",
        "source": "global_summit_news",
        "language": "en"
    },

    # ============================================================================
    # NEW ARTICLES FOR CHALLENGE TYPES
    # ============================================================================

    # Missing components challenge
    # Expected matches: "Phillip Charles Carr" from "Phil Carr", "Maria Rodriguez" from "Maria Elena Rodriguez"
    {
        "id": "article21",
        "title": "Financial Industry Leaders Gather for Annual Conference",
        "content": "Phil Carr presented the keynote address at yesterday's financial summit in Chicago. The audience was impressed by Carr's analysis of market trends. In related news, Maria Rodriguez from the Environmental Protection Agency announced new sustainability guidelines for corporations. Rodriguez's team has been working on these regulations for over a year.",
        "source": "financial_times",
        "language": "en"
    },

    # Out-of-order components challenge
    # Expected matches: "Carlos Alfonzo Diaz" from "Diaz, Carlos A.", "Yao Ming" from "Ming, Yao"
    {
        "id": "article22",
        "title": "International Art Exhibition Features Diverse Artists",
        "content": "The exhibition includes works by Diaz, Carlos A., whose Cuban-American heritage influences his vibrant paintings. Art critic Ming, Yao provided an insightful analysis of the cultural significance of the collection. The gallery also features pieces by García Márquez, Gabriel, showcasing the writer's lesser-known visual artwork.",
        "source": "art_review",
        "language": "en"
    },

    # Initials challenge
    # Expected matches: "Franklin Delano Roosevelt" from "F.D.R.", "Jennifer Lopez" from "J.Lo"
    {
        "id": "article23",
        "title": "Historical and Cultural Icons Through the Ages",
        "content": "The documentary examines how F.D.R.'s leadership transformed American politics during the Great Depression. Historians compare his policies to those of J.F.K. and L.B.J. in terms of lasting impact. In the entertainment segment, J.Lo's influence on modern pop culture is analyzed alongside other performers like G.R.R.M.'s impact on fantasy literature.",
        "source": "cultural_analysis",
        "language": "en"
    },

    # Nicknames challenge
    # Expected matches: "William Johnson" from "Bill Johnson", "Robert Williams" from "Bob Williams"
    {
        "id": "article24",
        "title": "Community Leaders Honored at Local Ceremony",
        "content": "Bill Johnson was recognized for his 30 years of service in education. The retired teacher from Boston has mentored hundreds of students. Bob Williams received the athletic achievement award for his contributions to youth basketball programs in Louisiana. The ceremony also honored Liz Taylor for her work with local environmental initiatives.",
        "source": "community_news",
        "language": "en"
    },

    # Missing spaces/hyphens challenge
    # Expected matches: "Jean-Claude Van Damme" from "JeanClaude VanDamme", "Sarah Jessica Parker" from "SarahJessica Parker"
    {
        "id": "article25",
        "title": "Entertainment Industry News and Updates",
        "content": "JeanClaude VanDamme announced his return to action films next year. The martial artist's new project begins filming in Brussels. SarahJessica Parker launched a new fashion line inspired by her iconic television character. Meanwhile, NeilPatrick Harris is set to direct an upcoming Broadway production.",
        "source": "entertainment_weekly",
        "language": "en"
    },

    # Phonetic similarity challenge
    # Expected matches: "Sean Connery" from "Shawn Connery", "Rachel Weisz" from "Rachel Wise"
    {
        "id": "article26",
        "title": "Film Industry Retrospective",
        "content": "The documentary celebrates the legacy of Shawn Connery and his iconic roles in cinema history. Film historian Mykel Smith discusses the Scottish actor's influence on the spy genre. Rachel Wise's award-winning performances are also highlighted, with critics praising her versatility across different genres.",
        "source": "film_critique",
        "language": "en"
    },

    # Transliteration differences challenge
    # Expected matches: "Beijing" from "Peking", "Pyotr Ilyich Tchaikovsky" from "Peter Tchaikovsky"
    {
        "id": "article27",
        "title": "Cultural Exchange Programs Expand Globally",
        "content": "The historical connection between Peking and international trade routes was discussed at the symposium. Researchers from Peiching University presented new archaeological findings. The evening concluded with a performance of Peter Tchaikovsky's compositions by the visiting orchestra. Muhammed Salah was appointed as cultural ambassador for the exchange program.",
        "source": "cultural_affairs",
        "language": "en"
    },

    # Semantic similarity challenge
    # Expected matches: "PennyLuck Pharmaceuticals" from "PennyLuck Drugs, Co.", "Mountain View Medical Center" from "Mountain View Healthcare Facility"
    {
        "id": "article28",
        "title": "Business Sector Developments and Healthcare Innovations",
        "content": "PennyLuck Drugs, Co. announced breakthrough research in treatments for rare genetic disorders. The Boston-based pharmaceutical company's stock rose 5% following the news. Mountain View Healthcare Facility unveiled its new state-of-the-art diagnostic imaging center in Colorado Springs. The hospital's expansion will serve patients throughout El Paso County.",
        "source": "business_health_news",
        "language": "en"
    },

    # Conceptual queries challenge
    # Expected matches: "Elon Musk" from "tech billionaire", "COVID-19" from "global pandemic that began in 2019"
    {
        "id": "article29",
        "title": "Global Challenges and Technological Solutions",
        "content": "The tech billionaire announced plans to accelerate sustainable energy adoption through innovative battery technology. The electric car company CEO emphasized the importance of reducing carbon emissions. Meanwhile, researchers continue to study the long-term effects of the global pandemic that began in 2019, with particular focus on respiratory health outcomes.",
        "source": "tech_health_review",
        "language": "en"
    },

    # Titles and honorifics challenge
    # Expected matches: "Dr. Jane Wilson" from "Professor Jane Wilson", "General James Williams" from "Gen. James Williams"
    {
        "id": "article30",
        "title": "Distinguished Leaders in Their Fields",
        "content": "Professor Jane Wilson received the prestigious award for her contributions to chemistry research at Stanford. Dr. Wilson's groundbreaking work has applications in sustainable energy. Gen. James Williams shared insights on international security at the defense symposium. The retired military officer emphasized the importance of diplomatic solutions alongside strategic preparedness.",
        "source": "professional_achievements",
        "language": "en"
    },

    # Truncated names challenge
    # Expected matches: "Christopher Montgomery" from "Chris Mont", "Alexandria Rodriguez" from "Alex Rod"
    {
        "id": "article31",
        "title": "Business and Media Personalities in the Spotlight",
        "content": "Chris Mont announced a major investment strategy for the upcoming fiscal year. The New York-based banker predicts significant market changes. Alex Rod's latest investigative report for The Washington Post has garnered national attention. Colleagues praise Rod's dedication to thorough journalism and fact-checking.",
        "source": "business_media_news",
        "language": "en"
    }
]

# Article processing configuration
NER_CONFIG = {
    "use_hybrid_ner": True,
    "elasticsearch_ner_model": "facebookai__xlm-roberta-large-finetuned-conll03-english",
    "extract_compound_entities": True
}

def get_articles():
    """
    Get the list of sample articles
    
    Returns:
        list: List of article dictionaries
    """
    return SAMPLE_ARTICLES

def get_ner_config():
    """
    Get the NER configuration
    
    Returns:
        dict: NER configuration
    """
    return NER_CONFIG

def load_articles(file_path=None):
    """
    Load articles from a file or use default
    
    Args:
        file_path: Optional path to article data file
        
    Returns:
        tuple: (articles, ner_config)
    """
    if file_path:
        import json
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                articles = data.get("articles", SAMPLE_ARTICLES)
                ner_config = data.get("ner", NER_CONFIG)
                return articles, ner_config
        except Exception as e:
            from entity_resolution_demo.utils import print_error, print_warning
            print_error(f"Error loading article data from {file_path}: {e}")
            print_warning("Using default article data")
    
    return SAMPLE_ARTICLES, NER_CONFIG
