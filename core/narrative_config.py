"""
Narrative Configuration Module
==============================
Define narrative domains with prototypes, colors, and metadata.

Each domain configuration includes:
- narrative_id → display_name mapping
- prototype sentences for embedding-based detection
- visualization colors
- domain-specific settings

Usage:
    from narrative_config import EV_CONFIG, AITECH_CONFIG, RETAIL_CONFIG
    
    # Or create custom config
    from narrative_config import NarrativeConfig
    
    my_config = NarrativeConfig(
        name='MyDomain',
        narratives={...},
        prototypes={...},
        colors={...}
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class NarrativeConfig:
    """Configuration for a narrative analysis domain."""
    
    name: str
    narratives: Dict[str, str]  # display_name -> narrative_id
    prototypes: Dict[str, List[str]]  # narrative_id -> list of prototype sentences
    colors: Dict[str, str]  # display_name -> hex color
    
    # Optional settings
    default_threshold: float = 0.30
    default_narrative: Optional[str] = None  # Fallback when none detected
    description: str = ""
    
    def __post_init__(self):
        # Validate that all narratives have prototypes and colors
        for display_name, narrative_id in self.narratives.items():
            if narrative_id not in self.prototypes:
                raise ValueError(f"Missing prototypes for narrative: {narrative_id}")
            if display_name not in self.colors:
                raise ValueError(f"Missing color for narrative: {display_name}")
        
        # Set default narrative if not specified
        if self.default_narrative is None:
            self.default_narrative = list(self.narratives.keys())[0]
    
    @property
    def narrative_ids(self) -> List[str]:
        """Get list of narrative IDs."""
        return list(self.prototypes.keys())
    
    @property
    def display_names(self) -> List[str]:
        """Get list of display names."""
        return list(self.narratives.keys())
    
    def id_to_name(self, narrative_id: str) -> str:
        """Convert narrative ID to display name."""
        for name, nid in self.narratives.items():
            if nid == narrative_id:
                return name
        return narrative_id
    
    def name_to_id(self, display_name: str) -> str:
        """Convert display name to narrative ID."""
        return self.narratives.get(display_name, display_name)
    
    def get_color(self, name_or_id: str) -> str:
        """Get color for narrative (accepts either name or id)."""
        if name_or_id in self.colors:
            return self.colors[name_or_id]
        # Try converting id to name
        display_name = self.id_to_name(name_or_id)
        return self.colors.get(display_name, '#808080')
    
    def to_dict(self) -> dict:
        """Export config as dictionary."""
        return {
            'name': self.name,
            'narratives': self.narratives,
            'prototypes': self.prototypes,
            'colors': self.colors,
            'default_threshold': self.default_threshold,
            'default_narrative': self.default_narrative,
            'description': self.description
        }
    
    def save(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'NarrativeConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def __repr__(self):
        return f"NarrativeConfig(name='{self.name}', narratives={len(self.narratives)})"


# =============================================================================
# EV / Electric Vehicle Domain
# =============================================================================

EV_CONFIG = NarrativeConfig(
    name='Electric Vehicles',
    description='Narratives around electric vehicle adoption, technology, and policy',
    
    narratives={
        'Climate/Environment': 'climate_positive',
        'Barriers to Adoption': 'barriers_to_adoption',
        'Performance': 'performance',
        'Battery Tech': 'battery_technology',
        'Mainstream Adoption': 'mainstream_adoption',
        'Grid Impact': 'grid_impact',
        'Geopolitics': 'geopolitics',
    },
    
    prototypes={
        'climate_positive': [
            "Electric vehicles reduce carbon emissions and help fight climate change",
            "EVs are cleaner and better for the environment than gasoline cars",
            "Transitioning to electric transportation is crucial for environmental sustainability",
            "Zero-emission vehicles are key to reducing pollution and greenhouse gases",
            "Clean energy vehicles help protect the environment and improve air quality",
            "Electric cars produce fewer emissions over their lifetime than internal combustion engines",
            "Switching to EVs is essential for meeting climate goals and reducing our carbon footprint",
        ],
        'barriers_to_adoption': [
            "Electric vehicles have limited driving range and cause range anxiety",
            "EVs are too expensive and cost more than traditional cars",
            "Lack of charging infrastructure makes EVs impractical for many drivers",
            "Long charging times and high upfront costs are major barriers to EV adoption",
            "Concerns about battery life and charging availability limit EV appeal",
            "The high price of electric vehicles puts them out of reach for average consumers",
            "Without more charging stations, electric cars remain inconvenient for long trips",
        ],
        'performance': [
            "Electric vehicles offer instant torque and superior acceleration",
            "EVs provide quick, responsive performance with powerful electric motors",
            "Electric cars deliver impressive speed and fast acceleration",
            "The instant power delivery of electric motors makes EVs incredibly responsive",
            "Electric vehicles outperform gas cars in acceleration and handling",
            "Tesla's Ludicrous mode demonstrates the incredible performance potential of electric powertrains",
        ],
        'battery_technology': [
            "New lithium-ion battery technology increases energy density and range",
            "Advances in battery cell chemistry improve EV performance and efficiency",
            "Battery innovation and breakthroughs are extending electric vehicle range",
            "Improved solid-state batteries promise longer range and faster charging",
            "Battery costs continue to decline, making EVs more affordable",
            "New battery chemistries like LFP and sodium-ion offer alternatives to traditional lithium-ion",
            "Battery recycling and second-life applications improve EV sustainability",
        ],
        'mainstream_adoption': [
            "Electric vehicle sales continue to grow as more consumers choose EVs",
            "Major automakers are committing to all-electric lineups in coming years",
            "EV adoption is accelerating as more affordable models reach the market",
            "Consumers increasingly prefer electric vehicles for their next purchase",
            "Electric cars are becoming mainstream as prices drop and options expand",
            "The tipping point for mass EV adoption is approaching as costs reach parity",
        ],
        'grid_impact': [
            "Electric vehicle charging puts strain on the power grid during peak hours",
            "Utilities must upgrade infrastructure to handle growing EV demand",
            "Smart charging and vehicle-to-grid technology can help balance electricity demand",
            "Renewable energy integration with EV charging creates a cleaner transportation system",
            "The grid needs massive investment to support widespread electric vehicle adoption",
            "Time-of-use pricing encourages EV owners to charge during off-peak hours",
        ],
        'geopolitics': [
            "China dominates the global EV battery supply chain and manufacturing",
            "Trade tensions affect electric vehicle and battery component sourcing",
            "Countries compete for critical minerals needed for EV batteries",
            "Domestic EV production is a matter of national economic security",
            "The shift to electric vehicles reshapes global automotive power dynamics",
            "Rare earth minerals and lithium supply chains create geopolitical dependencies",
        ],
    },
    
    colors={
        'Climate/Environment': '#00C853',
        'Barriers to Adoption': '#FF1744',
        'Performance': '#2196F3',
        'Battery Tech': '#FF6D00',
        'Mainstream Adoption': '#AA00FF',
        'Grid Impact': '#FF4081',
        'Geopolitics': '#00E5FF',
    },
    
    default_threshold=0.30,
    default_narrative='Mainstream Adoption',
)


# =============================================================================
# AI / Technology Domain
# =============================================================================

AITECH_CONFIG = NarrativeConfig(
    name='AI/Technology',
    description='Narratives around artificial intelligence, technology progress, and societal impact',
    
    narratives={
        'Progress & Innovation': 'progress_innovation',
        'Labor Displacement': 'labor_displacement',
        'Social & Cultural Harm': 'social_cultural_harm',
        'Governance & Regulation': 'governance_regulation',
        'Infrastructure Limits': 'infrastructure_limits',
        'Finance & Speculation': 'finance_speculation',
        'Existential Risk': 'existential_risk',
        'Geopolitics & Security': 'geopolitics_security',
        'Inequality & Concentration': 'inequality_concentration',
    },
    
    prototypes={
        'progress_innovation': [
            "Artificial intelligence breakthrough enables new capabilities and applications",
            "AI advances promise to solve complex problems in healthcare, science, and engineering",
            "Machine learning innovations are transforming industries and creating new possibilities",
            "Technology progress accelerates with each generation of more powerful AI systems",
            "AI assistants and tools enhance human productivity and creativity",
            "Large language models demonstrate remarkable reasoning and generation abilities",
            "Deep learning research continues to push the boundaries of what AI can achieve",
        ],
        'labor_displacement': [
            "Artificial intelligence threatens to automate millions of jobs and displace workers",
            "AI and automation will eliminate many traditional employment opportunities",
            "Workers fear losing their jobs to AI systems and intelligent automation",
            "The rise of AI creates economic anxiety about future employment prospects",
            "Automation driven by AI could lead to widespread unemployment and social disruption",
            "Many white-collar jobs are now at risk of being replaced by AI",
            "Companies are using AI to reduce headcount and cut labor costs",
        ],
        'social_cultural_harm': [
            "AI-generated misinformation and deepfakes threaten democratic discourse",
            "Social media algorithms amplify divisive content and spread harmful narratives",
            "AI systems perpetuate bias and discrimination against marginalized groups",
            "Technology addiction and screen time harm mental health, especially for youth",
            "Surveillance technology and AI tracking erode privacy and civil liberties",
            "AI-powered recommendation systems create filter bubbles and polarization",
            "Synthetic media and AI manipulation undermine trust in authentic content",
        ],
        'governance_regulation': [
            "Governments struggle to regulate rapidly evolving AI technology",
            "New AI legislation aims to ensure safety and protect consumer rights",
            "Tech companies face increasing regulatory scrutiny and compliance requirements",
            "International cooperation is needed to govern artificial intelligence development",
            "AI ethics guidelines and standards are being developed by industry and government",
            "Antitrust actions target big tech companies' market dominance",
            "Data privacy regulations like GDPR shape how AI systems can be deployed",
        ],
        'infrastructure_limits': [
            "AI training requires massive computing power and energy consumption",
            "Data center growth strains electricity grids and raises environmental concerns",
            "Semiconductor shortages limit the expansion of AI capabilities",
            "The infrastructure costs of running large AI models are enormous",
            "GPU availability constrains AI development and deployment",
            "Cloud computing capacity struggles to meet growing AI demand",
            "Energy requirements for AI raise sustainability questions",
        ],
        'finance_speculation': [
            "AI stocks surge as investors bet on artificial intelligence growth",
            "The AI boom drives tech valuations to new highs amid speculation",
            "Venture capital floods into AI startups chasing the next big thing",
            "AI hype creates bubble concerns as valuations disconnect from fundamentals",
            "NVIDIA and other AI chip makers see explosive growth in market cap",
            "Investors pour billions into generative AI companies seeking returns",
            "The AI gold rush attracts speculative investment across the sector",
        ],
        'existential_risk': [
            "Advanced AI systems could pose existential threats to humanity",
            "Researchers warn about the dangers of superintelligent AI systems",
            "AI safety concerns grow as capabilities advance toward human-level intelligence",
            "The alignment problem makes it difficult to ensure AI acts in human interests",
            "Uncontrolled AI development could lead to catastrophic outcomes",
            "Leading scientists call for pausing development of powerful AI systems",
            "The race to AGI raises serious questions about civilizational risk",
        ],
        'geopolitics_security': [
            "US and China compete for AI supremacy in a new technological cold war",
            "AI weapons and autonomous systems transform modern warfare",
            "Export controls aim to prevent adversaries from accessing AI technology",
            "National security concerns drive government investment in AI research",
            "AI enhances cyber warfare capabilities and creates new vulnerabilities",
            "Military applications of AI raise ethical and strategic questions",
            "Countries race to develop AI for intelligence and defense applications",
        ],
        'inequality_concentration': [
            "AI benefits concentrate among large tech companies and wealthy nations",
            "The digital divide widens as AI creates winners and losers",
            "Small businesses struggle to compete with AI-powered tech giants",
            "AI exacerbates economic inequality between skilled and unskilled workers",
            "A few companies control the AI infrastructure and capture most value",
            "Developing countries risk being left behind in the AI revolution",
            "The concentration of AI power raises concerns about market fairness",
        ],
    },
    
    colors={
        'Progress & Innovation': '#00C853',
        'Labor Displacement': '#FF6D00',
        'Social & Cultural Harm': '#FF1744',
        'Governance & Regulation': '#2196F3',
        'Infrastructure Limits': '#795548',
        'Finance & Speculation': '#D4AF37',
        'Existential Risk': '#9C27B0',
        'Geopolitics & Security': '#0288D1',
        'Inequality & Concentration': '#E91E63',
    },
    
    default_threshold=0.30,
    default_narrative='Progress & Innovation',
)


# =============================================================================
# Retail Investor Domain
# =============================================================================

RETAIL_CONFIG = NarrativeConfig(
    name='Retail Investor',
    description='Narratives in financial news targeting retail/individual investors',
    
    narratives={
        'Memes/Degen': 'memes_degen',
        'Inflation': 'inflation',
        'Politics/Geopolitics': 'politics_geo',
        'Fed/Monetary': 'fed_monetary',
        'AI Revolution': 'ai_revolution',
        'Crypto': 'crypto',
        'Passive vs Active': 'passive_active',
    },
    
    prototypes={
        'memes_degen': [
            "YOLO into GameStop calls, diamond hands to the moon with the WSB apes",
            "Degenerate gambling on meme stocks and risky options for massive gains or total loss",
            "Wallstreetbets culture of high-risk options trading and community-driven short squeezes",
            "Casino mentality with FDs and YOLO plays chasing lambos or bust",
            "Apes holding AMC and GME together against hedge funds with diamond hands",
            "Reddit retail traders coordinate to squeeze short sellers on meme stocks",
            "The YOLO mentality drives retail investors to take extreme risks for potential rewards",
        ],
        'inflation': [
            "Rising inflation erodes purchasing power and impacts consumer spending ability",
            "CPI increases show persistent inflation affecting cost of living and savings",
            "High prices and inflation concerns drive investors to seek inflation hedges",
            "Wage growth lags behind price increases as inflation reduces real income",
            "Inflation debate between transitory and persistent views affects market outlook",
            "Investors worry about inflation eating away at their retirement savings",
            "The cost of everything from groceries to housing continues to rise with inflation",
        ],
        'politics_geo': [
            "US election results and political policies impact market sentiment and regulation",
            "China tensions and geopolitical conflicts affect global trade and supply chains",
            "Trump tariffs and trade war policies create uncertainty for investors",
            "SEC regulatory actions and government policies shape investment landscape",
            "Geopolitical risks from Ukraine Russia conflict influence market volatility",
            "Political gridlock in Congress affects economic policy and market expectations",
            "International sanctions and trade restrictions reshape global investment flows",
        ],
        'fed_monetary': [
            "Federal Reserve rate hikes aim to control inflation and cool the economy",
            "Powell signals dovish Fed pivot as markets anticipate rate cuts ahead",
            "Quantitative easing and tightening affect liquidity and market conditions",
            "Yield curve inversion signals recession fears as Fed raises rates aggressively",
            "FOMC meeting minutes reveal hawkish stance on monetary policy tightening",
            "Interest rate expectations drive bond yields and stock valuations",
            "The Fed's dual mandate balances employment and price stability goals",
        ],
        'ai_revolution': [
            "ChatGPT and generative AI revolution drives massive investment in tech stocks",
            "NVIDIA dominates AI chip market with GPUs powering machine learning growth",
            "Artificial intelligence disrupts industries and creates new investment opportunities",
            "AI agents and models transform business operations and productivity gains",
            "Semiconductor companies benefit from surging demand for AI computing power",
            "The AI boom creates new winners in the stock market as investors chase growth",
            "Companies integrating AI see their valuations soar on productivity promises",
        ],
        'crypto': [
            "Bitcoin and Ethereum prices surge as cryptocurrency adoption grows",
            "Crypto market volatility creates opportunities and risks for retail investors",
            "Blockchain technology and DeFi promise to revolutionize financial services",
            "Cryptocurrency regulations and SEC enforcement shape the digital asset market",
            "Bitcoin halving events drive speculation about future price movements",
            "Crypto exchanges and custody solutions make digital assets more accessible",
            "NFTs and tokenization create new asset classes for investors to consider",
        ],
        'passive_active': [
            "Index funds and ETFs continue to attract assets from active managers",
            "The passive investing revolution reshapes how retail investors build portfolios",
            "Low-cost index investing outperforms most actively managed funds over time",
            "Debate continues over whether passive investing distorts market efficiency",
            "Target-date funds simplify retirement investing with automatic rebalancing",
            "Active managers struggle to justify fees as passive alternatives grow",
            "The rise of passive investing changes market dynamics and price discovery",
        ],
    },
    
    colors={
        'Memes/Degen': '#FF1744',
        'Inflation': '#FF6D00',
        'Politics/Geopolitics': '#00E5FF',
        'Fed/Monetary': '#2196F3',
        'AI Revolution': '#AA00FF',
        'Crypto': '#00C853',
        'Passive vs Active': '#FF4081',
    },
    
    default_threshold=0.30,
    default_narrative='Passive vs Active',
)


# =============================================================================
# Configuration Registry
# =============================================================================

CONFIG_REGISTRY = {
    'ev': EV_CONFIG,
    'electric_vehicles': EV_CONFIG,
    'aitech': AITECH_CONFIG,
    'ai': AITECH_CONFIG,
    'tech': AITECH_CONFIG,
    'retail': RETAIL_CONFIG,
    'retail_investor': RETAIL_CONFIG,
    'investor': RETAIL_CONFIG,
}


def get_config(domain: str) -> NarrativeConfig:
    """Get configuration by domain name."""
    domain_lower = domain.lower().replace(' ', '_').replace('-', '_')
    if domain_lower in CONFIG_REGISTRY:
        return CONFIG_REGISTRY[domain_lower]
    raise ValueError(f"Unknown domain: {domain}. Available: {list(set(CONFIG_REGISTRY.values()))}")


def list_domains() -> List[str]:
    """List available domain configurations."""
    return list(set(c.name for c in CONFIG_REGISTRY.values()))


# =============================================================================
# Custom Config Builder
# =============================================================================

def create_config_from_dict(config_dict: dict) -> NarrativeConfig:
    """Create a NarrativeConfig from a dictionary specification.
    
    Example:
        config_dict = {
            'name': 'My Domain',
            'narratives': {
                'Positive': 'positive',
                'Negative': 'negative',
            },
            'prototypes': {
                'positive': ['Good things happening', 'Progress being made'],
                'negative': ['Bad things happening', 'Problems arising'],
            },
            'colors': {
                'Positive': '#00FF00',
                'Negative': '#FF0000',
            }
        }
        config = create_config_from_dict(config_dict)
    """
    return NarrativeConfig(**config_dict)


if __name__ == '__main__':
    # Demo
    print("Available Narrative Configurations")
    print("=" * 50)
    
    for domain in list_domains():
        config = get_config(domain)
        print(f"\n{config.name}")
        print(f"  Description: {config.description}")
        print(f"  Narratives: {len(config.narratives)}")
        for name in config.display_names:
            nid = config.name_to_id(name)
            n_protos = len(config.prototypes[nid])
            color = config.get_color(name)
            print(f"    • {name} ({nid}): {n_protos} prototypes, color={color}")
