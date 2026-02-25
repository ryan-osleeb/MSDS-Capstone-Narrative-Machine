"""
Canonical News Dataset Unification
===================================
Unifies GDELT and NYTimes scraper outputs into analysis-ready tables.

Output Schema:
- doc_id: Unique identifier (source_type + hash)
- source_type: 'nyt' or 'gdelt'
- domain: Publisher domain (e.g., 'nytimes.com', 'reuters.com')
- outlet: Human-readable outlet name
- published_at: ISO 8601 timestamp
- url: Original article URL
- title: Article headline/title
- section: Content section (if available)
- full_text: Complete article text (if available)
- snippet: Abstract or lead paragraph
- topic_label: Content topic (EV, AI/Tech, Retail Investor, etc.)
- language: ISO language code (default 'en')
- text_len: Character count of full_text
- extraction_success: Boolean - did we get usable content?
- duplicate_group_id: Hash-based deduplication group

Usage:
    from canonical_news_dataset import CanonicalDatasetBuilder
    
    builder = CanonicalDatasetBuilder()
    
    # Load and transform NYT data
    builder.load_nyt_csv('nyt_articles.csv', topic_label='AI/Tech')
    
    # Load and transform GDELT data
    builder.load_gdelt_csv('historical_news.csv', topic_label='Retail Investor')
    
    # Build unified dataset
    unified_df = builder.build_unified_dataset()
    
    # Export
    builder.export_all('output_dir/')
"""

import pandas as pd
import numpy as np
import hashlib
import re
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

UNIFIED_SCHEMA = {
    'doc_id': str,              # Unique identifier
    'source_type': str,         # 'nyt' or 'gdelt'
    'domain': str,              # Publisher domain
    'outlet': str,              # Human-readable outlet name
    'published_at': str,        # ISO 8601 timestamp
    'url': str,                 # Original article URL
    'title': str,               # Headline
    'section': str,             # Content section
    'full_text': str,           # Complete text (if available)
    'snippet': str,             # Abstract/lead paragraph
    'topic_label': str,         # Content topic classification
    'language': str,            # ISO language code
    'text_len': int,            # Character count
    'extraction_success': bool, # Did we get usable content?
    'duplicate_group_id': str,  # Deduplication hash
}

# Domain to outlet name mapping
DOMAIN_OUTLET_MAP = {
    'nytimes.com': 'The New York Times',
    'bloomberg.com': 'Bloomberg',
    'cnbc.com': 'CNBC',
    'reuters.com': 'Reuters',
    'marketwatch.com': 'MarketWatch',
    'finance.yahoo.com': 'Yahoo Finance',
    'barrons.com': "Barron's",
    #'fool.com': 'The Motley Fool',
    #'seekingalpha.com': 'Seeking Alpha',
    #'wsj.com': 'The Wall Street Journal',
    #'ft.com': 'Financial Times',
    'businessinsider.com': 'Business Insider'
    #'forbes.com': 'Forbes',
    #'thestreet.com': 'TheStreet',
    #'investopedia.com': 'Investopedia',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url or pd.isna(url):
        return ''
    try:
        parsed = urlparse(str(url))
        domain = parsed.netloc.lower()
        # Remove 'www.' prefix
        domain = re.sub(r'^www\.', '', domain)
        return domain
    except Exception:
        return ''


def domain_to_outlet(domain: str) -> str:
    """Convert domain to human-readable outlet name."""
    if not domain:
        return 'Unknown'
    
    # Check exact match first
    if domain in DOMAIN_OUTLET_MAP:
        return DOMAIN_OUTLET_MAP[domain]
    
    # Check if domain contains any known domains
    for known_domain, outlet in DOMAIN_OUTLET_MAP.items():
        if known_domain in domain:
            return outlet
    
    # Fallback: capitalize domain
    return domain.replace('.com', '').replace('.', ' ').title()


def normalize_timestamp(date_val: Any) -> str:
    """Convert various date formats to ISO 8601."""
    if pd.isna(date_val) or date_val is None:
        return ''
    
    date_str = str(date_val).strip()
    
    # Already ISO format
    if re.match(r'^\d{4}-\d{2}-\d{2}T', date_str):
        return date_str[:19]  # Truncate to seconds
    
    # Try common formats
    formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ',  # NYT format with microseconds
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%Y%m%d',  # GDELT format
        '%m/%d/%Y',
        '%d/%m/%Y',
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str[:len(date_str)], fmt)
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
        except ValueError:
            continue
    
    # Last resort: try to extract date components
    date_match = re.search(r'(\d{4})[-/]?(\d{2})[-/]?(\d{2})', date_str)
    if date_match:
        y, m, d = date_match.groups()
        return f"{y}-{m}-{d}T00:00:00"
    
    return ''


def generate_doc_id(source_type: str, url: str, title: str = '', date: str = '') -> str:
    """Generate unique document ID from content hash."""
    # Combine identifiable elements
    content = f"{source_type}|{url}|{title}|{date}"
    hash_val = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    return f"{source_type}_{hash_val}"


def generate_duplicate_group_id(title: str, url: str) -> str:
    """Generate deduplication hash based on normalized title and URL."""
    # Normalize title
    title_norm = re.sub(r'[^\w\s]', '', str(title).lower())
    title_norm = re.sub(r'\s+', ' ', title_norm).strip()
    
    # Get URL path without query params
    url_clean = str(url).split('?')[0].rstrip('/')
    
    # Combine and hash
    content = f"{title_norm}|{url_clean}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


def clean_text(text: Any) -> str:
    """Clean and normalize text content."""
    if pd.isna(text) or text is None:
        return ''
    
    text = str(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common junk patterns
    text = re.sub(r'Advertisement\s*-?\s*scroll to continue reading', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?paywall.*?\]', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def deep_clean_article_text(text: Any, max_chars: int = 5000) -> str:
    """
    Aggressively clean web-scraped article text by removing boilerplate,
    navigation, sidebar content, cookie banners, and other non-article junk.
    
    This is critical for GDELT articles scraped from Internet Archive, which
    include full page HTML converted to text — sidebars, recipe widgets,
    trending stories, newsletter signup forms, etc.
    
    Parameters
    ----------
    text : str
        Raw extracted text from a web page.
    max_chars : int
        Maximum characters to keep (truncates from the end). Article body
        is usually in the first few thousand chars; later content is
        typically sidebar/footer junk.
    
    Returns
    -------
    str
        Cleaned article text.
    """
    if pd.isna(text) or text is None:
        return ''
    
    text = str(text)
    if len(text) < 20:
        return text.strip()
    
    # ── Phase 1: Remove obvious web boilerplate patterns ──
    
    # Cookie / privacy banners
    text = re.sub(
        r'(we use cookies|cookie policy|accept cookies|privacy policy|'
        r'by continuing to|consent to|manage preferences|'
        r'this site uses cookies|we value your privacy).*?(\.|$)',
        '', text, flags=re.IGNORECASE
    )
    
    # Newsletter / subscription prompts
    text = re.sub(
        r'(sign up for|subscribe to|enter your email|'
        r'get our newsletter|join our mailing list|'
        r'subscribe now|newsletter signup|'
        r'delivered to your inbox|morning briefing|'
        r'breaking news alerts|download our app).*?(\.|$)',
        '', text, flags=re.IGNORECASE
    )
    
    # Social media share buttons / prompts
    text = re.sub(
        r'(share this|share on|follow us on|tweet this|'
        r'facebook|twitter|instagram|linkedin|pinterest|'
        r'share via email|print this article|'
        r'click to share|copied to clipboard).*?\s',
        ' ', text, flags=re.IGNORECASE
    )
    
    # Ad/paywall markers
    text = re.sub(
        r'(advertisement|sponsored content|paid partner|'
        r'scroll to continue|continue reading|'
        r'read more below|skip to content|'
        r'already a subscriber|log in to read|'
        r'subscribe for full access).*?(\.|$)',
        '', text, flags=re.IGNORECASE
    )
    
    # Navigation / section headers (common in scraped pages)
    text = re.sub(
        r'(skip to main content|skip to navigation|'
        r'main menu|site navigation|breadcrumb|'
        r'back to top|table of contents|'
        r'related articles|recommended for you|'
        r'more from|you may also like|'
        r'trending now|most popular|most read|'
        r'editors.? picks|staff picks).*?(\.|$)',
        '', text, flags=re.IGNORECASE
    )
    
    # Footer junk
    text = re.sub(
        r'(all rights reserved|copyright ©?|terms of service|'
        r'terms of use|contact us|about us|'
        r'careers at|advertise with|'
        r'do not sell my|california privacy|'
        r'©\s*\d{4}).*?(\.|$)',
        '', text, flags=re.IGNORECASE
    )
    
    # ── Phase 2: Remove food/recipe content (common sidebar junk) ──
    # These appear when news sites have recipe widgets or lifestyle sidebars
    text = re.sub(
        r'(recipe|ingredients?:?|directions:?|prep time|cook time|'
        r'servings?:?|calories per|nutrition facts|'
        r'tablespoons?|teaspoons?|cups? of|ounces? of|'
        r'preheat oven|bake for|simmer|sauté).*?(\.|$)',
        '', text, flags=re.IGNORECASE
    )
    
    # Remove lines that look like recipe lists or ingredient lists
    lines = text.split('.')
    filtered_lines = []
    food_words = {'chicken', 'broccoli', 'salad', 'sauce', 'dish', 'flavor',
                  'recipe', 'cooking', 'bake', 'roast', 'fry', 'boil',
                  'ingredient', 'tablespoon', 'teaspoon', 'oven', 'stir',
                  'cream', 'butter', 'cheese', 'pasta', 'soup', 'cake',
                  'chocolate', 'garlic', 'onion', 'pepper', 'flour', 'sugar',
                  'cinnamon', 'vanilla', 'dough', 'dessert', 'appetizer'}
    
    for line in lines:
        line_lower = line.lower()
        food_count = sum(1 for w in food_words if w in line_lower)
        # Skip sentences with 2+ food words (likely recipe content)
        if food_count >= 2:
            continue
        filtered_lines.append(line)
    
    text = '.'.join(filtered_lines)
    
    # ── Phase 3: Remove common non-article content patterns ──
    
    # "Read more: [title]" or "See also: [title]" blocks
    text = re.sub(r'(read more|see also|related|watch:)\s*:?\s*[A-Z].*?(\.|$)',
                  '', text, flags=re.IGNORECASE)
    
    # Image captions and credits
    text = re.sub(r'(photo|image|credit|caption|source|getty|reuters|ap photo|'
                  r'afp|shutterstock|file photo):?\s*.*?(\.|$)',
                  '', text, flags=re.IGNORECASE)
    
    # Repeated whitespace cleanup
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.\s*\.', '.', text)
    text = re.sub(r'\s+\.', '.', text)
    
    # ── Phase 4: Truncate to article body ──
    # After cleaning, limit to max_chars. Real article text is front-loaded;
    # anything remaining at the tail is likely footer/sidebar content.
    text = text.strip()
    if len(text) > max_chars:
        # Try to truncate at a sentence boundary
        cutoff = text[:max_chars].rfind('.')
        if cutoff > max_chars * 0.7:
            text = text[:cutoff + 1]
        else:
            text = text[:max_chars]
    
    return text.strip()


def detect_language(text: str) -> str:
    """Simple language detection (default to English)."""
    # For now, assume English. Could integrate langdetect later.
    return 'en'


def calculate_extraction_success(text: str, title: str, min_text_len: int = 100) -> bool:
    """Determine if extraction was successful based on content quality."""
    has_title = bool(title and len(str(title).strip()) > 5)
    has_text = bool(text and len(str(text).strip()) >= min_text_len)
    
    return has_title or has_text


# =============================================================================
# NYT TRANSFORMER
# =============================================================================

class NYTTransformer:
    """Transform NYTimes scraper output to canonical format."""
    
    def __init__(self, topic_label: str = 'General'):
        self.topic_label = topic_label
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform NYT DataFrame to canonical schema."""
        records = []
        
        for _, row in df.iterrows():
            # Combine text fields for full_text
            full_text = self._combine_text_fields(row)
            
            # Get best snippet
            snippet = self._get_best_snippet(row)
            
            # Build canonical record
            record = {
                'source_type': 'nyt',
                'domain': 'nytimes.com',
                'outlet': 'The New York Times',
                'published_at': normalize_timestamp(row.get('date', '')),
                'url': str(row.get('web_url', '')),
                'title': clean_text(row.get('headline', '')),
                'section': str(row.get('section', '')).strip() or str(row.get('news_desk', '')).strip(),
                'full_text': full_text,
                'snippet': snippet,
                'topic_label': self.topic_label,
                'language': 'en',
                'text_len': len(full_text),
                'extraction_success': calculate_extraction_success(full_text, row.get('headline', '')),
            }
            
            # Generate IDs
            record['doc_id'] = generate_doc_id('nyt', record['url'], record['title'], record['published_at'])
            record['duplicate_group_id'] = generate_duplicate_group_id(record['title'], record['url'])
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _combine_text_fields(self, row: pd.Series) -> str:
        """Combine available text fields into full_text."""
        parts = []
        
        # Priority order for text content
        fields = ['lead_paragraph', 'abstract', 'snippet']
        
        for field in fields:
            text = clean_text(row.get(field, ''))
            if text and text not in parts:  # Avoid duplicates
                parts.append(text)
        
        return '\n\n'.join(parts)
    
    def _get_best_snippet(self, row: pd.Series) -> str:
        """Get the best available snippet/abstract."""
        for field in ['abstract', 'snippet', 'lead_paragraph']:
            val = clean_text(row.get(field, ''))
            if val:
                return val[:500]  # Cap at 500 chars
        return ''


# =============================================================================
# GDELT TRANSFORMER
# =============================================================================

class GDELTTransformer:
    """Transform GDELT scraper output to canonical format."""
    
    def __init__(self, topic_label: str = 'General'):
        self.topic_label = topic_label
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform GDELT DataFrame to canonical schema."""
        records = []
        
        for _, row in df.iterrows():
            url = str(row.get('url', ''))
            domain = extract_domain(url)
            
            # Get text content (GDELT workflow produces 'text' and 'title')
            # Use deep cleaning for GDELT text — it comes from full webpage scrapes
            # via Internet Archive and includes sidebars, recipes, nav menus, etc.
            full_text = deep_clean_article_text(row.get('text', ''))
            title = clean_text(row.get('title', ''))
            
            # Build canonical record
            record = {
                'source_type': 'gdelt',
                'domain': domain,
                'outlet': domain_to_outlet(domain),
                'published_at': normalize_timestamp(row.get('date', row.get('original_date', ''))),
                'url': url,
                'title': title,
                'section': '',  # GDELT doesn't provide section info
                'full_text': full_text,
                'snippet': full_text[:500] if full_text else '',
                'topic_label': self.topic_label,
                'language': detect_language(full_text),
                'text_len': len(full_text),
                'extraction_success': row.get('success', calculate_extraction_success(full_text, title)),
            }
            
            # Add GDELT-specific metadata if available
            self._add_gdelt_metadata(record, row)
            
            # Generate IDs
            record['doc_id'] = generate_doc_id('gdelt', record['url'], record['title'], record['published_at'])
            record['duplicate_group_id'] = generate_duplicate_group_id(record['title'], record['url'])
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _add_gdelt_metadata(self, record: dict, row: pd.Series) -> None:
        """Store GDELT-specific fields as JSON metadata."""
        gdelt_fields = ['EventCode', 'QuadClass', 'sentiment', 'impact', 
                        'NumMentions', 'Actor1Name', 'Actor2Name', 'archive_url']
        
        metadata = {}
        for field in gdelt_fields:
            if field in row and pd.notna(row[field]):
                metadata[field] = row[field]
        
        if metadata:
            record['gdelt_metadata'] = json.dumps(metadata)


# =============================================================================
# CANONICAL DATASET BUILDER
# =============================================================================

class CanonicalDatasetBuilder:
    """Main class for building unified news datasets."""
    
    def __init__(self):
        self.nyt_raw: Optional[pd.DataFrame] = None
        self.gdelt_raw: Optional[pd.DataFrame] = None
        self.nyt_canonical: Optional[pd.DataFrame] = None
        self.gdelt_canonical: Optional[pd.DataFrame] = None
        self.unified: Optional[pd.DataFrame] = None
        
        self._topic_labels = {}
    
    def load_nyt_csv(self, filepath: str, topic_label: str = 'General') -> pd.DataFrame:
        """Load and transform NYT CSV data."""
        print(f"Loading NYT data from {filepath}...")
        
        self.nyt_raw = pd.read_csv(filepath)
        print(f"  ✓ Loaded {len(self.nyt_raw):,} raw NYT records")
        
        transformer = NYTTransformer(topic_label=topic_label)
        self.nyt_canonical = transformer.transform(self.nyt_raw)
        
        print(f"  ✓ Transformed to {len(self.nyt_canonical):,} canonical records")
        self._print_quality_summary(self.nyt_canonical, 'NYT')
        
        return self.nyt_canonical
    
    def load_gdelt_csv(self, filepath: str, topic_label: str = 'General') -> pd.DataFrame:
        """Load and transform GDELT CSV data."""
        print(f"Loading GDELT data from {filepath}...")
        
        self.gdelt_raw = pd.read_csv(filepath)
        print(f"  ✓ Loaded {len(self.gdelt_raw):,} raw GDELT records")
        
        transformer = GDELTTransformer(topic_label=topic_label)
        self.gdelt_canonical = transformer.transform(self.gdelt_raw)
        
        print(f"  ✓ Transformed to {len(self.gdelt_canonical):,} canonical records")
        self._print_quality_summary(self.gdelt_canonical, 'GDELT')
        
        return self.gdelt_canonical
    
    def load_nyt_dataframe(self, df: pd.DataFrame, topic_label: str = 'General') -> pd.DataFrame:
        """Load NYT data from DataFrame."""
        self.nyt_raw = df
        transformer = NYTTransformer(topic_label=topic_label)
        self.nyt_canonical = transformer.transform(df)
        return self.nyt_canonical
    
    def load_gdelt_dataframe(self, df: pd.DataFrame, topic_label: str = 'General') -> pd.DataFrame:
        """Load GDELT data from DataFrame."""
        self.gdelt_raw = df
        transformer = GDELTTransformer(topic_label=topic_label)
        self.gdelt_canonical = transformer.transform(df)
        return self.gdelt_canonical
    
    def build_unified_dataset(self, deduplicate: bool = True) -> pd.DataFrame:
        """Combine all loaded datasets into unified table."""
        print("\nBuilding unified dataset...")
        
        dfs = []
        if self.nyt_canonical is not None:
            dfs.append(self.nyt_canonical)
        if self.gdelt_canonical is not None:
            dfs.append(self.gdelt_canonical)
        
        if not dfs:
            raise ValueError("No data loaded. Call load_nyt_csv() or load_gdelt_csv() first.")
        
        # Combine all sources
        self.unified = pd.concat(dfs, ignore_index=True)
        print(f"  ✓ Combined {len(self.unified):,} total records")
        
        # Deduplication
        if deduplicate:
            before = len(self.unified)
            self.unified = self._deduplicate()
            after = len(self.unified)
            print(f"  ✓ Deduplicated: {before:,} → {after:,} records ({before - after:,} duplicates removed)")
        
        # Sort by date
        self.unified = self.unified.sort_values('published_at', ascending=False)
        
        # Ensure schema compliance
        self._enforce_schema()
        
        self._print_quality_summary(self.unified, 'UNIFIED')
        
        return self.unified
    
    def _deduplicate(self) -> pd.DataFrame:
        """Remove exact duplicates and flag near-duplicates."""
        df = self.unified.copy()
        
        # Mark duplicates within each group
        df['is_duplicate'] = df.duplicated(subset=['duplicate_group_id'], keep='first')
        
        # Keep first occurrence (prefer NYT if both sources have same article)
        df = df.sort_values(['duplicate_group_id', 'source_type', 'text_len'], 
                           ascending=[True, True, False])  # NYT before GDELT, longer text preferred
        
        df = df.drop_duplicates(subset=['duplicate_group_id'], keep='first')
        df = df.drop(columns=['is_duplicate'], errors='ignore')
        
        return df
    
    def _enforce_schema(self) -> None:
        """Ensure unified DataFrame matches expected schema."""
        # Add missing columns with defaults
        for col, dtype in UNIFIED_SCHEMA.items():
            if col not in self.unified.columns:
                if dtype == str:
                    self.unified[col] = ''
                elif dtype == int:
                    self.unified[col] = 0
                elif dtype == bool:
                    self.unified[col] = False
        
        # Reorder columns
        schema_cols = list(UNIFIED_SCHEMA.keys())
        extra_cols = [c for c in self.unified.columns if c not in schema_cols]
        self.unified = self.unified[schema_cols + extra_cols]
    
    def _print_quality_summary(self, df: pd.DataFrame, label: str) -> None:
        """Print quality metrics for a dataset."""
        total = len(df)
        if total == 0:
            print(f"  [{label}] No records")
            return
        
        successful = df['extraction_success'].sum()
        avg_len = df[df['text_len'] > 0]['text_len'].mean() if (df['text_len'] > 0).any() else 0
        
        print(f"\n  [{label}] Quality Summary:")
        print(f"    • Total records: {total:,}")
        print(f"    • Extraction success: {successful:,} ({100*successful/total:.1f}%)")
        print(f"    • Average text length: {avg_len:,.0f} chars")
        
        if 'source_type' in df.columns:
            source_counts = df['source_type'].value_counts()
            print(f"    • By source: {dict(source_counts)}")
        
        if 'topic_label' in df.columns:
            topic_counts = df['topic_label'].value_counts()
            if len(topic_counts) > 1:
                print(f"    • By topic: {dict(topic_counts)}")
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate detailed quality report."""
        if self.unified is None:
            return {}
        
        df = self.unified
        
        report = {
            'total_records': len(df),
            'by_source': df['source_type'].value_counts().to_dict(),
            'by_topic': df['topic_label'].value_counts().to_dict(),
            'extraction_success_rate': df['extraction_success'].mean(),
            'avg_text_length': df[df['text_len'] > 0]['text_len'].mean(),
            'date_range': {
                'min': df[df['published_at'] != '']['published_at'].min(),
                'max': df[df['published_at'] != '']['published_at'].max(),
            },
            'unique_outlets': df['outlet'].nunique(),
            'outlets': df['outlet'].value_counts().head(10).to_dict(),
            'duplicate_groups': df['duplicate_group_id'].nunique(),
            'missing_values': {
                col: df[col].isna().sum() + (df[col] == '').sum()
                for col in ['title', 'full_text', 'snippet', 'published_at', 'url']
            }
        }
        
        return report
    
    def export_all(self, output_dir: str, formats: List[str] = ['csv', 'parquet']) -> Dict[str, str]:
        """Export all datasets to specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        datasets = {
            'nyt_raw': self.nyt_raw,
            'gdelt_raw': self.gdelt_raw,
            'nyt_canonical': self.nyt_canonical,
            'gdelt_canonical': self.gdelt_canonical,
            'unified': self.unified,
        }
        
        for name, df in datasets.items():
            if df is not None:
                for fmt in formats:
                    filepath = output_path / f"{name}.{fmt}"
                    if fmt == 'csv':
                        df.to_csv(filepath, index=False)
                    elif fmt == 'parquet':
                        df.to_parquet(filepath, index=False)
                    exports[f"{name}.{fmt}"] = str(filepath)
                    print(f"  ✓ Exported {filepath}")
        
        # Export quality report
        report_path = output_path / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.get_quality_report(), f, indent=2, default=str)
        exports['quality_report.json'] = str(report_path)
        print(f"  ✓ Exported {report_path}")
        
        return exports
    
    def export_unified(self, filepath: str) -> str:
        """Export just the unified dataset."""
        if self.unified is None:
            raise ValueError("No unified dataset. Call build_unified_dataset() first.")
        
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        
        if ext == '.csv':
            self.unified.to_csv(filepath, index=False)
        elif ext == '.parquet':
            self.unified.to_parquet(filepath, index=False)
        elif ext == '.json':
            self.unified.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
        print(f"✓ Exported unified dataset to {filepath}")
        return str(filepath)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_unify(nyt_csv: str = None, gdelt_csv: str = None, 
                nyt_topic: str = 'General', gdelt_topic: str = 'General',
                output_csv: str = 'unified_news.csv') -> pd.DataFrame:
    """
    Quick function to unify datasets with minimal code.
    
    Example:
        df = quick_unify(
            nyt_csv='ai_tech_archive.csv',
            gdelt_csv='historical_news.csv',
            nyt_topic='AI/Tech',
            gdelt_topic='Retail Investor',
            output_csv='unified_news.csv'
        )
    """
    builder = CanonicalDatasetBuilder()
    
    if nyt_csv:
        builder.load_nyt_csv(nyt_csv, topic_label=nyt_topic)
    
    if gdelt_csv:
        builder.load_gdelt_csv(gdelt_csv, topic_label=gdelt_topic)
    
    unified = builder.build_unified_dataset()
    
    if output_csv:
        builder.export_unified(output_csv)
    
    return unified


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("CANONICAL NEWS DATASET BUILDER")
    print("=" * 70)
    
    # Demo mode with sample data
    if len(sys.argv) == 1:
        print("\nUsage:")
        print("  python canonical_news_dataset.py <nyt_csv> <gdelt_csv> [output_dir]")
        print("\nOr use programmatically:")
        print("""
    from canonical_news_dataset import CanonicalDatasetBuilder
    
    builder = CanonicalDatasetBuilder()
    builder.load_nyt_csv('ai_tech_archive.csv', topic_label='AI/Tech')
    builder.load_gdelt_csv('historical_news.csv', topic_label='Retail Investor')
    unified = builder.build_unified_dataset()
    builder.export_all('output/')
        """)
        
        # Create sample demonstration
        print("\n" + "=" * 70)
        print("DEMONSTRATION WITH SAMPLE DATA")
        print("=" * 70)
        
        # Sample NYT data
        nyt_sample = pd.DataFrame([
            {
                'date': '2024-01-15T10:30:00Z',
                'headline': 'AI Companies Race to Build Smarter Chatbots',
                'abstract': 'Major technology firms are investing billions in artificial intelligence research.',
                'snippet': 'The competition to develop advanced AI systems has intensified.',
                'lead_paragraph': 'In Silicon Valley, the race to build the next generation of AI is heating up.',
                'web_url': 'https://www.nytimes.com/2024/01/15/technology/ai-chatbots.html',
                'section': 'Technology',
                'subsection': '',
                'byline': 'By Jane Smith',
                'word_count': 1500,
            },
            {
                'date': '2024-01-10T08:00:00Z',
                'headline': 'Electric Vehicle Sales Surge in 2023',
                'abstract': 'Global EV sales reached record levels last year.',
                'snippet': 'Consumers increasingly opted for electric vehicles.',
                'lead_paragraph': 'The electric vehicle market saw unprecedented growth in 2023.',
                'web_url': 'https://www.nytimes.com/2024/01/10/business/ev-sales-2023.html',
                'section': 'Business',
                'subsection': 'Energy',
                'byline': 'By John Doe',
                'word_count': 1200,
            },
        ])
        
        # Sample GDELT data
        gdelt_sample = pd.DataFrame([
            {
                'date': '20240112',
                'url': 'https://www.cnbc.com/2024/01/12/retail-investors-flock-to-meme-stocks.html',
                'title': 'Retail Investors Return to Meme Stock Trading',
                'text': 'Individual investors are once again piling into speculative stocks. The renewed interest comes amid volatile market conditions. Trading volumes on retail platforms have surged.',
                'success': True,
                'text_length': 250,
                'sentiment': 2.5,
                'impact': 3.0,
            },
            {
                'date': '20240108',
                'url': 'https://www.reuters.com/markets/stocks/market-outlook-2024.html',
                'title': 'Markets Look to Fed for 2024 Direction',
                'text': 'Investors are closely watching Federal Reserve signals for clues about interest rate policy. The central bank meeting could set the tone for markets.',
                'success': True,
                'text_length': 180,
                'sentiment': 1.2,
                'impact': 4.5,
            },
        ])
        
        # Build unified dataset
        builder = CanonicalDatasetBuilder()
        builder.load_nyt_dataframe(nyt_sample, topic_label='AI/Tech')
        builder.load_gdelt_dataframe(gdelt_sample, topic_label='Retail Investor')
        
        unified = builder.build_unified_dataset()
        
        print("\n" + "=" * 70)
        print("SAMPLE OUTPUT")
        print("=" * 70)
        print("\nUnified DataFrame columns:")
        print(list(unified.columns))
        
        print("\nSample records:")
        print(unified[['doc_id', 'source_type', 'outlet', 'title', 'topic_label']].to_string())
        
        print("\nQuality Report:")
        report = builder.get_quality_report()
        for key, value in report.items():
            print(f"  {key}: {value}")
    
    else:
        # Command-line mode
        nyt_csv = sys.argv[1] if len(sys.argv) > 1 else None
        gdelt_csv = sys.argv[2] if len(sys.argv) > 2 else None
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'output'
        
        builder = CanonicalDatasetBuilder()
        
        if nyt_csv and nyt_csv != 'none':
            builder.load_nyt_csv(nyt_csv, topic_label='NYT')
        
        if gdelt_csv and gdelt_csv != 'none':
            builder.load_gdelt_csv(gdelt_csv, topic_label='GDELT')
        
        unified = builder.build_unified_dataset()
        builder.export_all(output_dir)
        
        print(f"\n✓ All outputs saved to {output_dir}/")