import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import re

class WHOBatchProcessor:
    """
    Process ALL WHO fact sheets and prepare for semantic chunking
    Extract structured section-content mapping from HTML pages
    """
    
    def __init__(self, delay_between_requests: float = 2.0):
        self.delay = delay_between_requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Processing statistics
        self.stats = {
            'total_urls': 0,
            'successful': 0,
            'failed': 0,
            'total_words': 0,
            'start_time': None,
            'end_time': None
        }

    def get_comprehensive_fact_sheet_urls(self) -> List[str]:
        """Get ALL WHO fact sheet URLs using multiple methods"""
        print("Starting comprehensive URL discovery...")
        
        all_urls = set()
        
        # Method 1: Main page
        print("Scanning main fact sheets page...")
        main_urls = self._scan_main_page()
        all_urls.update(main_urls)
        print(f"Found {len(main_urls)} URLs from main page")
        
        # Method 2: Try different URL patterns that might exist
        print("Checking alternative URL patterns...")
        alt_urls = self._scan_alternative_patterns()
        new_alt_urls = alt_urls - all_urls
        all_urls.update(new_alt_urls)
        print(f"Found {len(new_alt_urls)} additional URLs from alternatives")
        
        final_urls = sorted(list(all_urls))
        print(f"Discovery complete: {len(final_urls)} total fact sheet URLs")
        
        return final_urls
    
    def _scan_main_page(self) -> set:
        """Scan the main fact sheets page"""
        base_url = "https://www.who.int/news-room/fact-sheets"
        urls = set()
        
        try:
            response = self.session.get(base_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/news-room/fact-sheets/detail/' in href:
                    if href.startswith('http'):
                        urls.add(href)
                    else:
                        urls.add(f"https://www.who.int{href}")
        
        except Exception as e:
            print(f"Error scanning main page: {e}")
        
        return urls
    
    def _scan_alternative_patterns(self) -> set:
        """Try to find fact sheets through alternative patterns"""
        urls = set()
        
        # Try common WHO health topic patterns
        alternative_bases = [
            "https://www.who.int/health-topics",
            "https://www.who.int/news-room/fact-sheets",
        ]
        
        for base in alternative_bases:
            try:
                time.sleep(1)  # Rate limiting
                response = self.session.get(base, timeout=20)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for any links that might lead to fact sheets
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        # Multiple patterns for fact sheet URLs
                        patterns = [
                            r'/news-room/fact-sheets/detail/',
                            r'/fact-sheets?/detail/',
                            r'/health-topics/[^/]+/fact-sheet'
                        ]
                        
                        for pattern in patterns:
                            if re.search(pattern, href):
                                if href.startswith('http'):
                                    urls.add(href)
                                else:
                                    urls.add(f"https://www.who.int{href}")
                                break
            
            except Exception as e:
                print(f"Error with {base}: {e}")
        
        return urls

    def process_single_url(self, url: str) -> Dict:
        """Process a single WHO fact sheet URL and extract section-content mapping"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove noise elements
            for element in soup(['script', 'style', 'noscript', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Get title
            title = "Unknown"
            title_elem = soup.find('title')
            if title_elem:
                title = re.sub(r'\s+', ' ', title_elem.get_text()).strip()
                title = re.sub(r'\s*-?\s*who\s*$', '', title, flags=re.I)
            
            # Find main content
            main_content = (
                soup.find('main') or 
                soup.find('article') or
                soup.find(class_=re.compile(r'content|main|article', re.I)) or
                soup.find('div', class_=re.compile(r'body|content', re.I)) or
                soup.find('body')
            )
            
            if not main_content:
                return {'url': url, 'error': 'No main content found'}
            
            # Extract section-content mapping
            sections_content = {}
            current_section = "Introduction"  # Default section for content before first heading
            current_section_content = []
            
            for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li']):
                
                if elem.name.startswith('h'):
                    # Save previous section if it has content
                    if current_section_content:
                        content_text = ' '.join(current_section_content)
                        if len(content_text.strip()) > 20:
                            sections_content[current_section] = content_text.strip()
                    
                    # Start new section
                    heading_text = re.sub(r'\s+', ' ', elem.get_text()).strip()
                    if heading_text and len(heading_text) > 3:
                        current_section = heading_text
                        current_section_content = []
                    continue
                
                # Extract paragraph text
                text = re.sub(r'\s+', ' ', elem.get_text()).strip()
                
                # Filter and collect valid content
                if (len(text) > 20 and 
                    not self._is_navigation_text(text) and
                    self._has_enough_letters(text)):
                    current_section_content.append(text)
            
            # Don't forget the last section
            if current_section_content:
                content_text = ' '.join(current_section_content)
                if len(content_text.strip()) > 20:
                    sections_content[current_section] = content_text.strip()
            
            # Calculate total word count
            total_words = sum(len(content.split()) for content in sections_content.values())
            
            # Create result
            result = {
                'title': title,
                'url': url,
                'sections_content': sections_content,
                'word_count': total_words
            }
            
            # Update stats
            self.stats['successful'] += 1
            self.stats['total_words'] += total_words
            
            return result
            
        except Exception as e:
            self.stats['failed'] += 1
            return {'url': url, 'error': str(e)}
    
    def _is_navigation_text(self, text: str) -> bool:
        """Check if text is navigation or noise content"""
        text_lower = text.lower()
        
        noise_indicators = [
            'skip to main content', 'world health organization', 
            'contact us', 'about who', 'privacy policy', 'terms of use',
            'follow us', 'subscribe', 'newsletter', 'fact sheets',
            'multimedia', 'podcasts', 'countries', 'regions'
        ]
        
        return any(indicator in text_lower for indicator in noise_indicators)
    
    def _has_enough_letters(self, text: str) -> bool:
        """Check if text has enough alphabetic content"""
        if not text:
            return False
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
        return alpha_ratio > 0.6

    def batch_process_all_diseases(self, max_diseases: Optional[int] = None) -> Dict:
        """Process all WHO diseases with progress tracking"""
        print("Starting batch processing of ALL WHO diseases...")
        print("=" * 60)
        
        self.stats['start_time'] = datetime.now()
        
        # Get all URLs
        urls = self.get_comprehensive_fact_sheet_urls()
        
        if max_diseases:
            urls = urls[:max_diseases]
            print(f"Limited to first {max_diseases} diseases for testing")
        
        self.stats['total_urls'] = len(urls)
        
        print(f"Processing {len(urls)} diseases...")
        print(f"Estimated time: {len(urls) * self.delay / 60:.1f} minutes")
        print("-" * 60)
        
        results = []
        
        for i, url in enumerate(urls, 1):
            disease_name = url.split('/')[-1].replace('-', ' ').title()
            print(f"[{i:3d}/{len(urls)}] Processing: {disease_name}")
            
            result = self.process_single_url(url)
            results.append(result)
            
            # Show progress
            if 'error' not in result:
                section_count = len(result['sections_content'])
                print(f"   ✓ {section_count} sections, {result['word_count']} words")
            else:
                print(f"   Error: {result['error']}")
            
            # Progress summary every 10 items
            if i % 10 == 0:
                success_rate = (self.stats['successful'] / i) * 100
                print(f"   Progress: {success_rate:.1f}% success rate, {self.stats['total_words']:,} total words")
            
            # Rate limiting
            if i < len(urls):
                time.sleep(self.delay)
        
        self.stats['end_time'] = datetime.now()
        
        # Final processing and saving
        return self._finalize_results(results)
    
    def _finalize_results(self, results: List[Dict]) -> Dict:
        """Finalize and save results"""
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        
        # Calculate final stats
        processing_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        # Print summary
        print(f"Total processing time: {processing_time/60:.1f} minutes")
        print(f"Successfully processed: {len(successful_results)} diseases")
        print(f"Failed: {len(failed_results)} diseases")
        print(f"Total words extracted: {self.stats['total_words']:,}")
        
        if successful_results:
            avg_words = self.stats['total_words'] / len(successful_results)
            avg_sections = sum(len(r['sections_content']) for r in successful_results) / len(successful_results)
            print(f"Average per disease: {avg_sections:.1f} sections, {avg_words:.0f} words")
        
        # Save results for chunking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if successful_results:
            # Filter out very short documents
            ready_for_chunking = [r for r in successful_results if r['word_count'] >= 50]
            
            chunking_file = f"who_ready_for_chunking_{timestamp}.json"
            with open(chunking_file, 'w', encoding='utf-8') as f:
                json.dump(ready_for_chunking, f, ensure_ascii=False, indent=2)
            print(f"Chunking-ready data saved to: {chunking_file}")
            print(f"Documents ready for chunking: {len(ready_for_chunking)}")
        
        # Return summary
        return {
            'total_processed': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'total_words': self.stats['total_words'],
            'processing_time_minutes': processing_time / 60,
            'output_file': chunking_file if successful_results else None
        }

# Main execution
if __name__ == "__main__":
    print("WHO COMPREHENSIVE DISEASE DATA PROCESSOR")
    print("=" * 60)
    print("This script will process ALL WHO fact sheets and prepare for chunking")
    print("Estimated time: 10-30 minutes depending on number of diseases")
    print()
    
    # Ask user for confirmation
    response = input("Process ALL diseases? (y/N): ").strip().lower()
    
    if response in ['y', 'Y']:
        processor = WHOBatchProcessor(delay_between_requests=2.0)
        summary = processor.batch_process_all_diseases()
        
        print(f"\nBATCH PROCESSING COMPLETE!")
        if summary['output_file']:
            print(f"Output file: {summary['output_file']}")
        print(f"Next step: Use the output file for semantic chunking")
    
    else:
        # Test mode
        print("Running in TEST MODE with first 5 diseases...")
        processor = WHOBatchProcessor(delay_between_requests=1.0)
        summary = processor.batch_process_all_diseases(max_diseases=5)
        print(f"Test complete! Check the generated file.")
