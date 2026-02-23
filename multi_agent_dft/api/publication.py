import os
import time
import requests
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
import hashlib
from urllib.parse import quote_plus

from multi_agent_dft.utils.logging import get_logger
from multi_agent_dft.utils.llm import get_llm_model

logger = get_logger(__name__)


class Cache:
    """A simple memory-based cache with TTL and disk backup."""
    
    def __init__(self, ttl: int = 86400, cache_dir: Optional[str] = None):
        """Initialize the cache."""
        self.cache = {}
        self.ttl = ttl  # Time to live in seconds
        self.lock = threading.Lock()
        self.cache_dir = cache_dir
        
        # Create cache directory if specified and doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Optional[Path]:
        """Get the file path for a cache key."""
        if not self.cache_dir:
            return None
        
        # Create a hash of the key to use as filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return Path(self.cache_dir) / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and is not expired."""
        with self.lock:
            # Check memory cache first
            if key in self.cache:
                entry = self.cache[key]
                timestamp = entry.get('timestamp', 0)
                
                # Check if the entry is expired
                if time.time() - timestamp <= self.ttl:
                    logger.debug(f"Memory cache hit for key: {key}")
                    return entry.get('value')
                else:
                    logger.debug(f"Memory cache expired for key: {key}")
                    del self.cache[key]
            
            # Check disk cache if enabled
            file_path = self._get_file_path(key)
            if file_path and file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        entry = json.load(f)
                    
                    timestamp = entry.get('timestamp', 0)
                    if time.time() - timestamp <= self.ttl:
                        # Add to memory cache and return
                        self.cache[key] = entry
                        logger.debug(f"Disk cache hit for key: {key}")
                        return entry.get('value')
                    else:
                        # Remove expired disk cache
                        logger.debug(f"Disk cache expired for key: {key}")
                        file_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error reading disk cache: {e}")
            
            logger.debug(f"Cache miss for key: {key}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        entry = {
            'value': value,
            'timestamp': time.time()
        }
        
        with self.lock:
            # Add to memory cache
            self.cache[key] = entry
            logger.debug(f"Added to memory cache: {key}")
            
            # Add to disk cache if enabled
            file_path = self._get_file_path(key)
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        json.dump(entry, f)
                    logger.debug(f"Added to disk cache: {key}")
                except Exception as e:
                    logger.warning(f"Error writing to disk cache: {e}")


class PublicationAPI:
    """
    Enhanced interface for accessing scientific publication databases.
    """
    
    def __init__(self, api_key: Optional[str] = None, config=None):
        """
        Initialize the publication API.
        
        Args:
            api_key: Optional API key for services that require authentication
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.api_key = api_key or os.environ.get("PUBLICATION_API_KEY", "")
        
        # Get API settings from config if available
        api_config = self.config.get('api', {}).get('publication', {})
        self.default_source = api_config.get('default_source', 'arxiv')
        self.max_results_default = api_config.get('max_results', 10)
        self.cache_ttl = api_config.get('cache_ttl', 86400)  # 24 hours cache
        
        # Cache directory
        cache_dir = api_config.get('cache_dir', os.path.join(os.path.expanduser("~"), ".multi_agent_dft", "publication_cache"))
        
        # User agent
        user_agent = api_config.get('user_agent', 
                                   "Multi-Agent DFT System/1.0 (research assistant; contact@example.org)")
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent
        })
        
        # Initialize cache
        self.cache = Cache(ttl=self.cache_ttl, cache_dir=cache_dir)
        
        # Initialize LLM for parameter extraction
        self.llm_model_name = api_config.get('llm_model', 'default_model')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search(self, query: str, source: str = None, max_results: int = None, 
               from_year: Optional[int] = None, to_year: Optional[int] = None) -> list:
        """
        Search for publications matching a query.
        
        Args:
            query: Search query
            source: Publication source (e.g., "arxiv", "crossref", "pubmed", "semanticscholar")
            max_results: Maximum number of results to return
            from_year: Filter results from this year onwards
            to_year: Filter results up to this year
            
        Returns:
            List of publication dictionaries
        """
        # Use defaults from config if not specified
        source = source or self.default_source
        max_results = max_results or self.max_results_default
        
        # Build cache key with all parameters
        cache_key = f"{source}:{query}:{max_results}:{from_year}:{to_year}"
        cached_results = self.cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Choose the appropriate search method based on source
        if source.lower() == 'arxiv':
            results = self._search_arxiv(query, max_results, from_year, to_year)
        elif source.lower() == 'crossref':
            results = self._search_crossref(query, max_results, from_year, to_year)
        elif source.lower() == 'pubmed':
            results = self._search_pubmed(query, max_results, from_year, to_year)
        elif source.lower() == 'semanticscholar':
            results = self._search_semantic_scholar(query, max_results, from_year, to_year)
        else:
            results = self._search_arxiv(query, max_results, from_year, to_year)  # Default to arXiv
        
        self.cache.set(cache_key, results)
        return results

    def _search_arxiv(self, query: str, max_results: int = 10, 
                      from_year: Optional[int] = None, to_year: Optional[int] = None) -> list:
        """
        Search for publications on arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            from_year: Filter results from this year onwards
            to_year: Filter results up to this year
            
        Returns:
            List of publication dictionaries
        """
        # Add year filters to query if specified
        if from_year or to_year:
            date_query = []
            if from_year:
                date_query.append(f"submittedDate:[{from_year}0101 TO 99991231]")
            if to_year:
                date_query.append(f"submittedDate:[00000101 TO {to_year}1231]")
            
            date_filter = " AND ".join(date_query)
            if query:
                query = f"({query}) AND {date_filter}"
            else:
                query = date_filter
        
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            response = self.session.get("http://export.arxiv.org/api/query", params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            namespaces = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            
            results = []
            for entry in root.findall(".//atom:entry", namespaces):
                try:
                    title = entry.find("atom:title", namespaces).text.strip()
                    abstract = entry.find("atom:summary", namespaces).text.strip()
                    published = entry.find("atom:published", namespaces).text.strip()
                    authors = [author.find("atom:name", namespaces).text.strip() 
                              for author in entry.findall(".//atom:author", namespaces)]
                    arxiv_id = entry.find("atom:id", namespaces).text.split("/")[-1]
                    primary_category = entry.find("arxiv:primary_category", namespaces).attrib["term"]
                    
                    # Extract the year for filtering (if needed)
                    pub_year = datetime.fromisoformat(published.replace('Z', '+00:00')).year
                    
                    # Apply year filtering if needed
                    if (from_year and pub_year < from_year) or (to_year and pub_year > to_year):
                        continue
                    
                    pub = {
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "published_date": published,
                        "id": arxiv_id,
                        "source": "arxiv",
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "journal": "arXiv",
                        "category": primary_category,
                        "year": pub_year,
                        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        "doi": None  # arXiv preprints don't have DOIs initially
                    }
                    results.append(pub)
                except (AttributeError, KeyError) as e:
                    logger.warning(f"Error parsing arXiv entry: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")
            return []

    def _search_crossref(self, query: str, max_results: int = 10,
                         from_year: Optional[int] = None, to_year: Optional[int] = None) -> list:
        """
        Search for publications using Crossref API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            from_year: Filter results from this year onwards
            to_year: Filter results up to this year
            
        Returns:
            List of publication dictionaries
        """
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "order": "desc"
        }
        
        # Add year filters if specified
        filter_parts = []
        if from_year:
            filter_parts.append(f"from-pub-date:{from_year}")
        if to_year:
            filter_parts.append(f"until-pub-date:{to_year}")
        
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        
        try:
            response = self.session.get("https://api.crossref.org/works", params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("message", {}).get("items", []):
                try:
                    # Extract basic metadata
                    title = item.get("title", ["Untitled"])[0]
                    
                    # Extract abstract
                    abstract = ""
                    if "abstract" in item:
                        abstract = item["abstract"]
                    
                    # Extract authors
                    authors = []
                    for author in item.get("author", []):
                        name_parts = []
                        if "given" in author:
                            name_parts.append(author["given"])
                        if "family" in author:
                            name_parts.append(author["family"])
                        if name_parts:
                            authors.append(" ".join(name_parts))
                    
                    # Extract date
                    published_date = None
                    year = None
                    if "published" in item:
                        published_parts = item["published"]
                        if "date-parts" in published_parts:
                            date_parts = published_parts["date-parts"][0]
                            if len(date_parts) >= 1:
                                year = date_parts[0]
                                if len(date_parts) >= 3:
                                    published_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                                elif len(date_parts) >= 2:
                                    published_date = f"{date_parts[0]}-{date_parts[1]:02d}"
                                else:
                                    published_date = f"{date_parts[0]}"
                    
                    # Apply year filtering if needed
                    if (from_year and year and year < from_year) or (to_year and year and year > to_year):
                        continue
                    
                    # Journal information
                    journal = item.get("container-title", ["Unknown"])[0] if item.get("container-title") else "Unknown"
                    
                    # Get DOI and URL
                    doi = item.get("DOI", "")
                    url = f"https://doi.org/{doi}" if doi else ""
                    
                    # Build publication object
                    pub = {
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "published_date": published_date,
                        "id": doi,
                        "source": "crossref",
                        "url": url,
                        "journal": journal,
                        "year": year,
                        "doi": doi
                    }
                    results.append(pub)
                except Exception as e:
                    logger.warning(f"Error parsing Crossref entry: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Crossref: {str(e)}")
            return []

    def _search_semantic_scholar(self, query: str, max_results: int = 10,
                                from_year: Optional[int] = None, to_year: Optional[int] = None) -> list:
        """
        Search for publications using Semantic Scholar API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            from_year: Filter results from this year onwards
            to_year: Filter results up to this year
            
        Returns:
            List of publication dictionaries
        """
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,venue,url,publicationTypes,externalIds"
        }
        
        # Add year range filter if specified
        year_filters = []
        if from_year:
            year_filters.append(f"year>={from_year}")
        if to_year:
            year_filters.append(f"year<={to_year}")
        
        if year_filters:
            params["filter"] = ",".join(year_filters)
        
        try:
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            
            response = self.session.get("https://api.semanticscholar.org/graph/v1/paper/search", 
                                        params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("data", []):
                try:
                    # Extract basic metadata
                    title = item.get("title", "Untitled")
                    abstract = item.get("abstract", "")
                    
                    # Extract authors
                    authors = [author.get("name", "") for author in item.get("authors", [])]
                    
                    # Extract year
                    year = item.get("year")
                    
                    # Apply year filtering if needed
                    if (from_year and year and year < from_year) or (to_year and year and year > to_year):
                        continue
                    
                    # Journal information
                    journal = item.get("venue", "Unknown")
                    
                    # Get identifiers
                    external_ids = item.get("externalIds", {})
                    doi = external_ids.get("DOI", "")
                    arxiv_id = external_ids.get("ARXIV", "")
                    
                    # Get URL
                    url = item.get("url", "")
                    if not url and doi:
                        url = f"https://doi.org/{doi}"
                    
                    # Build publication object
                    pub = {
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "published_date": str(year) if year else None,
                        "id": item.get("paperId", ""),
                        "source": "semanticscholar",
                        "url": url,
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                        "arxiv_id": arxiv_id
                    }
                    results.append(pub)
                except Exception as e:
                    logger.warning(f"Error parsing Semantic Scholar entry: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            return []

    def _search_pubmed(self, query: str, max_results: int = 10,
                      from_year: Optional[int] = None, to_year: Optional[int] = None) -> list:
        """
        Search for publications on PubMed.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            from_year: Filter results from this year onwards
            to_year: Filter results up to this year
            
        Returns:
            List of publication dictionaries
        """
        # Build date range filter
        date_filter = ""
        if from_year or to_year:
            if from_year and to_year:
                date_filter = f" AND {from_year}:{to_year}[dp]"
            elif from_year:
                date_filter = f" AND {from_year}:[dp]"
            elif to_year:
                date_filter = f" AND :{to_year}[dp]"
        
        # Build full query
        full_query = f"{query}{date_filter}"
        
        # First, search for IDs
        params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        
        try:
            search_response = self.session.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            # Get IDs
            ids = search_data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            
            # Fetch details for these IDs
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml"
            }
            
            fetch_response = self.session.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=fetch_params)
            fetch_response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(fetch_response.content)
            
            results = []
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Get article metadata
                    article_meta = article.find(".//Article")
                    if article_meta is None:
                        continue
                    
                    # Extract title
                    title_elem = article_meta.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None and title_elem.text else "Untitled"
                    
                    # Extract abstract
                    abstract_text = []
                    for abstract_part in article_meta.findall(".//AbstractText"):
                        if abstract_part.text:
                            label = abstract_part.get("Label", "")
                            if label:
                                abstract_text.append(f"{label}: {abstract_part.text}")
                            else:
                                abstract_text.append(abstract_part.text)
                    
                    abstract = " ".join(abstract_text)
                    
                    # Extract authors
                    authors = []
                    for author in article_meta.findall(".//Author"):
                        last_name = author.find("LastName")
                        fore_name = author.find("ForeName")
                        
                        name_parts = []
                        if fore_name is not None and fore_name.text:
                            name_parts.append(fore_name.text)
                        if last_name is not None and last_name.text:
                            name_parts.append(last_name.text)
                        
                        if name_parts:
                            authors.append(" ".join(name_parts))
                    
                    # Extract journal info
                    journal_elem = article_meta.find(".//Journal")
                    journal = "Unknown"
                    if journal_elem is not None:
                        journal_title = journal_elem.find(".//Title")
                        if journal_title is not None and journal_title.text:
                            journal = journal_title.text
                    
                    # Extract publication date
                    pub_date = article_meta.find(".//PubDate")
                    year = None
                    published_date = None
                    
                    if pub_date is not None:
                        year_elem = pub_date.find("Year")
                        if year_elem is not None and year_elem.text:
                            year = int(year_elem.text)
                        
                        # Build date string
                        date_parts = []
                        if year_elem is not None and year_elem.text:
                            date_parts.append(year_elem.text)
                        
                        month_elem = pub_date.find("Month")
                        if month_elem is not None and month_elem.text:
                            date_parts.append(month_elem.text)
                        
                        day_elem = pub_date.find("Day")
                        if day_elem is not None and day_elem.text:
                            date_parts.append(day_elem.text)
                        
                        if date_parts:
                            published_date = " ".join(date_parts)
                    
                    # Apply year filtering if needed
                    if (from_year and year and year < from_year) or (to_year and year and year > to_year):
                        continue
                    
                    # Get PMID
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""
                    
                    # Get DOI
                    doi = ""
                    article_ids = article.findall(".//ArticleId")
                    for article_id in article_ids:
                        if article_id.get("IdType", "") == "doi" and article_id.text:
                            doi = article_id.text
                            break
                    
                    # Build URL
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                    if not url and doi:
                        url = f"https://doi.org/{doi}"
                    
                    # Build publication object
                    pub = {
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "published_date": published_date,
                        "id": pmid,
                        "source": "pubmed",
                        "url": url,
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                        "pmid": pmid
                    }
                    results.append(pub)
                except Exception as e:
                    logger.warning(f"Error parsing PubMed entry: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []

    def search_multiple_sources(self, query: str, sources: List[str] = None, max_results: int = None,
                               from_year: Optional[int] = None, to_year: Optional[int] = None) -> list:
        """
        Search for publications across multiple sources.
        
        Args:
            query: Search query
            sources: List of sources to search
            max_results: Maximum number of results to return per source
            from_year: Filter results from this year onwards
            to_year: Filter results up to this year
            
        Returns:
            List of publication dictionaries from all sources
        """
        if sources is None:
            sources = ["arxiv", "crossref", "semanticscholar", "pubmed"]
        
        # Adjust max_results per source
        per_source_max = max(1, int(max_results / len(sources))) if max_results else self.max_results_default
        
        all_results = []
        for source in sources:
            try:
                results = self.search(query, source, per_source_max, from_year, to_year)
                all_results.extend(results)
                logger.info(f"Found {len(results)} results from {source}")
            except Exception as e:
                logger.error(f"Error searching {source}: {str(e)}")
        
        # Sort by relevance or recency
        if all_results:
            # Try to sort by year if available
            all_results.sort(key=lambda x: x.get("year", 0) or 0, reverse=True)
        
        # Limit total results if specified
        if max_results and len(all_results) > max_results:
            all_results = all_results[:max_results]
        
        return all_results

    def get_full_text(self, publication: Dict[str, Any]) -> str:
        """
        Attempt to retrieve the full text of a publication.
        
        Args:
            publication: Publication dictionary
            
        Returns:
            Full text content as a string, or empty string if not available
        """
        source = publication.get("source", "").lower()
        
        # Build cache key
        pub_id = publication.get("id", "")
        cache_key = f"fulltext:{source}:{pub_id}"
        cached_text = self.cache.get(cache_key)
        if cached_text is not None:
            return cached_text
        
        full_text = ""
        
        # Try different methods based on source
        try:
            if source == "arxiv":
                arxiv_id = pub_id
                full_text = self._get_arxiv_fulltext(arxiv_id)
            elif publication.get("doi"):
                # Try to get from DOI
                full_text = self._get_fulltext_from_doi(publication["doi"])
            
            # If we have a URL but no full text yet, try scraping
            if not full_text and publication.get("url"):
                full_text = self._scrape_fulltext(publication["url"])
        except Exception as e:
            logger.error(f"Error retrieving full text: {str(e)}")
        
        # Cache the result
        self.cache.set(cache_key, full_text)
        return full_text

    def _get_arxiv_fulltext(self, arxiv_id: str) -> str:
        """
        Try to get full text from arXiv using the PDF URL.
        
        Args:
            arxiv_id: arXiv ID
            
        Returns:
            Extracted text or empty string
        """
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        try:
            # Note: You would need to implement PDF text extraction
            # This is just a placeholder
            logger.info(f"Would download and extract text from {pdf_url}")
            return f"[Full text extraction from arXiv PDF not implemented]"
        except Exception as e:
            logger.error(f"Error extracting text from arXiv PDF: {str(e)}")
            return ""

    def _get_fulltext_from_doi(self, doi: str) -> str:
        """
        Try to get full text using a DOI.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Extracted text or empty string
        """
        url = f"https://doi.org/{doi}"
        
        try:
            # Try to follow the DOI and scrape the landing page
            return self._scrape_fulltext(url)
        except Exception as e:
            logger.error(f"Error getting full text from DOI: {str(e)}")
            return ""

    def _scrape_fulltext(self, url: str) -> str:
        """
        Try to scrape full text from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted text or empty string
        """
        try:
            response = self.session.get(url, allow_redirects=True)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Look for main content
            main_content = None
            
            # Try common content containers
            for selector in ["article", "main", ".content", "#content", ".article", "#article"]:
                content = soup.select(selector)
                if content:
                    main_content = content[0]
                    break
            
            # If we found a main content section, extract text from it
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                # If no main content found, extract text from the whole page
                text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)
            
            return text
            
        except Exception as e:
            logger.error(f"Error scraping text from URL {url}: {str(e)}")
            return ""
                    
    def extract_parameters(self, publication: Dict[str, Any], dft_code: str, system: str) -> list:
            """
            Extract DFT parameters from a publication for a specific code and system.
            
            Args:
                publication: Publication data (dict with title, abstract, etc.)
                dft_code: Name of the DFT code (gaussian, vasp, cp2k, etc.)
                system: The system being studied
                
            Returns:
                List of dictionaries with parameter information
            """
            # Check if we have a cached result
            cache_key = f"param_extract:{publication.get('id', '')}:{dft_code}:{system}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Try to get full text if available
            text_to_analyze = publication.get("abstract", "")
            full_text = self.get_full_text(publication)
            if full_text:
                # Use the full text if available, otherwise use abstract
                text_to_analyze = full_text
            
            # Create citation string
            author_str = ", ".join(publication.get("authors", [])[:3])
            if len(publication.get("authors", [])) > 3:
                author_str += " et al."
                
            journal = publication.get("journal", "Unknown")
            year = publication.get("year", "")
            citation = f"{author_str}, {journal} {year}"
            
            # Truncate text_to_analyze if it's too long
            if len(text_to_analyze) > 10000:
                text_to_analyze = text_to_analyze[:10000] + "..."
            
            # Prepare the extraction prompt
            prompt = self._create_parameter_extraction_prompt(
                text_to_analyze, dft_code, system, publication.get("title", "")
            )
            
            # Use LLM to extract parameters
            try:
                # Get LLM model
                model = get_llm_model(self.llm_model_name)
                response = model.generate(prompt, max_tokens=1000)
                
                # Process the response
                params = self._parse_parameter_extraction(response, citation, system)
                
                # Cache the result
                self.cache.set(cache_key, params)
                
                return params
            except Exception as e:
                logger.error(f"Error extracting parameters with LLM: {str(e)}")
                
                # Fallback to regex-based extraction
                params = self._extract_parameters_regex(text_to_analyze, citation, dft_code, system)
                
                # Cache the result
                self.cache.set(cache_key, params)
                
                return params
        
    def _create_parameter_extraction_prompt(self, text: str, dft_code: str, system: str, title: str) -> str:
        """
        Create a prompt for parameter extraction.
        
        Args:
            text: Text to analyze
            dft_code: DFT code
            system: System being studied
            title: Publication title
            
        Returns:
            Prompt string
        """
        return f"""
        Extract DFT calculation parameters from the following publication text related to {dft_code.upper()} 
        calculations on {system}.
        
        Publication title: {title}
        
        Focus on numerical and technical parameters such as:
        - For Gaussian: functional, basis set, grid size, convergence criteria, etc.
        - For VASP: ENCUT, KPOINTS, IBRION, ISIF, ISMEAR, etc.
        - For CP2K: Basis sets, XC functionals, cutoffs, etc.
        
        TEXT TO ANALYZE:
        {text}
        
        Extract all parameters that are specifically mentioned for {dft_code.upper()} calculations.
        Format your response as a JSON array of objects with fields:
        - "param_name": The name of the parameter
        - "param_value": The value of the parameter
        - "context": A short phrase describing how the parameter was used (optional)
        
        Example:
        [
            {{
                "param_name": "functional",
                "param_value": "B3LYP",
                "context": "geometry optimization"
            }},
            {{
                "param_name": "basis set",
                "param_value": "6-31G(d)",
                "context": "for non-metal atoms"
            }}
        ]
        
        If no parameters are found, return an empty array: []
        """
    
    def _parse_parameter_extraction(self, response: str, citation: str, system: str) -> list:
        """
        Parse the LLM response to extract parameters.
        
        Args:
            response: LLM response text
            citation: Citation string
            system: System being studied
            
        Returns:
            List of parameter dictionaries
        """
        result = []
        
        # Try to parse JSON from the response
        import json
        import re
        
        # Look for JSON array in the response
        json_match = re.search(r'(\[.*\])', response.replace('\n', ' '), re.DOTALL)
        if json_match:
            try:
                params = json.loads(json_match.group(1))
                for param in params:
                    if isinstance(param, dict) and 'param_name' in param and 'param_value' in param:
                        result.append({
                            'citation': citation,
                            'system': system,
                            'param_name': param['param_name'],
                            'param_value': param['param_value'],
                            'context': param.get('context', '')
                        })
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLM response")
        
        # Fallback: Use regex to extract parameters
        param_patterns = [
            r'(?:parameter|param):\s*([A-Za-z0-9_\-\/]+)\s*value:\s*([A-Za-z0-9_\-\.]+\s*(?:[A-Za-z]+)?)',
            r'([A-Za-z0-9_\-\/]+)\s*=\s*([A-Za-z0-9_\-\.]+\s*(?:[A-Za-z]+)?)',
            r'([A-Za-z0-9_\-\/]+):\s*([A-Za-z0-9_\-\.]+\s*(?:[A-Za-z]+)?)',
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, response)
            for param_name, param_value in matches:
                # Skip common non-parameter matches
                if param_name.lower() in ['title', 'abstract', 'doi', 'author', 'year', 'journal']:
                    continue
                
                result.append({
                    'citation': citation,
                    'system': system,
                    'param_name': param_name.strip(),
                    'param_value': param_value.strip(),
                    'context': ''
                })
        
        return result
    
    def _extract_parameters_regex(self, text: str, citation: str, dft_code: str, system: str) -> list:
        """
        Extract parameters using regex patterns as a fallback method.
        
        Args:
            text: Text to analyze
            citation: Citation string
            dft_code: DFT code
            system: System being studied
            
        Returns:
            List of parameter dictionaries
        """
        result = []
        
        # Define code-specific parameter patterns
        param_patterns = {
            'gaussian': [
                r'(?:using|with)\s+([A-Z][A-Za-z0-9]+)(?:/|\s+and\s+|\s+with\s+)([A-Za-z0-9\-]+(?:\([a-z,+]*\))?)',
                r'(?:functional|method)\s*(?:of|is|was|:)?\s*([A-Z][A-Za-z0-9]+)',
                r'(?:basis set|basis)\s*(?:of|is|was|:)?\s*([A-Za-z0-9\-]+(?:\([a-z,+]*\))?)',
                r'convergence\s+(?:criteria|threshold)(?:\s+(?:of|is|was|:))?\s*([\d\.]+\s*(?:[a-zA-Z]+)?)',
                r'SCF\s+(?:convergence|threshold)(?:\s+(?:of|is|was|:))?\s*([\d\.]+\s*(?:[a-zA-Z]+)?)'
            ],
            'vasp': [
                r'ENCUT\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\d\.]+\s*(?:eV)?)',
                r'energy\s+cutoff\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\d\.]+\s*(?:eV)?)',
                r'(?:k-point|KPOINTS|k point)\s*(?:grid|mesh|sampling)?\s*(?:=|:|\s+is\s+|\s+of\s+)?\s*(\d+\s*×\s*\d+\s*×\s*\d+|\d+x\d+x\d+)',
                r'(?:ISMEAR|smearing)\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\-\d]+)',
                r'(?:SIGMA|width)\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\d\.]+\s*(?:eV)?)',
                r'IBRION\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\-\d]+)',
                r'ISIF\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\d]+)'
            ],
            'cp2k': [
                r'(?:basis set|BASIS_SET)\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([A-Za-z0-9\-]+(?:\-[A-Z]+)?)',
                r'(?:functional|FUNCTIONAL|XC_FUNCTIONAL)\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([A-Z0-9_]+)',
                r'(?:cutoff|CUTOFF)\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\d\.]+\s*(?:Ry|Ha|eV|Hartree)?)',
                r'(?:RELATIVE_CUTOFF|relative cutoff)\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\d\.]+)',
                r'(?:MAX_SCF|scf iterations|max scf)\s*(?:=|:|\s+is\s+|\s+of\s+)\s*([\d]+)'
            ]
        }
        
        # Use generic patterns if code not found
        generic_patterns = [
            r'([A-Za-z][A-Za-z0-9_\-\/]+)\s*=\s*([A-Za-z0-9_\-\.]+\s*(?:[A-Za-z]+)?)',
            r'([A-Za-z][A-Za-z0-9_\-\/]+):\s*([A-Za-z0-9_\-\.]+\s*(?:[A-Za-z]+)?)',
            r'([A-Za-z][A-Za-z0-9_\-\/]+)\s+(?:is|was|were)\s+([A-Za-z0-9_\-\.]+\s*(?:[A-Za-z]+)?)',
        ]
        
        # Get patterns for the specific code or use generic ones
        code_patterns = param_patterns.get(dft_code.lower(), []) + generic_patterns
        
        # Process each pattern
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        param_name, param_value = match
                    else:
                        # For patterns with a single capture group, infer parameter name
                        param_value = match[0]
                        param_name = self._infer_param_name(pattern, dft_code)
                else:
                    # For patterns with a single capture group (string match)
                    param_value = match
                    param_name = self._infer_param_name(pattern, dft_code)
                
                # Skip if either is empty
                if not param_name or not param_value:
                    continue
                
                # Clean up parameter name and value
                param_name = param_name.strip().lower()
                param_value = param_value.strip()
                
                # Skip common non-parameter matches
                if param_name.lower() in ['title', 'abstract', 'doi', 'author', 'year', 'journal']:
                    continue
                
                # Add to results if not already present
                if not any(p['param_name'].lower() == param_name.lower() and 
                           p['param_value'].lower() == param_value.lower() for p in result):
                    result.append({
                        'citation': citation,
                        'system': system,
                        'param_name': param_name,
                        'param_value': param_value,
                        'context': ''
                    })
        
        return result
    
    def _infer_param_name(self, pattern: str, dft_code: str) -> str:
        """
        Infer parameter name from regex pattern.
        
        Args:
            pattern: Regex pattern
            dft_code: DFT code
            
        Returns:
            Inferred parameter name
        """
        # Map patterns to parameter names
        pattern_to_name = {
            r'ENCUT': 'ENCUT',
            r'energy\s+cutoff': 'energy cutoff',
            r'(?:k-point|KPOINTS|k point)': 'kpoints',
            r'(?:ISMEAR|smearing)': 'smearing',
            r'(?:SIGMA|width)': 'sigma',
            r'IBRION': 'IBRION',
            r'ISIF': 'ISIF',
            r'(?:basis set|BASIS_SET)': 'basis set',
            r'(?:functional|FUNCTIONAL|XC_FUNCTIONAL)': 'functional',
            r'(?:cutoff|CUTOFF)': 'cutoff',
            r'(?:RELATIVE_CUTOFF|relative cutoff)': 'relative cutoff',
            r'(?:MAX_SCF|scf iterations|max scf)': 'max scf',
            r'SCF\s+(?:convergence|threshold)': 'SCF convergence',
            r'convergence\s+(?:criteria|threshold)': 'convergence threshold'
        }
        
        for key, name in pattern_to_name.items():
            if re.search(key, pattern, re.IGNORECASE):
                return name
        
        return "parameter"
    
    def analyze_publications(self, publications: list, focus_keywords: list = None) -> dict:
        """
        Analyze a list of publications for trends and insights.
        
        Args:
            publications: List of publication dictionaries
            focus_keywords: List of keywords to focus on
            
        Returns:
            Dictionary containing analysis results
        """
        if not publications:
            return {"error": "No publications to analyze"}
        
        # Use empty list if focus_keywords is None
        focus_keywords = focus_keywords or []
        
        years = []
        for pub in publications:
            date_str = pub.get("published_date", "")
            try:
                if "-" in date_str:
                    year = int(date_str.split("-")[0])
                elif " " in date_str:
                    year = int(date_str.split(" ")[0])
                else:
                    year = int(date_str[:4])
                years.append(year)
            except (ValueError, IndexError):
                pass
                
        # Count keyword occurrences
        keyword_counts = {keyword: 0 for keyword in focus_keywords}
        for pub in publications:
            abstract = pub.get("abstract", "").lower()
            title = pub.get("title", "").lower()
            for keyword in focus_keywords:
                keyword_counts[keyword] += abstract.count(keyword.lower()) + title.count(keyword.lower())
        
        # Journal analysis
        journals = {}
        for pub in publications:
            journal = pub.get("journal", "Unknown")
            journals[journal] = journals.get(journal, 0) + 1
        
        # Author analysis
        author_counts = {}
        for pub in publications:
            for author in pub.get("authors", []):
                author_counts[author] = author_counts.get(author, 0) + 1
        
        # Source analysis
        sources = {}
        for pub in publications:
            source = pub.get("source", "Unknown")
            sources[source] = sources.get(source, 0) + 1
        
        # Year distribution
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        # Create analysis results
        analysis_results = {
            "total_publications": len(publications),
            "year_range": {
                "min": min(years) if years else None,
                "max": max(years) if years else None,
                "distribution": sorted(year_counts.items())
            },
            "keyword_analysis": {
                "counts": keyword_counts,
                "most_common": sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            },
            "journal_analysis": {
                "total_journals": len(journals),
                "most_common": sorted(journals.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "author_analysis": {
                "total_authors": len(author_counts),
                "most_common": sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "source_distribution": sources
        }
    
        return analysis_results
    
    def generate_report(self, publications: list, analysis: dict = None, include_citations: bool = True) -> str:
        """
        Generate a formatted report based on publications and analysis.
        
        Args:
            publications: List of publication dictionaries
            analysis: Optional analysis results dictionary
            include_citations: Whether to include citation references
            
        Returns:
            Formatted report as a string
        """
        if not publications:
            return "No publications available for report generation."
        
        if analysis is None:
            analysis = self.analyze_publications(publications)
        
        # Create a mapping for citations [1], [2], etc.
        citation_map = {}
        if include_citations:
            for i, pub in enumerate(publications, 1):
                citation_map[pub.get('id', f'unknown_{i}')] = i
        
        report = []
        report.append("# Publication Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total publications analyzed: {analysis.get('total_publications', 0)}")
        report.append("")
        
        year_range = analysis.get("year_range", {})
        min_year = year_range.get("min")
        max_year = year_range.get("max")
        if min_year and max_year:
            report.append("## Publication Timeline")
            report.append(f"Publications span from {min_year} to {max_year}")
            report.append("")
            report.append("### Year Distribution")
            for year, count in year_range.get("distribution", []):
                report.append(f"- {year}: {count} publications")
            report.append("")
        
        keyword_analysis = analysis.get("keyword_analysis", {})
        if keyword_analysis.get("most_common"):
            report.append("## Keyword Analysis")
            for keyword, count in keyword_analysis.get("most_common", []):
                report.append(f"- '{keyword}': mentioned {count} times")
            report.append("")
        
        journal_analysis = analysis.get("journal_analysis", {})
        if journal_analysis:
            report.append("## Top Journals")
            for journal, count in journal_analysis.get("most_common", []):
                report.append(f"- {journal}: {count} publications")
            report.append("")
        
        author_analysis = analysis.get("author_analysis", {})
        if author_analysis:
            report.append("## Top Authors")
            for author, count in author_analysis.get("most_common", []):
                report.append(f"- {author}: {count} publications")
            report.append("")
        
        sources = analysis.get("source_distribution", {})
        if sources:
            report.append("## Source Distribution")
            for source, count in sources.items():
                report.append(f"- {source}: {count} publications")
            report.append("")
        
        report.append("## Key Publications")
        recent_pubs = sorted(publications, key=lambda x: x.get('published_date', ""), reverse=True)[:5]
        for i, pub in enumerate(recent_pubs, 1):
            pub_id = pub.get('id', f'unknown_{i}')
            citation_num = ""
            if include_citations and pub_id in citation_map:
                citation_num = f" [{citation_map[pub_id]}]"
                
            report.append(f"### {i}. {pub.get('title', 'Untitled')}{citation_num}")
            report.append(f"**Authors**: {', '.join(pub.get('authors', ['Unknown']))}")
            report.append(f"**Source**: {pub.get('source', 'Unknown')} ({pub.get('journal', 'Unknown')})")
            report.append(f"**Published**: {pub.get('published_date', 'Unknown')}")
            report.append(f"**URL**: {pub.get('url', 'N/A')}")
            report.append("")
            report.append(f"**Abstract**: {pub.get('abstract', 'No abstract available')[:300]}...")
            report.append("")
        
        # Add references section
        if include_citations:
            report.append("## References")
            for i, pub in enumerate(publications, 1):
                # Format authors
                authors = pub.get('authors', ['Unknown'])
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                
                # Format journal and year
                journal = pub.get('journal', 'Unknown')
                year = pub.get('year', '')
                
                # Format title with quotes
                title = f'"{pub.get("title", "Untitled")}"'
                
                # Create citation
                citation = f"[{i}] {author_str}. {title} {journal}, {year}."
                report.append(citation)
        
        return "\n".join(report)
    def get_dft_parameters(self, dft_code: str, system: str, max_results: int = 10) -> list:
        """
        Search for and extract DFT parameters for a specific code and system.
        
        Args:
            dft_code: DFT code (gaussian, vasp, cp2k, etc.)
            system: System being studied (e.g., "metal oxide", "protein-ligand complex")
            max_results: Maximum number of publications to analyze
            
        Returns:
            List of extracted parameters with citation information
        """
        # Create search query
        query = f"{system} {dft_code} DFT parameters"
        
        # Search multiple sources
        publications = self.search_multiple_sources(
            query=query,
            sources=["arxiv", "crossref", "semanticscholar"],
            max_results=max_results
        )
        
        # Extract parameters from each publication
        all_params = []
        for publication in publications:
            params = self.extract_parameters(publication, dft_code, system)
            all_params.extend(params)
        
        # Remove duplicates (same parameter name and value)
        unique_params = []
        seen = set()
        for param in all_params:
            key = (param['param_name'].lower(), param['param_value'].lower())
            if key not in seen:
                seen.add(key)
                unique_params.append(param)
        
        return unique_params

    def batch_extract_parameters(self, publications: List[Dict[str, Any]], dft_code: str, system: str) -> list:
        """
        Extract parameters from a batch of publications.
        
        Args:
            publications: List of publication dictionaries
            dft_code: DFT code
            system: System being studied
            
        Returns:
            List of extracted parameters
        """
        all_params = []
        for publication in publications:
            params = self.extract_parameters(publication, dft_code, system)
            all_params.extend(params)
        return all_params

    def generate_dft_parameter_report(self, parameters: List[Dict[str, Any]], dft_code: str, system: str) -> str:
        """
        Generate a formatted report of DFT parameters.
        
        Args:
            parameters: List of parameter dictionaries
            dft_code: DFT code
            system: System being studied
            
        Returns:
            Formatted report as a string
        """
        if not parameters:
            return f"No parameters found for {dft_code.upper()} calculations on {system}."
        
        report = []
        report.append(f"# DFT Parameter Report for {dft_code.upper()} - {system}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total parameters found: {len(parameters)}")
        report.append("")
        
        # Group parameters by name
        param_groups = {}
        for param in parameters:
            name = param['param_name'].lower()
            if name not in param_groups:
                param_groups[name] = []
            param_groups[name].append(param)
        
        # Report on each parameter group
        for name, params in sorted(param_groups.items()):
            report.append(f"## {name.title()}")
            report.append(f"Found in {len(params)} publication(s)")
            report.append("")
            
            # List individual values
            for param in params:
                context = f" ({param['context']})" if param.get('context') else ""
                report.append(f"- **{param['param_value']}**{context} - {param['citation']}")
            
            report.append("")
        
        # Generate recommendations
        report.append("## Recommendations")
        for name, params in sorted(param_groups.items()):
            # Find most common value
            value_counts = {}
            for param in params:
                value = param['param_value']
                value_counts[value] = value_counts.get(value, 0) + 1
            
            most_common = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[0]
            report.append(f"- For **{name.title()}**, consider using **{most_common[0]}** (found in {most_common[1]} publication(s))")
        
        return "\n".join(report)