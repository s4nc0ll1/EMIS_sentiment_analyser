"""
EMIS API Client 
A robust client for interacting with the EMIS API with proper error handling,
logging, and software engineering best practices.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum

import requests
from dotenv import load_dotenv

# Configure stdout encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIEndpoint(Enum):
    """Enumeration for API endpoints."""
    BASE_URL = "https://api.emis.com"
    COMPANIES_MATCH = "/v2/company/companies/match"
    DOCUMENTS_SEARCH = "/v2/company/documents"


@dataclass
class SearchParameters:
    """Data class for search parameters."""
    country_name: Optional[str] = None
    company_name: Optional[str] = None
    keyword: Optional[str] = None
    industry_name: Optional[str] = None
    limit: int = 100
    order: str = "date:desc"


@dataclass
class CompanyMatch:
    """Data class for company match results."""
    id: str
    name: str


# --- MODIFICACIÓN 1: Añadir campos a la dataclass ---
@dataclass
class Document:
    """Data class for document information."""
    id: str
    date: str
    title: str
    abstract: str
    topics: List[Dict[str, str]] = field(default_factory=list)
    industries: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class SearchResult:
    """Data class for search results."""
    documents: List[Document] = field(default_factory=list)
    country_code: Optional[str] = None
    company_id: Optional[str] = None
    matched_company_name: Optional[str] = None
    industry_code: Optional[str] = None


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


class APIClientError(Exception):
    """Custom exception for API client errors."""
    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load data from file."""
        pass


class JSONDataLoader(DataLoader):
    """JSON file data loader."""
    
    def load(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON data from file with proper error handling."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON data from {file_path}")
            return data
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise DataLoaderError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in {file_path}: {e}"
            logger.error(error_msg)
            raise DataLoaderError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading {file_path}: {e}"
            logger.error(error_msg)
            raise DataLoaderError(error_msg)


class LookupManager:
    """Manages lookup data for countries and industries."""
    
    DATA_DIR = Path("data")
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self._country_lookup: Dict[str, str] = {}
        self._industry_lookup: Dict[str, str] = {}
        self._initialize_lookups()
    
    def _initialize_lookups(self) -> None:
        """Initialize lookup dictionaries."""
        self._load_countries()
        self._load_industries()
    
    def _load_countries(self) -> None:
        """Load countries from JSON file and create lookup dictionary."""
        try:
            countries_file = self.DATA_DIR / "countries.json"
            data = self.data_loader.load(countries_file)
            
            for country in data.get('data', {}).get('items', []):
                if 'name' in country and 'countryCode' in country:
                    self._country_lookup[country['name'].lower()] = country['countryCode']
            
            logger.info(f"Successfully loaded {len(self._country_lookup)} countries")
        except DataLoaderError as e:
            logger.error(f"Failed to load countries: {e}")
    
    def _load_industries(self) -> None:
        """Load industries from JSON file and create lookup dictionary."""
        try:
            industries_file = self.DATA_DIR / "industries.json"
            data = self.data_loader.load(industries_file)
            
            for industry in data:
                if 'name' in industry and 'code' in industry:
                    self._industry_lookup[industry['name'].lower()] = industry['code']
            
            logger.info(f"Successfully loaded {len(self._industry_lookup)} industries")
        except DataLoaderError as e:
            logger.error(f"Failed to load industries: {e}")
    
    def get_country_code(self, country_name: str) -> Optional[str]:
        if not country_name: return None
        code = self._country_lookup.get(country_name.lower())
        if code: logger.info(f"Found country '{country_name}' -> Code: {code}")
        else: logger.warning(f"Country '{country_name}' not found in lookup")
        return code
    
    def get_industry_code(self, industry_name: str) -> Optional[str]:
        if not industry_name: return None
        code = self._industry_lookup.get(industry_name.lower())
        if code: logger.info(f"Found industry '{industry_name}' -> Code: {code}")
        else: logger.warning(f"Industry '{industry_name}' not found in lookup")
        return code
    
    @property
    def countries_count(self) -> int: return len(self._country_lookup)
    
    @property
    def industries_count(self) -> int: return len(self._industry_lookup)


class HTTPClient:
    """HTTP client for making API requests."""
    
    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EMIS-API-Client/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url + endpoint
        request_params = params.copy() if params else {}
        request_params['token'] = self.api_key
        
        try:
            logger.info(f"Making GET request to: {url}")
            response = self.session.get(url, params=request_params, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Request successful: {response.status_code}")
            return response.json()
        
        except requests.exceptions.Timeout:
            raise APIClientError(f"Request timeout for URL: {url}")
        except requests.exceptions.HTTPError as e:
            raise APIClientError(f"HTTP error for URL: {url} - {e}")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"Request failed for URL: {url} - {e}")
        except json.JSONDecodeError:
            raise APIClientError(f"Invalid JSON response from URL: {url}")
    
    def close(self) -> None:
        self.session.close()


class SearchValidator:
    """Validates search parameters."""
    
    @staticmethod
    def validate_search_parameters(params: SearchParameters) -> None:
        if not any([params.country_name, params.company_name, params.keyword, params.industry_name]):
            raise ValidationError("At least one search parameter must be provided")
        if params.limit <= 0: raise ValidationError("Limit must be a positive integer")
        if params.limit > 1000: logger.warning(f"Large limit specified: {params.limit}")
    
    @staticmethod
    def validate_api_key(api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ValidationError("API key cannot be empty")


class DocumentProcessor:
    """Processes document data from API responses."""
    
    @staticmethod
    def process_documents(api_response: Dict[str, Any]) -> List[Document]:
        documents = []
        if not api_response or "data" not in api_response:
            logger.warning("API response missing 'data' field")
            return documents
        
        items = api_response["data"].get("items", [])
        logger.info(f"Processing {len(items)} documents from API response")
        
        for item in items:
            try:
                # --- MODIFICACIÓN 2: Extraer topics e industries ---
                document = Document(
                    id=item.get("id", ""),
                    date=item.get("creationDate", ""),
                    title=item.get("title", "No title"),
                    abstract=item.get("abstract", ""),
                    topics=item.get("topics", []),
                    industries=item.get("industries", [])
                )
                documents.append(document)
            except Exception as e:
                logger.warning(f"Error processing document item: {e}")
                continue
        
        return documents


class CompanyMatcher:
    """Handles company matching operations."""
    
    def __init__(self, http_client: HTTPClient, lookup_manager: LookupManager):
        self.http_client = http_client
        self.lookup_manager = lookup_manager
    
    def match_company(self, company_name: str, country_name: Optional[str] = None) -> Optional[CompanyMatch]:
        if not company_name: return None
        params = {"company_name": company_name}
        
        if country_name:
            if country_code := self.lookup_manager.get_country_code(country_name):
                params["country_code"] = country_code
            else:
                return None
        
        try:
            response = self.http_client.get(APIEndpoint.COMPANIES_MATCH.value, params=params)
            return self._process_company_match_response(response, company_name)
        except APIClientError as e:
            logger.error(f"Company matching failed: {e}")
            return None
    
    def _process_company_match_response(self, response: Dict[str, Any], company_name: str) -> Optional[CompanyMatch]:
        if not response or "data" not in response or not (items := response["data"].get("items")):
            logger.warning(f"No matches found for company '{company_name}'")
            return None
        
        matched_company = items[0]
        if not (company_id := matched_company.get("companyId")):
            logger.warning(f"Matched company missing 'companyId' for '{company_name}'")
            return None
        
        matched_name = matched_company.get("companyName", company_name)
        logger.info(f"Successfully matched company '{matched_name}' (ID: {company_id})")
        return CompanyMatch(id=str(company_id), name=matched_name)


class EmisDocuments:
    """Main EMIS API client for document operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EMIS_API_KEY")
        SearchValidator.validate_api_key(self.api_key)
        
        self.data_loader = JSONDataLoader()
        self.lookup_manager = LookupManager(self.data_loader)
        self.http_client = HTTPClient(APIEndpoint.BASE_URL.value, self.api_key)
        self.company_matcher = CompanyMatcher(self.http_client, self.lookup_manager)
        self.document_processor = DocumentProcessor()
        self.validator = SearchValidator()
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Search for documents based on provided criteria."""
        if not self.api_key: raise ValidationError("EMIS API key not set.")
        
        search_params = SearchParameters(**kwargs)
        self.validator.validate_search_parameters(search_params)
        resolved_params = self._resolve_search_parameters(search_params)
        documents = self._search_documents(resolved_params)
        return self._build_search_result(documents, resolved_params)
    
    def _resolve_search_parameters(self, params: SearchParameters) -> Dict[str, Any]:
        resolved = {"limit": params.limit, "order": params.order}
        
        country_code = self.lookup_manager.get_country_code(params.country_name) if params.country_name else None
        if country_code: resolved["country"] = country_code
        
        company_match = self.company_matcher.match_company(params.company_name, params.country_name) if params.company_name else None
        if company_match: resolved["company_ids"] = company_match.id
        
        industry_code = self.lookup_manager.get_industry_code(params.industry_name) if params.industry_name else None
        if industry_code: resolved["industry"] = industry_code
        
        if params.keyword: resolved["keyword"] = params.keyword
        
        resolved["_metadata"] = {"country_code": country_code, "company_match": company_match, "industry_code": industry_code}
        return resolved
    
    def _search_documents(self, resolved_params: Dict[str, Any]) -> List[Document]:
        api_call_params = resolved_params.copy()
        api_call_params.pop("_metadata", {})
        
        if not any(field in api_call_params for field in ["country", "company_ids", "keyword", "industry"]):
            logger.error("No valid search parameters after resolution")
            return []
        
        logger.info(f"Searching documents with parameters: {api_call_params}")
        try:
            response = self.http_client.get(APIEndpoint.DOCUMENTS_SEARCH.value, params=api_call_params)
            documents = self.document_processor.process_documents(response)
            logger.info(f"Successfully retrieved {len(documents)} documents")
            return documents
        except APIClientError as e:
            logger.error(f"Document search failed: {e}")
            return []
        
    def _build_search_result(self, documents: List[Document], resolved_params: Dict[str, Any]) -> Dict[str, Any]:
        metadata = resolved_params.get("_metadata", {})
        company_match = metadata.get("company_match")
        
        return {
            # --- MODIFICACIÓN 3: Incluir topics e industries en el resultado ---
            "documents": [
                {
                    "id": doc.id,
                    "date": doc.date,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "topics": doc.topics,
                    "industries": doc.industries
                }
                for doc in documents
            ],
            "country_code": metadata.get("country_code"),
            "company_id": company_match.id if company_match else None,
            "matched_company_name": company_match.name if company_match else None,
            "industry_code": metadata.get("industry_code")
        }
    
    def get_stats(self) -> Dict[str, int]:
        return {"countries_loaded": self.lookup_manager.countries_count, "industries_loaded": self.lookup_manager.industries_count}
    
    def close(self) -> None: self.http_client.close()
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()

def create_emis_client(api_key: Optional[str] = None) -> EmisDocuments:
    return EmisDocuments(api_key=api_key)

if __name__ == "__main__":
    try:
        with create_emis_client() as client:
            stats = client.get_stats()
            print(f"Loaded data: {stats}")
            result = client.run(country_name="United States", company_name="Apple Inc", keyword="financial", limit=5)
            print(json.dumps(result, indent=2))
            
    except (ValidationError, APIClientError) as e:
        logger.error(f"Search failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")