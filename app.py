"""
EMIS Document Analyzer
A Streamlit application for analyzing EMIS documents with sentiment analysis capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title='EMIS Document Analyzer',
    layout='wide',
    initial_sidebar_state='expanded'
)


@dataclass
class SearchParameters:
    """Data class for search parameters."""
    country_name: Optional[str] = None
    company_name: Optional[str] = None
    keyword: Optional[str] = None
    industry_name: Optional[str] = None
    limit: int = 100


@dataclass
class SearchResult:
    """Data class for search results."""
    documents: List[Dict[str, Any]]
    country_code: Optional[str] = None
    company_id: Optional[str] = None
    matched_company_name: Optional[str] = None
    industry_code: Optional[str] = None


@dataclass
class SentimentAnalysis:
    """Data class for sentiment analysis results."""
    sentiment: str
    score: float
    analysis_text: str


class DataLoader:
    """Handles loading of static data files."""
    
    DATA_DIR = Path("data")
    
    @staticmethod
    @st.cache_data(ttl=3600) # Cache JSON file loading for 1 hour
    def load_json_file(file_path: Path, error_message: str) -> Dict[str, Any]:
        """Load and parse JSON file with error handling."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"Error: {file_path} not found.")
            logger.error(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            st.error(f"JSON decode error in {file_path}: {e}")
            logger.error(f"JSON decode error in {file_path}: {e}")
            return {}
        except Exception as e:
            st.error(f"An error occurred while loading {file_path}: {e}")
            logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    @classmethod
    @st.cache_data(ttl=3600) # Cache countries list for 1 hour
    def load_countries(cls) -> List[str]:
        """Load country names from countries.json."""
        file_path = cls.DATA_DIR / "countries.json"
        data = cls.load_json_file(file_path, "countries")
        
        if not data:
            return [""]
        
        try:
            countries = sorted([
                item['name'] for item in data.get('data', {}).get('items', [])
                if 'name' in item
            ])
            return [""] + countries
        except Exception as e:
            st.error(f"Error processing countries data: {e}")
            logger.error(f"Error processing countries data: {e}")
            return [""]
    
    @classmethod
    @st.cache_data(ttl=3600) # Cache industries list for 1 hour
    def load_industries(cls) -> List[str]:
        """Load industry names from industries.json."""
        file_path = cls.DATA_DIR / "industries.json"
        data = cls.load_json_file(file_path, "industries")
        
        if not data:
            return [""]
        
        try:
            industries = sorted([
                item['name'] for item in data
                if 'name' in item
            ])
            return [""] + industries
        except Exception as e:
            st.error(f"Error processing industries data: {e}")
            logger.error(f"Error processing industries data: {e}")
            return [""]


@st.cache_resource(ttl=None)
class SentimentAnalyzer:
    """Handles sentiment analysis operations."""
    
    def __init__(self):
        self._initialize_nltk()
        self.analyzer = SentimentIntensityAnalyzer()
    
    def _initialize_nltk(self) -> None:
        """Initialize NLTK resources with automatic download."""
        try:
            # Test if vader_lexicon is available
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            test_analyzer = SentimentIntensityAnalyzer()
            test_analyzer.polarity_scores("test")
            logger.info("NLTK vader_lexicon found and working.")
        except (LookupError, FileNotFoundError, OSError) as e:
            # Si no se encuentra, descargarlo automáticamente 
            logger.info(f"NLTK vader_lexicon not found ({e}), downloading...")
            try:
                import nltk
                nltk.download('vader_lexicon', quiet=True)
                logger.info("NLTK vader_lexicon downloaded successfully.")
                
                # Verificar que funciona después de la descarga
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                test_analyzer = SentimentIntensityAnalyzer()
                test_analyzer.polarity_scores("test")
                logger.info("NLTK vader_lexicon verified after download.")
                
            except Exception as download_error:
                error_msg = f"Failed to download NLTK vader_lexicon: {download_error}"
                st.error(error_msg)
                logger.error(error_msg)
                raise RuntimeError(error_msg) from download_error

    def analyze_text(self, text: str) -> SentimentAnalysis:
        """Perform sentiment analysis on text."""
        if not text or not text.strip():
            return SentimentAnalysis(
                sentiment="Unknown",
                score=0.0,
                analysis_text="No text to analyze."
            )
        
        scores = self.analyzer.polarity_scores(text)
        compound_score = scores['compound']
        
        sentiment = self._determine_sentiment(compound_score)
        analysis_text = self._format_analysis_text(sentiment, compound_score, scores)
        
        return SentimentAnalysis(
            sentiment=sentiment,
            score=compound_score,
            analysis_text=analysis_text
        )
    
    def _determine_sentiment(self, compound_score: float) -> str:
        """Determine sentiment based on compound score."""
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def _format_analysis_text(self, sentiment: str, compound_score: float, scores: Dict[str, float]) -> str:
        """Format analysis text for display."""
        return (
            f"The overall sentiment is **{sentiment.lower()}** "
            f"with a compound score of **{compound_score:.2f}**. "
            f"Breakdown: (Positive: {scores['pos']:.2f}, "
            f"Negative: {scores['neg']:.2f}, Neutral: {scores['neu']:.2f})"
        )

class DocumentAnalyzer:
    """Handles document analysis operations."""
    
    def __init__(self):
        # SentimentAnalyzer is now cached globally, so instantiating it here
        # will retrieve the single cached instance.
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_documents(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze sentiment for multiple documents."""
        analyses = []
        
        for doc in documents:
            text_to_analyze = f"{doc.get('title', '')}. {doc.get('abstract', '')}"
            analysis_result = self.sentiment_analyzer.analyze_text(text_to_analyze)
            
            analyses.append({
                "date": doc.get("date"),
                "sentiment": analysis_result.sentiment
            })
        
        if analyses:
            df = pd.DataFrame(analyses)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df
        
        return pd.DataFrame()


class SessionStateManager:
    """Manages Streamlit session state."""
    
    SESSION_KEYS = [
        "logged_in",
        "result",
        "general_analysis_df",
        "search_info",
        "emis_api_token" # <-- Añadir esta clave para el token
    ]
    
    @staticmethod
    def initialize() -> None:
        """Initialize session state variables."""
        defaults = {
            "logged_in": False,
            "result": None,
            "general_analysis_df": None,
            "search_info": None,
            "emis_api_token": None # <-- Inicializar el token en None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def reset_search_data() -> None:
        """Reset search-related session state."""
        search_keys = ["result", "general_analysis_df", "search_info"]
        for key in search_keys:
            st.session_state[key] = None
    
    @staticmethod
    def logout() -> None:
        """Reset all session state on logout."""
        for key in SessionStateManager.SESSION_KEYS:
            st.session_state[key] = None if key != "logged_in" else False
        st.session_state["logged_in"] = False # Asegurar que logged_in es False


class UIComponents:
    """Handles UI component rendering."""
    
    SENTIMENT_COLORS = {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "orange",
        "Unknown": "grey"
    }
    
    @staticmethod
    def load_css() -> None:
        """Load custom CSS from file."""
        css_path = Path("static/styles.css")
        try:
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"{css_path} not found. Skipping custom CSS.")
    
    @staticmethod
    def render_logo() -> None:
        """Render application logo."""
        logo_path = "static/logo2.png"
        try:
            st.logo(logo_path, size="large")
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
    
    @staticmethod
    def render_search_filters() -> SearchParameters:
        """Render search filter sidebar."""
        st.sidebar.header("Search Filters")
        
        # DataLoader methods are now cached
        country_list = DataLoader.load_countries()
        industry_list = DataLoader.load_industries()
        
        selected_country = st.sidebar.selectbox(
            "Country Name (Optional)",
            options=country_list,
            index=0
        )
        
        company_input = st.sidebar.text_input("Company Name (Optional)")
        
        selected_industry = st.sidebar.selectbox(
            "Industry (Optional)",
            options=industry_list,
            index=0
        )
        
        keyword_input = st.sidebar.text_input("Keyword (Optional)")
        
        return SearchParameters(
            country_name=selected_country if selected_country else None,
            company_name=company_input if company_input else None,
            industry_name=selected_industry if selected_industry else None,
            keyword=keyword_input if keyword_input else None
        )
    
    @staticmethod
    def render_search_info(search_info: Dict[str, Any]) -> None:
        """Render search parameters information."""
        if not search_info:
            return
        
        with st.container(border=True):
            st.markdown("##### Search Parameters Used")
            
            if search_info.get("country_name"):
                code = search_info.get('country_code', 'Not Found')
                st.markdown(f"**Country:** `{search_info['country_name']}`   **→**   **Resolved Code:** `{code}`")
            
            if search_info.get("industry_name"):
                code = search_info.get('industry_code', 'Not Found')
                st.markdown(f"**Industry:** `{search_info['industry_name']}`   **→**   **Resolved Code:** `{code}`")
            
            if search_info.get("company_name"):
                UIComponents._render_company_info(search_info)
            
            if search_info.get("keyword"):
                st.markdown(f"**Keyword:** `{search_info['keyword']}`")
    
    @staticmethod
    def _render_company_info(search_info: Dict[str, Any]) -> None:
        """Render company information."""
        company_id = search_info.get('company_id', 'Not Found')
        matched_name = search_info.get('matched_company_name')
        
        company_line = f"**Company:** `{search_info['company_name']}`"
        if matched_name and matched_name.lower() != search_info['company_name'].lower():
            company_line += f"   **→**   **Resolved Name:** `{matched_name}`"
        company_line += f"   **→**   **Resolved ID:** `{company_id}`"
        
        st.markdown(company_line)
    
    @classmethod
    def render_sentiment_charts(cls, df: pd.DataFrame) -> None:
        """Render sentiment analysis charts."""
        st.header("Overall Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cls._render_sentiment_distribution(df)
        
        with col2:
            cls._render_sentiment_timeline(df)
    
    @classmethod
    def _render_sentiment_distribution(cls, df: pd.DataFrame) -> None:
        """Render sentiment distribution chart."""
        st.subheader("Document Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        
        fig_bar = px.bar(
            sentiment_counts,
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={'x': 'Sentiment', 'y': 'Number of Documents'},
            title="Total Count by Sentiment",
            color=sentiment_counts.index,
            color_discrete_map=cls.SENTIMENT_COLORS
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    @classmethod
    def _render_sentiment_timeline(cls, df: pd.DataFrame) -> None:
        """Render sentiment timeline chart."""
        st.subheader("Sentiment Over Time")
        time_series_data = (
            df.groupby(['date', 'sentiment'])
            .size()
            .reset_index(name='count')
            .sort_values('date')
        )
        
        fig_line = px.line(
            time_series_data,
            x='date',
            y='count',
            color='sentiment',
            labels={'date': 'Date', 'count': 'Number of Documents', 'sentiment': 'Sentiment'},
            title="Daily Sentiment Trend",
            markers=True,
            color_discrete_map=cls.SENTIMENT_COLORS
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    @staticmethod
    def render_sentiment_result(analysis: SentimentAnalysis) -> None:
        """Render sentiment analysis result."""
        color = UIComponents.SENTIMENT_COLORS.get(analysis.sentiment, "grey")
        
        st.markdown(
            f"""
            <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-top: 5px;'>
                <p style='margin-bottom: 5px;'>
                    <strong>Sentiment: <span style="color:{color}; font-weight:bold;">{analysis.sentiment}</span></strong>
                </p>
                <p style='margin-bottom: 0;'>
                    <strong>Analysis:</strong> {analysis.analysis_text}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


class DocumentFilter:
    """Handles document filtering operations."""
    
    @staticmethod
    def extract_unique_dates(documents: List[Dict[str, Any]]) -> List[str]:
        """Extract unique dates from documents."""
        dates = set()
        
        for doc in documents:
            doc_date = doc.get("date", "")
            if doc_date:
                try:
                    parsed_date = pd.to_datetime(doc_date).date()
                    dates.add(str(parsed_date))
                except Exception:
                    dates.add(doc_date)
        
        return sorted(list(dates))
    
    @staticmethod
    def filter_documents_by_date(documents: List[Dict[str, Any]], target_date: str) -> List[Dict[str, Any]]:
        """Filter documents by specific date."""
        filtered_docs = []
        
        for doc in documents:
            doc_date = doc.get("date", "")
            if doc_date:
                try:
                    parsed_doc_date = pd.to_datetime(doc_date).date()
                    if str(parsed_doc_date) == target_date:
                        filtered_docs.append(doc)
                except Exception:
                    if doc_date == target_date:
                        filtered_docs.append(doc)
        
        return filtered_docs


class EmisDocumentApp:
    """Main application class."""
    
    def __init__(self):
        self.session_manager = SessionStateManager()
        self.ui_components = UIComponents()
        self.document_analyzer = DocumentAnalyzer()
        self.document_filter = DocumentFilter()
        self.emis_client = None # Will be set by the cached getter
        
        # Initialize configuration
        self.session_manager.initialize()

    @st.cache_resource(ttl=3600) # Cache the EMIS client for 1 hour. Adjust TTL based on token expiry
    def _get_emis_client(_self, api_token: str): # _self para evitar el hashing de la instancia
        """Initializes and caches the EmisDocuments client."""
        # Import here to avoid potential circular dependencies and ensure lazy loading
        from emis_api import EmisDocuments
        from config import Config
        
        # Ya no es necesario establecer el token en Config aquí, ya que el token se pasa directamente a EmisDocuments.
        # Además, Config es volátil en re-ejecuciones de Streamlit.
        
        try:
            client = EmisDocuments(api_key=api_token) # Pasar el token directamente
            return client
        except Exception as e:
            logger.error(f"Failed to initialize EMIS client with provided token: {e}")
            raise # Re-raise to be caught by _handle_login
    
    def run(self) -> None:
        """Run the main application."""
        if not st.session_state.logged_in:
            self._render_login_page()
        else:
            self._render_main_app()
    
    def _render_login_page(self) -> None:
        """Render login page."""
        st.title("Login to EMIS Document Analyzer")
        self.ui_components.render_logo()
        
        with st.form("login_form"):
            emis_token = st.text_input("EMIS API Token", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                self._handle_login(emis_token)
    
    def _handle_login(self, emis_token: str) -> None:
        """Handle login process."""
        if not emis_token:
            st.error("Please enter the EMIS API Token.")
            return
        
        try:
            # Guardar el token en session_state para que persista
            st.session_state.emis_api_token = emis_token 
            
            # Usar el getter del cliente cacheado
            self.emis_client = self._get_emis_client(emis_token)
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        except Exception as e:
            st.error(f"Login failed. Could not initialize services: {e}")
            logger.error(f"Login failed: {e}")
            # Limpiar el token de sesión si el login falla, para no quedar en un estado inconsistente
            st.session_state.emis_api_token = None
            st.session_state.logged_in = False
    
    def _render_main_app(self) -> None:
        """Render main application interface."""
        self.ui_components.load_css()
        self.ui_components.render_logo()
        
        # Ensure client is initialized (will try to fetch from cache or re-initialize)
        if not self._ensure_client_initialized():
            return
        
        # Render search interface
        search_params = self.ui_components.render_search_filters()
        
        if st.sidebar.button("Search EMIS Documents"):
            self._handle_search(search_params)
        
        # Render results
        self._render_results()
        
        # Render logout button
        self._render_logout_button()
    
    def _ensure_client_initialized(self) -> bool:
        """Ensure EMIS client is initialized."""
        if self.emis_client is None:
            # Intentar obtener el token de session_state, donde debe persistir
            current_token = st.session_state.get("emis_api_token") 
            
            if st.session_state.logged_in and current_token:
                try:
                    self.emis_client = self._get_emis_client(current_token)
                    return True
                except Exception as e:
                    st.error(f"Failed to re-initialize EMIS client with cached token: {e}. Please try logging out and in again.")
                    logger.error(f"Client re-initialization failed: {e}")
                    self.session_manager.logout() # Force logout if client cannot be re-initialized
                    st.rerun()
                    return False
            else:
                # Esto ocurre si st.session_state.logged_in es True pero emis_api_token no está o es None.
                # También cubre el caso de que st.session_state.logged_in sea False (aunque la primera condición `if self.emis_client is None` ya lo habría atrapado).
                logger.warning("Inconsistent session state: logged_in but no valid token or client. Forcing logout.")
                self.session_manager.logout()
                st.rerun()
                return False
        return True
    
    def _handle_search(self, search_params: SearchParameters) -> None:
        """Handle search operation."""
        if not self._validate_search_params(search_params):
            st.warning("Please select a Country, enter a Company, select an Industry, or enter a Keyword to start a search.")
            return
        
        self.session_manager.reset_search_data()
        spinner_text = self._generate_spinner_text(search_params)
        
        with st.spinner(spinner_text):
            try:
                result = self._perform_search(search_params)
                self._store_search_results(result, search_params)
            except ValueError as e:
                st.error(f"API Configuration or Input Error: {e}")
                logger.error(f"Search API error: {e}")
            except Exception as e:
                st.error(f"Error during EMIS search: {e}")
                logger.error(f"Search error: {e}")
    
    def _validate_search_params(self, search_params: SearchParameters) -> bool:
        """Validate search parameters."""
        return any([
            search_params.country_name,
            search_params.company_name,
            search_params.industry_name,
            search_params.keyword
        ])
    
    def _generate_spinner_text(self, search_params: SearchParameters) -> str:
        """Generate spinner text based on search parameters."""
        parts = []
        if search_params.company_name:
            parts.append(f"about '{search_params.company_name}'")
        if search_params.country_name:
            parts.append(f"in '{search_params.country_name}'")
        if search_params.industry_name:
            parts.append(f"in industry '{search_params.industry_name}'")
        if search_params.keyword:
            parts.append(f"with keyword '{search_params.keyword}'")
        
        return f"Searching for documents {' and '.join(parts)}..."
    
    def _perform_search(self, search_params: SearchParameters) -> Dict[str, Any]:
        """Perform the actual search operation."""
        return self.emis_client.run(
            country_name=search_params.country_name,
            company_name=search_params.company_name,
            keyword=search_params.keyword,
            industry_name=search_params.industry_name,
            limit=search_params.limit
        )
    
    def _store_search_results(self, result: Dict[str, Any], search_params: SearchParameters) -> None:
        """Store search results in session state."""
        st.session_state.result = result["documents"]
        st.session_state.search_info = {
            "country_name": search_params.country_name,
            "company_name": search_params.company_name,
            "keyword": search_params.keyword,
            "industry_name": search_params.industry_name,
            "country_code": result.get("country_code"),
            "company_id": result.get("company_id"),
            "matched_company_name": result.get("matched_company_name"),
            "industry_code": result.get("industry_code")
        }
    
    def _render_results(self) -> None:
        """Render search results."""
        if st.session_state.result is None:
            return
        
        documents = st.session_state.result
        
        # Siempre renderizamos la información de los parámetros de búsqueda
        # si la búsqueda fue intentada y la información está disponible.
        if st.session_state.search_info:
            self.ui_components.render_search_info(st.session_state.search_info)

        # Si la lista de documentos está vacía, mostramos un mensaje informativo
        if not documents:
            st.info("No documents found for the given criteria. Please try adjusting your search parameters.")
            return # Terminamos la función aquí, ya que no hay documentos que analizar o listar.

        # Si llegamos aquí, significa que se encontraron documentos.
        st.subheader(f"Search Results ({len(documents)} documents found)")
        
        # Renderizar la sección de análisis general y documentos individuales
        self._render_general_analysis_section(documents)
        self._render_individual_documents_section(documents)
    
    def _render_general_analysis_section(self, documents: List[Dict[str, Any]]) -> None:
        """Render general analysis section."""
        if st.button("Make General Analysis", type="primary"):
            with st.spinner("Analyzing all documents... This may take a moment."):
                df = self.document_analyzer.analyze_documents(documents)
                if not df.empty:
                    st.session_state.general_analysis_df = df
                else:
                    st.warning("No data available to perform analysis.")
        
        # Render charts if analysis exists
        if st.session_state.general_analysis_df is not None:
            self.ui_components.render_sentiment_charts(st.session_state.general_analysis_df)
    
    def _render_individual_documents_section(self, documents: List[Dict[str, Any]]) -> None:
        """Render individual documents section."""
        st.header("Individual Document Details")
        
        # Date filtering
        unique_dates = self.document_filter.extract_unique_dates(documents)
        filtered_docs = self._apply_date_filter(documents, unique_dates)
        
        # Render documents
        self._render_document_list(filtered_docs)
    
    def _apply_date_filter(self, documents: List[Dict[str, Any]], unique_dates: List[str]) -> List[Dict[str, Any]]:
        """Apply date filter to documents."""
        if not unique_dates:
            st.warning("No dates available for filtering. Showing all documents.")
            return documents
        
        date_options = ["All Dates"] + unique_dates
        selected_date = st.selectbox(
            "Filter by Date:",
            options=date_options,
            index=0,
            help="Select a specific date to filter documents, or choose 'All Dates' to show all documents"
        )
        
        if selected_date == "All Dates":
            filtered_docs = documents
        else:
            filtered_docs = self.document_filter.filter_documents_by_date(documents, selected_date)
        
        # Display filter info
        if selected_date != "All Dates":
            st.info(f"Showing {len(filtered_docs)} document(s) for date: {selected_date}")
        else:
            st.info(f"Showing all {len(filtered_docs)} documents")
        
        return filtered_docs
    
    def _render_document_list(self, documents: List[Dict[str, Any]]) -> None:
        """Render list of documents."""
        if not documents:
            st.info("No documents found for the selected date.")
            return
        
        for idx, doc in enumerate(documents):
            self._render_single_document(doc, idx)
            
            # Add separator between documents
            if idx < len(documents) - 1:
                st.markdown("""<hr style="border: none; height: 2px; background-color: #e0e0e0; margin: 20px 0;">""", unsafe_allow_html=True)
    
    def _render_single_document(self, doc: Dict[str, Any], idx: int) -> None:
        """Render a single document."""
        doc_id = doc.get("id", "N/A")
        title = doc.get("title", "No title")
        date = doc.get("date", "")
        
        raw_abstract = doc.get("abstract")
        abstract = raw_abstract.strip() if isinstance(raw_abstract, str) else ""
        
        with st.container(border=True):
            st.markdown(f"**Title:** {title}")
            st.caption(f"Document ID: {doc_id} | Date: {date}")
            
            with st.expander("Abstract"):
                st.write(abstract if abstract else "No abstract available.")
            
            if st.button("Analyze", key=f"analyse_{idx}_{doc_id}"):
                self._analyze_single_document(title, abstract)
    
    def _analyze_single_document(self, title: str, abstract: str) -> None:
        """Analyze a single document."""
        text_to_analyze = f"{title}. {abstract}"
        
        with st.spinner("Analyzing sentiment with NLTK..."):
            analysis_result = self.document_analyzer.sentiment_analyzer.analyze_text(text_to_analyze)
        
        self.ui_components.render_sentiment_result(analysis_result)
    
    def _render_logout_button(self) -> None:
        """Render logout button in sidebar."""
        with st.sidebar:
            if st.button("Logout"):
                self._handle_logout()
    
    def _handle_logout(self) -> None:
        """Handle logout process."""
        # Limpiar la entrada del caché del cliente específica para el token actual
        # Asegurarse de usar la misma clave que se usó para almacenar.
        current_token = st.session_state.get("emis_api_token")
        if current_token:
            self._get_emis_client.clear(api_token=current_token) 

        self.session_manager.logout() # Esto restablece todas las claves de session_state incluyendo logged_in y emis_api_token
        self.emis_client = None
        
        # El Config global no necesita ser limpiado aquí si ya no almacena el token de forma persistente.
        # try:
        #     from config import Config # Import here to ensure it's available
        #     Config._Config__conf["EMIS"]["token"] = ""
        # except Exception as e:
        #     logger.warning(f"Could not clear config on logout: {e}")
        
        st.rerun()


def main():
    """Main entry point of the application."""
    app = EmisDocumentApp()
    app.run()


if __name__ == "__main__":
    main()