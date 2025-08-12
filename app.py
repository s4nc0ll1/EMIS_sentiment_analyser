"""
EMIS Document Analyzer
A Streamlit application for analyzing EMIS documents with sentiment analysis capabilities.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="EMIS Document Analyzer", layout="wide", initial_sidebar_state="expanded"
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
    @st.cache_data(ttl=3600)  # Cache JSON file loading for 1 hour
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
    @st.cache_data(ttl=3600)  # Cache countries list for 1 hour
    def load_countries(cls) -> List[str]:
        """Load country names from countries.json."""
        file_path = cls.DATA_DIR / "countries.json"
        data = cls.load_json_file(file_path, "countries")

        if not data:
            return [""]

        try:
            countries = sorted(
                [
                    item["name"]
                    for item in data.get("data", {}).get("items", [])
                    if "name" in item
                ]
            )
            return [""] + countries
        except Exception as e:
            st.error(f"Error processing countries data: {e}")
            logger.error(f"Error processing countries data: {e}")
            return [""]

    @classmethod
    @st.cache_data(ttl=3600)  # Cache industries list for 1 hour
    def load_industries(cls) -> List[str]:
        """Load industry names from industries.json."""
        file_path = cls.DATA_DIR / "industries.json"
        data = cls.load_json_file(file_path, "industries")

        if not data:
            return [""]

        try:
            industries = sorted([item["name"] for item in data if "name" in item])
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
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            test_analyzer = SentimentIntensityAnalyzer()
            test_analyzer.polarity_scores("test")
            logger.info("NLTK vader_lexicon found and working.")
        except (LookupError, FileNotFoundError, OSError) as e:
            logger.info(f"NLTK vader_lexicon not found ({e}), downloading...")
            try:
                import nltk

                nltk.download("vader_lexicon", quiet=True)
                logger.info("NLTK vader_lexicon downloaded successfully.")
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
        if not text or not text.strip():
            return SentimentAnalysis(
                sentiment="Unknown", score=0.0, analysis_text="No text to analyze."
            )

        scores = self.analyzer.polarity_scores(text)
        compound_score = scores["compound"]
        sentiment = self._determine_sentiment(compound_score)
        analysis_text = self._format_analysis_text(sentiment, compound_score, scores)

        return SentimentAnalysis(
            sentiment=sentiment, score=compound_score, analysis_text=analysis_text
        )

    def _determine_sentiment(self, compound_score: float) -> str:
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def _format_analysis_text(
        self, sentiment: str, compound_score: float, scores: Dict[str, float]
    ) -> str:
        return (
            f"The overall sentiment is **{sentiment.lower()}** "
            f"with a compound score of **{compound_score:.2f}**. "
            f"Breakdown: (Positive: {scores['pos']:.2f}, "
            f"Negative: {scores['neg']:.2f}, Neutral: {scores['neu']:.2f})"
        )


class DocumentAnalyzer:
    """Handles document analysis operations."""

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def analyze_documents(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        analyses = []
        for doc in documents:
            text_to_analyze = f"{doc.get('title', '')}. {doc.get('abstract', '')}"
            analysis_result = self.sentiment_analyzer.analyze_text(text_to_analyze)
            analyses.append(
                {"date": doc.get("date"), "sentiment": analysis_result.sentiment}
            )

        if analyses:
            df = pd.DataFrame(analyses)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        return pd.DataFrame()


class SessionStateManager:
    """Manages Streamlit session state."""

    SESSION_KEYS = [
        "logged_in",
        "result",
        "general_analysis_df",
        "search_info",
        "emis_api_token",
    ]

    @staticmethod
    def initialize() -> None:
        defaults = {
            "logged_in": False,
            "result": None,
            "general_analysis_df": None,
            "search_info": None,
            "emis_api_token": None,
        }
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def reset_search_data() -> None:
        for key in ["result", "general_analysis_df", "search_info"]:
            st.session_state[key] = None

    @staticmethod
    def logout() -> None:
        for key in SessionStateManager.SESSION_KEYS:
            st.session_state[key] = None if key != "logged_in" else False
        st.session_state["logged_in"] = False


class UIComponents:
    """Handles UI component rendering."""

    SENTIMENT_COLORS = {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "orange",
        "Unknown": "grey",
    }

    @staticmethod
    def load_css() -> None:
        css_path = Path("static/styles.css")
        try:
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"{css_path} not found. Skipping custom CSS.")

    @staticmethod
    def render_logo() -> None:
        logo_path = "static/logo2.png"
        try:
            st.logo(logo_path, size="large")
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")

    @staticmethod
    def render_search_filters() -> SearchParameters:
        st.sidebar.header("Search Filters")
        country_list = DataLoader.load_countries()
        industry_list = DataLoader.load_industries()

        selected_country = st.sidebar.selectbox(
            "Country Name (Optional)", options=country_list, index=0
        )
        company_input = st.sidebar.text_input("Company Name (Optional)")
        selected_industry = st.sidebar.selectbox(
            "Industry (Optional)", options=industry_list, index=0
        )
        keyword_input = st.sidebar.text_input("Keyword (Optional)")

        return SearchParameters(
            country_name=selected_country or None,
            company_name=company_input or None,
            industry_name=selected_industry or None,
            keyword=keyword_input or None,
        )

    @staticmethod
    def render_search_info(search_info: Dict[str, Any]) -> None:
        if not search_info:
            return
        with st.container(border=True):
            st.markdown("##### Search Parameters Used")
            if search_info.get("country_name"):
                st.markdown(
                    f"**Country:** `{search_info['country_name']}`   **→**   **Resolved Code:** `{search_info.get('country_code', 'Not Found')}`"
                )
            if search_info.get("industry_name"):
                st.markdown(
                    f"**Industry:** `{search_info['industry_name']}`   **→**   **Resolved Code:** `{search_info.get('industry_code', 'Not Found')}`"
                )
            if search_info.get("company_name"):
                UIComponents._render_company_info(search_info)
            if search_info.get("keyword"):
                st.markdown(f"**Keyword:** `{search_info['keyword']}`")

    @staticmethod
    def _render_company_info(search_info: Dict[str, Any]) -> None:
        company_id = search_info.get("company_id", "Not Found")
        matched_name = search_info.get("matched_company_name")
        company_line = f"**Company:** `{search_info['company_name']}`"
        if matched_name and matched_name.lower() != search_info["company_name"].lower():
            company_line += f"   **→**   **Resolved Name:** `{matched_name}`"
        company_line += f"   **→**   **Resolved ID:** `{company_id}`"
        st.markdown(company_line)

    @classmethod
    def render_sentiment_charts(cls, df: pd.DataFrame) -> None:
        st.subheader("Overall Sentiment Analysis")
        col1, col2 = st.columns(2)
        with col1:
            cls._render_sentiment_distribution(df)
        with col2:
            cls._render_sentiment_timeline(df)

    @classmethod
    def _render_sentiment_distribution(cls, df: pd.DataFrame) -> None:
        sentiment_counts = df["sentiment"].value_counts()
        fig_bar = px.bar(
            sentiment_counts,
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={"x": "Sentiment", "y": "Number of Documents"},
            title="Total Count by Sentiment",
            color=sentiment_counts.index,
            color_discrete_map=cls.SENTIMENT_COLORS,
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    @classmethod
    def _render_sentiment_timeline(cls, df: pd.DataFrame) -> None:
        time_series_data = (
            df.groupby(["date", "sentiment"])
            .size()
            .reset_index(name="count")
            .sort_values("date")
        )
        fig_line = px.line(
            time_series_data,
            x="date",
            y="count",
            color="sentiment",
            labels={
                "date": "Date",
                "count": "Number of Documents",
                "sentiment": "Sentiment",
            },
            title="Daily Sentiment Trend",
            markers=True,
            color_discrete_map=cls.SENTIMENT_COLORS,
        )
        st.plotly_chart(fig_line, use_container_width=True)

    @staticmethod
    def render_sentiment_result(analysis: SentimentAnalysis) -> None:
        color = UIComponents.SENTIMENT_COLORS.get(analysis.sentiment, "grey")
        st.markdown(
            f"""
            <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-top: 5px;'>
                <p style='margin-bottom: 5px;'><strong>Sentiment: <span style="color:{color}; font-weight:bold;">{analysis.sentiment}</span></strong></p>
                <p style='margin-bottom: 0;'><strong>Analysis:</strong> {analysis.analysis_text}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # --- NUEVA FUNCIÓN ---
    @classmethod
    def render_topics_industries_charts(cls, documents: List[Dict[str, Any]]) -> None:
        """Render charts for top topics and industries."""
        st.subheader("Content Analysis: Topics & Industries")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Top Topics")
            topic_df = DocumentFilter.extract_and_count_tags(documents, "topics")
            if not topic_df.empty:
                top_n = 15
                fig = px.bar(
                    topic_df.head(top_n).sort_values("count", ascending=True),
                    x="count",
                    y="name",
                    orientation="h",
                    labels={"name": "Topic", "count": "Doc Count"},
                    title=f"Top {min(top_n, len(topic_df))} Topics",
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}, title_x=0.5, height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No topics found in the search results.")
        with col2:
            st.markdown("##### Top Industries")
            industry_df = DocumentFilter.extract_and_count_tags(documents, "industries")
            if not industry_df.empty:
                top_n = 15
                fig = px.bar(
                    industry_df.head(top_n).sort_values("count", ascending=True),
                    x="count",
                    y="name",
                    orientation="h",
                    labels={"name": "Industry", "count": "Doc Count"},
                    title=f"Top {min(top_n, len(industry_df))} Industries",
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}, title_x=0.5, height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No industries found in the search results.")


class DocumentFilter:
    """Handles document filtering operations."""

    @staticmethod
    def extract_unique_dates(documents: List[Dict[str, Any]]) -> List[str]:
        dates = set()
        for doc in documents:
            if doc_date := doc.get("date"):
                try:
                    dates.add(str(pd.to_datetime(doc_date).date()))
                except Exception:
                    dates.add(doc_date)
        return sorted(list(dates))

    @staticmethod
    def filter_documents_by_date(
        documents: List[Dict[str, Any]], target_date: str
    ) -> List[Dict[str, Any]]:
        if not target_date or target_date == "All Dates":
            return documents
        filtered_docs = []
        for doc in documents:
            if doc_date := doc.get("date"):
                try:
                    parsed_doc_date = str(pd.to_datetime(doc_date).date())
                except Exception:
                    parsed_doc_date = doc_date
                if parsed_doc_date == target_date:
                    filtered_docs.append(doc)
        return filtered_docs

    # --- NUEVA FUNCIÓN ---
    @staticmethod
    def extract_and_count_tags(
        documents: List[Dict[str, Any]], tag_key: str
    ) -> pd.DataFrame:
        """Extract and count tags (topics or industries) from documents."""
        tag_counts = {}
        for doc in documents:
            for tag in doc.get(tag_key, []):
                if name := tag.get("name"):
                    tag_counts[name] = tag_counts.get(name, 0) + 1
        if not tag_counts:
            return pd.DataFrame()
        df = pd.DataFrame(list(tag_counts.items()), columns=["name", "count"])
        return df.sort_values("count", ascending=False).reset_index(drop=True)

    # --- NUEVA FUNCIÓN ---
    @staticmethod
    def filter_documents_by_tag(
        documents: List[Dict[str, Any]], selected_tag: str, tag_key: str
    ) -> List[Dict[str, Any]]:
        """Filter documents that contain a specific tag."""
        capitalized_key = tag_key.capitalize()
        if not selected_tag or selected_tag.startswith("All "):
            return documents
        filtered_docs = []
        for doc in documents:
            for tag in doc.get(tag_key, []):
                if tag.get("name") == selected_tag:
                    filtered_docs.append(doc)
                    break
        return filtered_docs


class EmisDocumentApp:
    """Main application class."""

    def __init__(self):
        self.session_manager = SessionStateManager()
        self.ui_components = UIComponents()
        self.document_analyzer = DocumentAnalyzer()
        self.document_filter = DocumentFilter()
        self.emis_client = None
        self.session_manager.initialize()

    @st.cache_resource(ttl=3600)
    def _get_emis_client(_self, api_token: str):
        from emis_api import EmisDocuments

        try:
            return EmisDocuments(api_key=api_token)
        except Exception as e:
            logger.error(f"Failed to initialize EMIS client: {e}")
            raise

    def run(self) -> None:
        if not st.session_state.logged_in:
            self._render_login_page()
        else:
            self._render_main_app()

    def _render_login_page(self) -> None:
        st.title("Login to EMIS Document Analyzer")
        self.ui_components.render_logo()
        with st.form("login_form"):
            emis_token = st.text_input("EMIS API Token", type="password")
            if st.form_submit_button("Login"):
                self._handle_login(emis_token)

    def _handle_login(self, emis_token: str) -> None:
        if not emis_token:
            st.error("Please enter the EMIS API Token.")
            return
        try:
            st.session_state.emis_api_token = emis_token
            self.emis_client = self._get_emis_client(emis_token)
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
            st.session_state.emis_api_token = None
            st.session_state.logged_in = False

    def _render_main_app(self) -> None:
        self.ui_components.load_css()
        self.ui_components.render_logo()
        if not self._ensure_client_initialized():
            return

        search_params = self.ui_components.render_search_filters()
        if st.sidebar.button("Search EMIS Documents", type="primary"):
            self._handle_search(search_params)

        self._render_results()
        self._render_logout_button()

    def _ensure_client_initialized(self) -> bool:
        if self.emis_client is None:
            current_token = st.session_state.get("emis_api_token")
            if st.session_state.logged_in and current_token:
                try:
                    self.emis_client = self._get_emis_client(current_token)
                    return True
                except Exception as e:
                    st.error(
                        f"Client re-initialization failed: {e}. Please log out and in again."
                    )
                    self.session_manager.logout()
                    st.rerun()
                    return False
            else:
                self.session_manager.logout()
                st.rerun()
                return False
        return True

    def _handle_search(self, search_params: SearchParameters) -> None:
        if not any(
            [
                search_params.country_name,
                search_params.company_name,
                search_params.industry_name,
                search_params.keyword,
            ]
        ):
            st.warning("Please provide at least one search criterion.")
            return

        self.session_manager.reset_search_data()
        parts = [
            f"about '{p}'"
            for p in [
                search_params.company_name,
                search_params.country_name,
                search_params.industry_name,
                search_params.keyword,
            ]
            if p
        ]
        spinner_text = f"Searching for documents {' and '.join(parts)}..."

        with st.spinner(spinner_text):
            try:
                result = self.emis_client.run(**vars(search_params))
                self._store_search_results(result, search_params)
            except Exception as e:
                st.error(f"Error during EMIS search: {e}")
                logger.error(f"Search error: {e}")

    def _store_search_results(
        self, result: Dict[str, Any], search_params: SearchParameters
    ) -> None:
        st.session_state.result = result["documents"]
        st.session_state.search_info = {
            **vars(search_params),
            **{
                k: result.get(k)
                for k in [
                    "country_code",
                    "company_id",
                    "matched_company_name",
                    "industry_code",
                ]
            },
        }

    # --- MÉTODO MODIFICADO ---
    def _render_results(self) -> None:
        """Render search results, including analysis and filtered document list."""
        if st.session_state.result is None:
            return

        documents = st.session_state.result
        if st.session_state.search_info:
            self.ui_components.render_search_info(st.session_state.search_info)

        if not documents:
            st.info("No documents found for the given criteria.")
            return

        st.subheader(f"Search Results ({len(documents)} documents found)")

        # --- NUEVA ESTRUCTURA ---
        with st.container(border=True):
            st.header("Results Analysis")
            self._render_general_analysis_section(documents)
            st.markdown("---")
            self.ui_components.render_topics_industries_charts(documents)

        self._render_individual_documents_section(documents)

    def _render_general_analysis_section(self, documents: List[Dict[str, Any]]) -> None:
        if st.button("Make General Sentiment Analysis"):
            with st.spinner("Analyzing all documents..."):
                df = self.document_analyzer.analyze_documents(documents)
                st.session_state.general_analysis_df = df if not df.empty else None

        if st.session_state.general_analysis_df is not None:
            self.ui_components.render_sentiment_charts(
                st.session_state.general_analysis_df
            )

    # --- MÉTODO MODIFICADO ---
    def _render_individual_documents_section(
        self, documents: List[Dict[str, Any]]
    ) -> None:
        """Render individual documents section with interactive filters."""
        st.header("Document Details & Filters")

        filtered_docs = self._apply_document_filters(documents)

        st.info(
            f"Showing {len(filtered_docs)} of {len(documents)} documents based on active filters."
        )
        self._render_document_list(filtered_docs)

    def _apply_document_filters(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply all available filters and return the filtered list."""
        filtered_docs = documents

        topic_df = self.document_filter.extract_and_count_tags(documents, "topics")
        industry_df = self.document_filter.extract_and_count_tags(
            documents, "industries"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            if not topic_df.empty:
                options = ["All Topics"] + topic_df["name"].tolist()
                selected = st.selectbox("Filter by Topic:", options=options)
                filtered_docs = self.document_filter.filter_documents_by_tag(
                    filtered_docs, selected, "topics"
                )
        with col2:
            if not industry_df.empty:
                options = ["All Industries"] + industry_df["name"].tolist()
                selected = st.selectbox("Filter by Industry:", options=options)
                filtered_docs = self.document_filter.filter_documents_by_tag(
                    filtered_docs, selected, "industries"
                )
        with col3:
            # Dates are filtered based on the already-filtered documents for better context
            unique_dates = self.document_filter.extract_unique_dates(filtered_docs)
            if unique_dates:
                options = ["All Dates"] + unique_dates
                selected = st.selectbox("Filter by Date:", options=options)
                filtered_docs = self.document_filter.filter_documents_by_date(
                    filtered_docs, selected
                )

        return filtered_docs

    def _render_document_list(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            st.warning("No documents match the current filter criteria.")
            return

        for idx, doc in enumerate(documents):
            self._render_single_document(doc, idx)
            if idx < len(documents) - 1:
                st.markdown(
                    """<hr style="height: 2px; background-color: #e0e0e0;">""",
                    unsafe_allow_html=True,
                )

    def _render_single_document(self, doc: Dict[str, Any], idx: int) -> None:
        doc_id, title, date, abstract = (
            doc.get("id"),
            doc.get("title"),
            doc.get("date"),
            doc.get("abstract", ""),
        )
        with st.container(border=True):
            st.markdown(f"**Title:** {title}")
            st.caption(f"Document ID: {doc_id} | Date: {date}")
            with st.expander("Abstract"):
                st.write(abstract or "No abstract available.")
            if st.button("Analyze Sentiment", key=f"analyse_{idx}_{doc_id}"):
                text_to_analyze = f"{title}. {abstract}"
                with st.spinner("Analyzing..."):
                    analysis_result = (
                        self.document_analyzer.sentiment_analyzer.analyze_text(
                            text_to_analyze
                        )
                    )
                self.ui_components.render_sentiment_result(analysis_result)

    def _render_logout_button(self) -> None:
        with st.sidebar:
            if st.button("Logout"):
                self._handle_logout()

    def _handle_logout(self) -> None:
        if current_token := st.session_state.get("emis_api_token"):
            self._get_emis_client.clear(api_token=current_token)
        self.session_manager.logout()
        self.emis_client = None
        st.rerun()


if __name__ == "__main__":
    EmisDocumentApp().run()
