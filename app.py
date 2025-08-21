"""
Streamlit application for visualizing CEIC time series data with sentiment analysis.

This module provides a web interface for loading, visualizing, and analyzing
CEIC economic data series along with related news sentiment analysis.
"""

import os
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from streamlit_plotly_events import plotly_events
from ceic_api_client.pyceic import Ceic

from series import SeriesVisualizer

# Load environment variables
load_dotenv()

# Configuration constants
class Config:
    """Application configuration constants."""
    LOGO_PATH = "./static/logo2.png"
    LOGO_WIDTH = 250
    SIDEBAR_LOGO_WIDTH = 200
    EMIS_API_URL = "https://api.emis.com/v2/company/documents"
    MAX_DOCUMENTS_PAGES = 10
    DOCUMENTS_PER_PAGE = 100
    MAX_TITLE_LENGTH = 50


@dataclass
class SessionStateKeys:
    """Session state keys to avoid magic strings."""
    LOGGED_IN = 'logged_in'
    CEIC_CLIENT = 'ceic_client'
    VISUALIZER_OBJECT = 'visualizer_object'
    SENTIMENT_DF = 'sentiment_df'
    RAW_DOCUMENTS = 'raw_documents'
    SELECTED_SENTIMENT_DATE = 'selected_sentiment_date'


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class DataFetchError(Exception):
    """Raised when data fetching operations fail."""
    pass


class SentimentAnalysisError(Exception):
    """Raised when sentiment analysis fails."""
    pass


class AppInitializer:
    """Handles application initialization and setup."""

    @staticmethod
    def setup_page_config() -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Visor de Series CEIC",
            layout="wide"
        )

    @staticmethod
    def initialize_session_state() -> None:
        """Initialize session state with default values."""
        default_values = {
            SessionStateKeys.LOGGED_IN: False,
            SessionStateKeys.CEIC_CLIENT: None,
            SessionStateKeys.VISUALIZER_OBJECT: None,
            SessionStateKeys.SENTIMENT_DF: None,
            SessionStateKeys.RAW_DOCUMENTS: [],
            SessionStateKeys.SELECTED_SENTIMENT_DATE: None
        }
        
        for key, default_value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    @st.cache_resource
    def download_nltk_resources() -> None:
        """Download required NLTK resources."""
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            st.info("Descargando recursos para análisis de sentimientos (VADER)...")
            nltk.download('vader_lexicon')


class AuthenticationManager:
    """Handles user authentication operations."""

    @staticmethod
    def render_login_page() -> None:
        """Render the login page interface."""
        st.image(Config.LOGO_PATH, width=Config.LOGO_WIDTH)
        st.title("Visor de Series Point-in-Time")
        
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Iniciar Sesión")
            
            if submitted:
                AuthenticationManager._handle_login_submission(username, password)

    @staticmethod
    def _handle_login_submission(username: str, password: str) -> None:
        """Handle login form submission."""
        if not username or not password:
            st.warning("Por favor, introduce tu usuario y contraseña.")
            return
        
        with st.spinner("Iniciando sesión..."):
            try:
                client = Ceic.login(username, password)
                st.session_state[SessionStateKeys.CEIC_CLIENT] = client
                st.session_state[SessionStateKeys.LOGGED_IN] = True
                st.success("¡Inicio de sesión exitoso!")
                st.rerun()
            except Exception as e:
                st.error(f"Error en el inicio de sesión: {e}")
                AuthenticationManager._reset_authentication_state()

    @staticmethod
    def _reset_authentication_state() -> None:
        """Reset authentication-related session state."""
        st.session_state[SessionStateKeys.LOGGED_IN] = False
        st.session_state[SessionStateKeys.CEIC_CLIENT] = None


class SeriesDataManager:
    """Manages series data loading and processing."""

    @staticmethod
    def load_series(series_id: str) -> None:
        """Load series data and metadata."""
        with st.spinner(f"Cargando datos para la serie ID {series_id}..."):
            try:
                ceic = st.session_state[SessionStateKeys.CEIC_CLIENT]
                visualizer = SeriesVisualizer(ceic, series_id)
                visualizer.fetch_all_data()
                
                if visualizer.metadata and visualizer.series_data:
                    st.session_state[SessionStateKeys.VISUALIZER_OBJECT] = visualizer
                    st.success(f"Datos cargados para la serie: '{visualizer.metadata.name}'")
                else:
                    SeriesDataManager._handle_load_failure(series_id)
            except Exception as e:
                st.error(f"Ocurrió un error al cargar la serie: {e}")
                st.session_state[SessionStateKeys.VISUALIZER_OBJECT] = None

    @staticmethod
    def _handle_load_failure(series_id: str) -> None:
        """Handle series loading failure."""
        st.session_state[SessionStateKeys.VISUALIZER_OBJECT] = None
        st.error(f"No se pudieron cargar los datos para la serie ID {series_id}. "
                f"Verifica el ID.")

    @staticmethod
    def clear_analysis_data() -> None:
        """Clear sentiment analysis related data."""
        st.session_state[SessionStateKeys.SENTIMENT_DF] = None
        st.session_state[SessionStateKeys.RAW_DOCUMENTS] = []
        st.session_state[SessionStateKeys.SELECTED_SENTIMENT_DATE] = None


class DocumentFetcher:
    """Handles document fetching from EMIS API."""

    def __init__(self):
        self.api_token = os.getenv("EMIS_DOCUMENTS_API_KEY")

    def fetch_documents(self, country_code: str, series_name: str, 
                       start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Fetch documents from EMIS API with pagination.
        
        Args:
            country_code: Country identifier
            series_name: Name of the series to search for
            start_date: Start date for document search
            end_date: End date for document search
            
        Returns:
            List of document dictionaries
            
        Raises:
            DataFetchError: If API request fails
        """
        if not self.api_token:
            raise DataFetchError("EMIS API token not found")

        all_documents = []
        
        for page in range(Config.MAX_DOCUMENTS_PAGES):
            offset = page * Config.DOCUMENTS_PER_PAGE
            
            params = {
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "keyword": series_name,
                "country": country_code,
                "order": "date",
                "limit": Config.DOCUMENTS_PER_PAGE,
                "offset": offset,
                "token": self.api_token
            }
            
            try:
                response = requests.get(Config.EMIS_API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                documents_page = data.get('data', {}).get('items', [])
                if not documents_page:
                    break
                    
                all_documents.extend(documents_page)
                
            except requests.exceptions.RequestException as e:
                raise DataFetchError(f"API request failed: {e}")

        return all_documents


class SentimentAnalyzer:
    """Handles sentiment analysis operations."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_documents(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze sentiment of documents and return aggregated results.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            DataFrame with daily sentiment scores
            
        Raises:
            SentimentAnalysisError: If analysis fails
        """
        if not documents:
            return pd.DataFrame()

        sentiment_results = []
        
        for doc in documents:
            text = self._extract_text_from_document(doc)
            if not text:
                continue
                
            score = self._calculate_sentiment_score(text)
            if self._is_valid_sentiment_score(score):
                sentiment_results.append({
                    "date": doc.get("creationDate"),
                    "sentiment_score": score,
                    "title": doc.get('title', '')[:Config.MAX_TITLE_LENGTH] + "..."
                })

        if not sentiment_results:
            return pd.DataFrame()

        return self._aggregate_sentiment_by_date(sentiment_results)

    def _extract_text_from_document(self, doc: Dict[str, Any]) -> str:
        """Extract meaningful text from document."""
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        text = f"{title}. {abstract}".strip()
        return text if text != '.' else ''

    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for text."""
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

    def _is_valid_sentiment_score(self, score: float) -> bool:
        """Validate sentiment score is within expected range."""
        return -1 <= score <= 1

    def _aggregate_sentiment_by_date(self, sentiment_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Aggregate sentiment scores by date."""
        df = pd.DataFrame(sentiment_results)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        df_grouped = df.groupby('date').agg({
            'sentiment_score': 'mean',
            'title': 'count'
        }).reset_index()
        
        df_grouped.rename(columns={'title': 'doc_count'}, inplace=True)
        
        # Validate final results
        if (df_grouped['sentiment_score'].max() > 1 or 
            df_grouped['sentiment_score'].min() < -1):
            raise SentimentAnalysisError("Sentiment scores outside valid range")
        
        return df_grouped


class ContextAnalyzer:
    """Handles context analysis operations."""

    def __init__(self):
        self.document_fetcher = DocumentFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()

    def perform_analysis(self, visualizer: SeriesVisualizer, 
                        start_date: datetime, end_date: datetime) -> None:
        """
        Perform complete context analysis including document fetching and sentiment analysis.
        
        Args:
            visualizer: SeriesVisualizer instance
            start_date: Analysis start date
            end_date: Analysis end date
        """
        SeriesDataManager.clear_analysis_data()
        
        with st.spinner("Iniciando búsqueda de documentos..."):
            try:
                # Extract metadata
                meta = visualizer.metadata
                country_code = self._extract_country_code(meta)
                series_name = self._extract_series_name(meta)
                
                if not all([country_code, series_name]):
                    st.error("Faltan datos (país, nombre de serie) para la búsqueda.")
                    return

                # Fetch documents
                documents = self._fetch_documents_with_progress(
                    country_code, series_name, start_date, end_date
                )
                
                if not documents:
                    st.warning(f"No se encontraron documentos para '{series_name}' "
                              f"en {country_code} en las fechas seleccionadas.")
                    return

                # Analyze sentiment
                sentiment_df = self._analyze_sentiment_with_progress(documents)
                
                # Store results
                st.session_state[SessionStateKeys.RAW_DOCUMENTS] = documents
                st.session_state[SessionStateKeys.SENTIMENT_DF] = sentiment_df
                
                st.success(f"Análisis completado. Se procesaron {len(documents)} "
                          f"documentos en {len(sentiment_df)} días.")
                
            except (DataFetchError, SentimentAnalysisError) as e:
                st.error(f"Error durante el análisis: {e}")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado durante el análisis: {e}")
                st.error(traceback.format_exc())

    def _extract_country_code(self, meta) -> Optional[str]:
        """Extract country code from metadata."""
        return getattr(meta.country, 'id', None) if hasattr(meta, 'country') else None

    def _extract_series_name(self, meta) -> Optional[str]:
        """Extract series name from metadata."""
        return getattr(meta, 'name', None)

    def _fetch_documents_with_progress(self, country_code: str, series_name: str,
                                     start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Fetch documents with progress indication."""
        spinner = st.spinner("Buscando documentos...")
        with spinner:
            documents = self.document_fetcher.fetch_documents(
                country_code, series_name, start_date, end_date
            )
            spinner.text = f"Encontrados {len(documents)} documentos"
            return documents

    def _analyze_sentiment_with_progress(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze sentiment with progress indication."""
        with st.spinner(f"Analizando {len(documents)} documentos..."):
            return self.sentiment_analyzer.analyze_documents(documents)


class ChartRenderer:
    """Handles chart rendering operations."""

    @staticmethod
    def render_sentiment_chart(df: pd.DataFrame) -> None:
        """Render sentiment analysis chart with click events."""
        st.subheader("Análisis de Sentimiento de Noticias")
        st.info("Haz clic en un punto del gráfico para ver los documentos de esa fecha.")

        if df.empty:
            st.info("El análisis de contexto se ejecutó, pero no hay datos para mostrar.")
            return

        ChartRenderer._validate_sentiment_dataframe(df)
        ChartRenderer._display_sentiment_metrics(df)
        ChartRenderer._show_sentiment_chart_with_events(df)

    @staticmethod
    def _validate_sentiment_dataframe(df: pd.DataFrame) -> None:
        """Validate sentiment DataFrame structure."""
        if "sentiment_score" not in df.columns:
            st.error("❌ No se encontró la columna 'sentiment_score' en el DataFrame.")
            st.write("Columnas disponibles:", df.columns.tolist())
            raise ValueError("Invalid DataFrame structure")

    @staticmethod
    def _display_sentiment_metrics(df: pd.DataFrame) -> None:
        """Display sentiment analysis metrics."""
        st.write("**Información del DataFrame de Sentimientos:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Días totales", len(df))
        with col2:
            st.metric("Sentimiento mín", f"{df['sentiment_score'].min():.3f}")
        with col3:
            st.metric("Sentimiento máx", f"{df['sentiment_score'].max():.3f}")
        with col4:
            st.metric("Sentimiento promedio", f"{df['sentiment_score'].mean():.3f}")

        with st.expander("Ver muestra de datos"):
            st.write(df.head(10))

    @staticmethod
    def _show_sentiment_chart_with_events(df: pd.DataFrame) -> None:
        """Show sentiment chart and handle click events."""
        fig = px.line(
            df,
            x="date",
            y="sentiment_score",
            title="Evolución del Sentimiento Promedio Diario",
            labels={"date": "Fecha", "sentiment_score": "Sentimiento Promedio"},
            markers=True
        )

        # Add reference lines
        fig.add_hline(y=0, line_width=2, line_dash="dash", 
                     line_color="gray", annotation_text="Neutral")
        fig.add_hline(y=0.5, line_width=1, line_dash="dot", 
                     line_color="green", annotation_text="Positivo")
        fig.add_hline(y=-0.5, line_width=1, line_dash="dot", 
                     line_color="red", annotation_text="Negativo")

        # Set VADER range and layout
        fig.update_layout(yaxis=dict(range=[-1.1, 1.1]), title_x=0.5)

        # Capture clicks
        selected_points = plotly_events(fig, click_event=True, 
                                      key="sentiment_chart_click")

        if selected_points:
            ChartRenderer._handle_chart_click(selected_points[0])

    @staticmethod
    def _handle_chart_click(selected_point: Dict[str, Any]) -> None:
        """Handle chart click event."""
        clicked_date_str = selected_point['x']
        clicked_date = datetime.strptime(clicked_date_str, '%Y-%m-%d').date()
        
        if st.session_state[SessionStateKeys.SELECTED_SENTIMENT_DATE] != clicked_date:
            st.session_state[SessionStateKeys.SELECTED_SENTIMENT_DATE] = clicked_date
            st.rerun()


class DocumentRenderer:
    """Handles document display operations."""

    @staticmethod
    def render_documents_for_date(documents: List[Dict[str, Any]], 
                                selected_date: datetime.date) -> None:
        """Render documents for a specific date."""
        st.subheader(f"Documentos del {selected_date.strftime('%Y-%m-%d')}")

        filtered_documents = DocumentRenderer._filter_documents_by_date(
            documents, selected_date
        )

        if not filtered_documents:
            st.warning("No se encontraron documentos para esta fecha.")
            return

        DocumentRenderer._display_documents(filtered_documents)

    @staticmethod
    def _filter_documents_by_date(documents: List[Dict[str, Any]], 
                                target_date: datetime.date) -> List[Dict[str, Any]]:
        """Filter documents by specific date."""
        filtered_documents = []
        
        for doc in documents:
            doc_date = pd.to_datetime(doc.get("creationDate")).date()
            if doc_date == target_date:
                filtered_documents.append(doc)
        
        return filtered_documents

    @staticmethod
    def _display_documents(documents: List[Dict[str, Any]]) -> None:
        """Display documents in expandable sections."""
        for doc in documents:
            title = doc.get('title', 'Sin Título')
            
            with st.expander(f"**{title}**"):
                source_name = doc.get('source', {}).get('name', 'N/A')
                source_type = doc.get('sourceType', {}).get('name', 'N/A')
                
                st.caption(f"Fuente: {source_name} | Tipo: {source_type}")
                st.markdown(doc.get('abstract', 'Sin resumen disponible.'))
                
                if link := doc.get('publicationLink'):
                    st.link_button("Leer noticia original", link)


class MetadataRenderer:
    """Handles metadata display operations."""

    @staticmethod
    def render_metadata(meta) -> None:
        """Render series metadata in a structured format."""
        st.subheader("Metadatos de la Serie")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Nombre de la Serie", 
                     value=getattr(meta, 'name', 'N/A'))
            st.metric(label="País", 
                     value=getattr(meta.country, 'name', 'N/A') 
                           if hasattr(meta, 'country') else 'N/A')
            st.metric(label="Fuente", 
                     value=getattr(meta.source, 'name', 'N/A') 
                           if hasattr(meta, 'source') else 'N/A')
        
        with col2:
            st.metric(label="ID de la Serie", 
                     value=str(getattr(meta, 'id', 'N/A')))
            st.metric(label="Frecuencia", 
                     value=getattr(meta.frequency, 'name', 'N/A') 
                           if hasattr(meta, 'frequency') else 'N/A')
            st.metric(label="Última Actualización", 
                     value=str(getattr(meta, 'last_update_time', 'N/A')))
        
        st.markdown("---")


class SeriesChartRenderer:
    """Handles series chart rendering operations."""

    @staticmethod
    def render_series_chart(df: pd.DataFrame, meta) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
        """
        Render series chart with date selection.
        
        Returns:
            Tuple of (start_date, end_date) or (None, None) if invalid
        """
        st.subheader("Gráfico de la Serie (Datos Revisados)")
        
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        
        # Date selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Fecha de Inicio", 
                value=min_date, 
                min_value=min_date, 
                max_value=max_date, 
                key="date_start"
            )
        
        with col2:
            end_date = st.date_input(
                "Fecha de Fin", 
                value=max_date, 
                min_value=min_date, 
                max_value=max_date, 
                key="date_end"
            )

        # Validate date range
        if start_date > end_date:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
            return None, None

        # Filter and display data
        df_filtered = df[
            (df["Date"].dt.date >= start_date) & 
            (df["Date"].dt.date <= end_date)
        ]

        if not df_filtered.empty:
            SeriesChartRenderer._display_chart(df_filtered, meta)
        else:
            st.warning("No hay datos disponibles en el rango de fechas seleccionado.")

        return start_date, end_date

    @staticmethod
    def _display_chart(df: pd.DataFrame, meta) -> None:
        """Display the actual chart."""
        fig = px.line(
            df, 
            x="Date", 
            y="Value", 
            title=getattr(meta, 'name', 'Gráfico de la Serie'),
            labels={"Date": "Fecha", "Value": "Valor"}, 
            markers=True
        )
        
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)


class MainApplication:
    """Main application controller."""

    def __init__(self):
        self.context_analyzer = ContextAnalyzer()

    def render_sidebar(self) -> Optional[str]:
        """Render sidebar and return series ID input."""
        st.sidebar.image(Config.LOGO_PATH, width=Config.SIDEBAR_LOGO_WIDTH)
        st.sidebar.title("Opciones")
        
        series_id_input = st.sidebar.text_input(
            "Introduce el ID de la Serie", 
            help="Ej: 193568001"
        )
        
        return series_id_input

    def render_main_content(self) -> None:
        """Render main application content."""
        st.title("Explorador de Datos de Series Temporales")
        st.markdown("Introduce el ID de una serie para cargar sus metadatos y visualizar su gráfico.")

        series_id_input = self.render_sidebar()

        # Handle series loading
        if st.sidebar.button("Cargar Datos de la Serie"):
            self._handle_series_load(series_id_input)

        # Display series data if available
        visualizer = st.session_state.get(SessionStateKeys.VISUALIZER_OBJECT)
        if visualizer:
            self._display_series_content(visualizer)

        # Display sentiment analysis results
        self._display_sentiment_results()

    def _handle_series_load(self, series_id_input: str) -> None:
        """Handle series loading button click."""
        if series_id_input:
            SeriesDataManager.clear_analysis_data()
            SeriesDataManager.load_series(series_id_input)
        else:
            st.warning("Por favor, introduce un ID de serie.")

    def _display_series_content(self, visualizer: SeriesVisualizer) -> None:
        """Display series metadata, chart, and analysis options."""
        if visualizer.metadata:
            MetadataRenderer.render_metadata(visualizer.metadata)

        df_series = visualizer.process_series_data()

        if df_series is not None and not df_series.empty:
            start_date, end_date = SeriesChartRenderer.render_series_chart(
                df_series, visualizer.metadata
            )

            st.markdown("---")
            st.subheader("Análisis Adicional")
            
            if st.button("Context analysis"):
                if start_date and end_date:
                    self.context_analyzer.perform_analysis(
                        visualizer, 
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.min.time())
                    )

        elif df_series is not None:
            st.info("La serie seleccionada no contiene puntos de datos para graficar.")

    def _display_sentiment_results(self) -> None:
        """Display sentiment analysis results and related documents."""
        sentiment_df = st.session_state.get(SessionStateKeys.SENTIMENT_DF)
        if sentiment_df is not None:
            ChartRenderer.render_sentiment_chart(sentiment_df)

        selected_date = st.session_state.get(SessionStateKeys.SELECTED_SENTIMENT_DATE)
        if selected_date:
            documents = st.session_state.get(SessionStateKeys.RAW_DOCUMENTS, [])
            DocumentRenderer.render_documents_for_date(documents, selected_date)


def main():
    """Main application entry point."""
    # Initialize application
    AppInitializer.setup_page_config()
    AppInitializer.initialize_session_state()
    AppInitializer.download_nltk_resources()

    # Handle authentication and main app
    if st.session_state[SessionStateKeys.LOGGED_IN]:
        Ceic.set_server("https://api.ceicdata.com/v2")
        app = MainApplication()
        app.render_main_content()
    else:
        AuthenticationManager.render_login_page()


if __name__ == '__main__':
    main()