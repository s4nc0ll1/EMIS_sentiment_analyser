"""
SeriesVisualizer module for handling CEIC time series data visualization.

This module provides comprehensive functionality for fetching, processing,
and visualizing CEIC economic time series data including vintages analysis.
"""

import threading
from typing import Optional, List, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


class ColorPalette:
    """Color constants for consistent styling."""
    TEALISH = "#00A88F"
    LAVENDER = "#F2CEEF"
    DEEP_PURPLE = "#792D82"


class DataFetchError(Exception):
    """Raised when data fetching operations fail."""
    pass


class DataProcessingError(Exception):
    """Raised when data processing operations fail."""
    pass


class VisualizationError(Exception):
    """Raised when visualization operations fail."""
    pass


class SeriesDataFetcher:
    """Handles all data fetching operations for series data."""

    def __init__(self, ceic_client, series_id: str):
        self.ceic_client = ceic_client
        self.series_id = series_id

    def fetch_metadata(self) -> Optional[Any]:
        """
        Fetch series metadata.
        
        Returns:
            Series metadata object or None if fetch fails
            
        Raises:
            DataFetchError: If metadata fetching fails
        """
        try:
            result = self.ceic_client.series_metadata(series_id=self.series_id)
            return result.data[0].metadata
        except Exception as e:
            raise DataFetchError(f"Error fetching metadata for series {self.series_id}: {e}")

    def fetch_series_data(self) -> Optional[Any]:
        """
        Fetch series data points.
        
        Returns:
            Series data object or None if fetch fails
            
        Raises:
            DataFetchError: If series data fetching fails
        """
        try:
            result = self.ceic_client.series_data(series_id=self.series_id)
            return result.data[0]
        except Exception as e:
            raise DataFetchError(f"Error fetching series data for {self.series_id}: {e}")

    def fetch_vintages_data(self, vintages_count: int = 10000, 
                           vintages_start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch vintages data with optional date filtering.
        
        Args:
            vintages_count: Maximum number of vintages to fetch
            vintages_start_date: Start date for vintages filtering
            
        Returns:
            DataFrame with vintages data or None if fetch fails
            
        Raises:
            DataFetchError: If vintages data fetching fails
        """
        try:
            params = {"series_id": self.series_id}
            
            if vintages_start_date:
                params["vintages_start_date"] = vintages_start_date
            else:
                params["vintages_count"] = vintages_count
            
            data = self.ceic_client.series_vintages_as_dict(**params)
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index)
            df.columns = pd.to_datetime(df.columns)
            
            # Filter and sort data
            return df.sort_index(ascending=True).loc['2014-03-01':].copy()
            
        except Exception as e:
            raise DataFetchError(f"Error fetching vintages data for {self.series_id}: {e}")


class SeriesDataProcessor:
    """Handles processing of series data into usable formats."""

    @staticmethod
    def process_time_points_to_dataframe(series_data) -> Optional[pd.DataFrame]:
        """
        Convert series time points to a pandas DataFrame.
        
        Args:
            series_data: Raw series data object
            
        Returns:
            DataFrame with Date and Value columns
            
        Raises:
            DataProcessingError: If processing fails
        """
        if not series_data:
            return None
        
        try:
            time_points = series_data.time_points
            if isinstance(time_points, dict):
                time_points = list(time_points.values())
            
            # Sort time points by date
            sorted_time_points = sorted(
                time_points, 
                key=lambda tp: tp.date if hasattr(tp, "date") else tp['date']
            )
            
            # Extract dates and values
            dates = [
                tp.date if hasattr(tp, "date") else tp['date'] 
                for tp in sorted_time_points
            ]
            values = [
                tp.value if hasattr(tp, "value") else tp['value'] 
                for tp in sorted_time_points
            ]
            
            return pd.DataFrame({
                'Date': pd.to_datetime(dates), 
                'Value': values
            })
            
        except Exception as e:
            raise DataProcessingError(f"Error processing time points: {e}")

    @staticmethod
    def calculate_vintage_differences(df_vintages: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate differences between consecutive vintage updates.
        
        Args:
            df_vintages: DataFrame with vintage data
            
        Returns:
            DataFrame with revision differences
        """
        if df_vintages is None or df_vintages.empty:
            return pd.DataFrame()
        
        df_sorted = df_vintages.sort_index(axis=1, ascending=True)
        return df_sorted.diff(axis=1)

    @staticmethod
    def calculate_first_last_differences(df_vintages: pd.DataFrame) -> pd.Series:
        """
        Calculate differences between first and last available values per vintage.
        
        Args:
            df_vintages: DataFrame with vintage data
            
        Returns:
            Series with differences indexed by date
        """
        if df_vintages is None or df_vintages.empty:
            return pd.Series()
        
        df_sorted = df_vintages.sort_index(axis=1, ascending=True)
        
        first_vals = df_sorted.apply(
            lambda row: row.dropna().iloc[0] if row.dropna().size > 0 else np.nan, 
            axis=1
        )
        last_vals = df_sorted.apply(
            lambda row: row.dropna().iloc[-1] if row.dropna().size > 0 else np.nan, 
            axis=1
        )
        
        diff = last_vals - first_vals
        diff.index = pd.to_datetime(diff.index)
        return diff.sort_index()


class SeriesVisualizer:
    """
    Main class for visualizing CEIC series data with comprehensive analysis capabilities.
    
    This class provides methods for fetching, processing, and visualizing time series data
    including advanced vintage analysis and revision tracking.
    """

    def __init__(self, ceic_client, series_id: str, 
                 vintages_count: int = 10000, 
                 vintages_start_date: Optional[str] = None):
        """
        Initialize the SeriesVisualizer.
        
        Args:
            ceic_client: CEIC API client instance
            series_id: Unique identifier for the series
            vintages_count: Maximum number of vintages to fetch
            vintages_start_date: Start date for vintages filtering
        """
        self.series_id = series_id
        self.vintages_count = vintages_count
        self.vintages_start_date = vintages_start_date
        
        # Initialize components
        self.data_fetcher = SeriesDataFetcher(ceic_client, series_id)
        self.data_processor = SeriesDataProcessor()
        
        # Initialize data containers
        self.metadata = None
        self.series_data = None
        self.df_reversed = None

    def fetch_all_data(self) -> None:
        """
        Fetch all required data (metadata, series data, vintages) concurrently.
        
        Raises:
            DataFetchError: If any data fetching operation fails
        """
        def fetch_metadata():
            try:
                self.metadata = self.data_fetcher.fetch_metadata()
            except DataFetchError as e:
                print(f"Metadata fetch failed: {e}")
                self.metadata = None

        def fetch_series_data():
            try:
                self.series_data = self.data_fetcher.fetch_series_data()
            except DataFetchError as e:
                print(f"Series data fetch failed: {e}")
                self.series_data = None

        def fetch_vintages_data():
            try:
                self.df_reversed = self.data_fetcher.fetch_vintages_data(
                    self.vintages_count, 
                    self.vintages_start_date
                )
            except DataFetchError as e:
                print(f"Vintages data fetch failed: {e}")
                self.df_reversed = None

        # Execute fetching operations concurrently
        threads = [
            threading.Thread(target=fetch_metadata),
            threading.Thread(target=fetch_series_data),
            threading.Thread(target=fetch_vintages_data)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()

    def process_series_data(self) -> Optional[pd.DataFrame]:
        """
        Process raw series data into a clean DataFrame.
        
        Returns:
            DataFrame with processed series data or None if processing fails
        """
        try:
            return self.data_processor.process_time_points_to_dataframe(self.series_data)
        except DataProcessingError as e:
            print(f"Series data processing failed: {e}")
            return None

    def plot_series(self, df: Optional[pd.DataFrame] = None) -> Optional[go.Figure]:
        """
        Create a line plot of the series data.
        
        Args:
            df: DataFrame with series data. If None, processes internal data
            
        Returns:
            Plotly figure object or None if plotting fails
        """
        if df is None:
            df = self.process_series_data()
        
        if df is None or df.empty:
            return None
        
        try:
            country_name = self._get_country_name()
            series_name = self._get_series_name()
            title = f"{country_name}, {series_name} Latest revised data"
            
            fig = px.line(
                df, 
                x='Date', 
                y='Value', 
                markers=True,
                title=title,
                color_discrete_sequence=[ColorPalette.TEALISH]
            )
            
            fig.update_layout(
                xaxis_title="Date", 
                yaxis_title="Value"
            )
            fig.update_xaxes(tickformat="%Y-%m-%d")
            fig.update_traces(hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error creating series plot: {e}")

    def style_vintages_table(self) -> Optional[Any]:
        """
        Create a styled table highlighting vintage changes.
        
        Returns:
            Styled DataFrame or None if styling fails
        """
        if self.df_reversed is None or self.df_reversed.empty:
            return None
        
        try:
            df_sorted = self.df_reversed.sort_index(ascending=True, axis=1).copy()
            df_sorted.columns = df_sorted.columns.strftime('%Y-%m-%d')
            df_sorted.index = df_sorted.index.strftime('%Y-%m-%d')
            
            def highlight_vintage_changes(row):
                styles = [''] * len(row)
                values = row.values
                
                for i in range(1, len(values)):
                    if pd.notna(values[i]):
                        if (pd.isna(values[i-1]) or 
                            (pd.notna(values[i-1]) and values[i] != values[i-1])):
                            styles[i] = f'background-color: {ColorPalette.LAVENDER}'
                
                return styles
            
            return df_sorted.style.apply(
                highlight_vintage_changes, 
                axis=1
            ).format("{:.2f}")
            
        except Exception as e:
            raise VisualizationError(f"Error styling vintages table: {e}")

    def plot_vintages_heatmap(self) -> Optional[plt.Figure]:
        """
        Create a heatmap showing differences between consecutive vintage updates.
        
        Returns:
            Matplotlib figure or None if plotting fails
        """
        if self.df_reversed is None or self.df_reversed.empty:
            return None
        
        try:
            df_revision_diff = self.data_processor.calculate_vintage_differences(
                self.df_reversed
            )
            
            df_revision_diff.columns = df_revision_diff.columns.strftime('%Y-%m-%d')
            df_revision_diff.index = df_revision_diff.index.strftime('%Y-%m-%d')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                df_revision_diff, 
                cmap='coolwarm', 
                center=0,
                cbar_kws={'label': 'Difference'}, 
                ax=ax
            )
            
            ax.set_title('Heatmap of Differences Between Consecutive Vintage Updates')
            ax.set_xlabel('Vintage Update (YYYY-MM-DD)')
            ax.set_ylabel('Timepoint Date (YYYY-MM-DD)')
            
            plt.setp(
                ax.get_xticklabels(), 
                rotation=45, 
                ha="right", 
                rotation_mode="anchor"
            )
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error creating vintages heatmap: {e}")

    def plot_animated_vintages(self) -> Optional[go.Figure]:
        """
        Create an animated plot showing vintage evolution over time.
        
        Returns:
            Plotly figure with animation or None if plotting fails
        """
        if self.df_reversed is None or self.df_reversed.empty:
            return None
        
        try:
            df_sorted = self.df_reversed.sort_index(axis=1, ascending=True)
            df_long = df_sorted.reset_index().melt(
                id_vars='index', 
                var_name='vintage', 
                value_name='value'
            )
            df_long.rename(columns={'index': 'time'}, inplace=True)
            
            df_long['time'] = pd.to_datetime(df_long['time'])
            df_long['vintage'] = pd.to_datetime(df_long['vintage']).dt.strftime('%Y-%m-%d')
            
            # Calculate y-axis range with padding
            y_min = df_long['value'].min()
            y_max = df_long['value'].max()
            y_padding = (y_max - y_min) * 0.05
            
            fig = px.line(
                df_long, 
                x="time", 
                y="value", 
                animation_frame="vintage",
                title="Animated Timeseries Vintages",
                range_y=[y_min - y_padding, y_max + y_padding],
                color_discrete_sequence=[ColorPalette.DEEP_PURPLE]
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Value",
                xaxis=dict(tickformat="%Y-%m-%d"),
            )
            
            fig.update_traces(hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y}")
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error creating animated vintages plot: {e}")

    def plot_vintage_comparison(self, date1: Optional[str] = None, 
                              date2: Optional[str] = None) -> Optional[go.Figure]:
        """
        Create a comparison plot between two vintage dates.
        
        Args:
            date1: First vintage date for comparison
            date2: Second vintage date for comparison
            
        Returns:
            Plotly figure comparing two vintages or None if plotting fails
        """
        if self.df_reversed is None or self.df_reversed.empty:
            return None
        
        try:
            df_str = self.df_reversed.copy()
            df_str.index = df_str.index.astype(str)
            df_str.columns = df_str.columns.astype(str)
            
            available_dates = df_str.columns.tolist()
            
            # Use first two dates if none provided
            if date1 is None or date2 is None:
                if len(available_dates) < 2:
                    return None
                date1 = available_dates[0]
                date2 = available_dates[1]
            
            if date1 not in available_dates or date2 not in available_dates:
                return None
            
            trace0 = go.Scatter(
                x=df_str.index, 
                y=df_str[date1], 
                mode="lines+markers",
                name=date1, 
                line=dict(color=ColorPalette.TEALISH)
            )
            
            trace1 = go.Scatter(
                x=df_str.index, 
                y=df_str[date2], 
                mode="lines+markers",
                name=date2, 
                line=dict(color=ColorPalette.DEEP_PURPLE)
            )
            
            fig = go.Figure(data=[trace0, trace1])
            fig.update_layout(
                title=f"Vintage Comparison: {date1} vs {date2}",
                xaxis_title="Time",
                yaxis_title="Value"
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error creating vintage comparison plot: {e}")

    def plot_vintage_differences(self) -> Optional[plt.Figure]:
        """
        Create a bar plot showing differences between first and last values per vintage.
        
        Returns:
            Matplotlib figure or None if plotting fails
        """
        if self.df_reversed is None or self.df_reversed.empty:
            return None
        
        try:
            diff = self.data_processor.calculate_first_last_differences(
                self.df_reversed
            )
            
            if diff.empty:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate bar width
            width = (
                (diff.index[1] - diff.index[0]).days * 0.8 
                if len(diff.index) > 1 else 10
            )
            
            # Set colors based on positive/negative values
            colors = [
                ColorPalette.DEEP_PURPLE if value >= 0 else ColorPalette.LAVENDER 
                for value in diff.values
            ]
            
            ax.bar(diff.index, diff.values, width=width, color=colors)
            ax.axhline(0, color='black', linewidth=1)
            
            ax.set_title("Difference Between Last and First Available Values per Vintage")
            ax.set_xlabel("Vintage Date")
            ax.set_ylabel("Difference (Last - First)")
            
            plt.setp(
                ax.get_xticklabels(), 
                rotation=45, 
                ha="right", 
                rotation_mode="anchor"
            )
            
            fig.tight_layout()
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Error creating vintage differences plot: {e}")

    def _get_country_name(self) -> str:
        """Get country name from metadata."""
        return (
            getattr(self.metadata.country, 'name', 'N/A') 
            if hasattr(self.metadata, 'country') else 'N/A'
        )

    def _get_series_name(self) -> str:
        """Get series name from metadata."""
        return getattr(self.metadata, 'name', 'N/A')

    def get_summary_statistics(self) -> dict:
        """
        Generate summary statistics for the series.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.process_series_data()
        if df is None or df.empty:
            return {}
        
        return {
            'count': len(df),
            'mean': df['Value'].mean(),
            'std': df['Value'].std(),
            'min': df['Value'].min(),
            'max': df['Value'].max(),
            'first_date': df['Date'].min(),
            'last_date': df['Date'].max(),
            'series_name': self._get_series_name(),
            'country_name': self._get_country_name()
        }