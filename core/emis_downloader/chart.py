import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import os

def prepare_data(csv_path: str, db_path: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, csv_path)
    db_path = os.path.join(BASE_DIR, db_path)

    csv_df = pd.read_csv(
        csv_path,
        usecols=[0, 1],
        header=None,
        names=["date", "csv_value"],
        parse_dates=["date"],
        dayfirst=True,
        skiprows=33
    )
    csv_df["date"] = pd.to_datetime(csv_df["date"], format='mixed', dayfirst=True, errors='coerce')
    csv_df = csv_df.dropna(subset=["date"])
    csv_df.sort_values("date", inplace=True)
    csv_df["date"] = csv_df["date"].dt.strftime("%d/%m/%Y")
    series_values = csv_df[["date", "csv_value"]].values.tolist()

    conn = sqlite3.connect(db_path)
    query = """
        SELECT date, AVG(sentiment_score) AS sentiment_score
        FROM documents
        GROUP BY date
    """
    sqlite_df = pd.read_sql_query(query, conn, parse_dates=["date"])
    conn.close()
    sqlite_df["date"] = pd.to_datetime(sqlite_df["date"], format='mixed', dayfirst=True, errors='coerce')
    sqlite_df = sqlite_df.dropna(subset=["date"])
    sqlite_df.sort_values("date", inplace=True)
    sqlite_df["date"] = sqlite_df["date"].dt.strftime("%d/%m/%Y")
    sqlite_df["sentiment_score"] = sqlite_df["sentiment_score"] * 100
    sentiment_values = sqlite_df[["date", "sentiment_score"]].values.tolist()

    return series_values, sentiment_values

def plot_values_over_time(series_values, sentiment_values):

    """
    Plots time points for series and sentiment values.
    Parameters
    ----------
    series_values : list of [str, float]
        2D array where each element is [date, value] for the main series.
        Date should be in "DD/MM/YYYY" format.
    sentiment_values : list of [str, float]
        2D array where each element is [date, sentiment_score] for sentiment.
        Sentiment score should be a percentage (0-100), date in "DD/MM/YYYY" format.
    """

    fig = go.Figure()

    series_dates, series_vals = zip(*series_values) if series_values else ([], [])
    sentiment_dates, sentiment_scores = zip(*sentiment_values) if sentiment_values else ([], [])

    fig.add_trace(go.Scatter(
        x=series_dates,
        y=series_vals,
        mode="lines+markers",
        name="Series Values"
    ))
    fig.add_trace(go.Scatter(
        x=sentiment_dates,
        y=sentiment_scores,
        mode="lines+markers",
        name="Sentiment Score (%)"
    ))
    fig.update_xaxes(
        tickformat="%d/%m/%Y",
        title="Date (DD/MM/YYYY)"
    )
    fig.update_layout(
        title="Series Values vs Sentiment Score Over Time",
        yaxis_title="Value",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

# Codigo usado para testear

st.title("Series Values vs Sentiment Score Over Time")

csv_path = "../../data/dataset.csv"
sqlite_path = "../../data/emis_docs.sqlite"

series_values, sentiment_values = prepare_data(csv_path, sqlite_path)
plot_values_over_time(series_values, sentiment_values)


