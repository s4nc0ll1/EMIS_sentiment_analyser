import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import os

def plot_values_over_time(csv_path: str, db_path: str):
    """Plots values from a CSV and averaged sentiment_score from SQLite over time (DD/MM/YYYY format)."""

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
        skiprows=33  # Skip first 33 rows
    )

    conn = sqlite3.connect(db_path)
    query = """
        SELECT date, AVG(sentiment_score) AS sentiment_score
        FROM documents
        GROUP BY date
    """
    sqlite_df = pd.read_sql_query(query, conn, parse_dates=["date"])
    conn.close()

    csv_df["date"] = pd.to_datetime(csv_df["date"], format='mixed', dayfirst=True, errors='coerce')
    sqlite_df["date"] = pd.to_datetime(sqlite_df["date"], format='mixed', dayfirst=True, errors='coerce')

    csv_df.sort_values("date", inplace=True)
    sqlite_df.sort_values("date", inplace=True)

    csv_df["date"] = csv_df["date"].dt.strftime("%d/%m/%Y")
    sqlite_df["date"] = sqlite_df["date"].dt.strftime("%d/%m/%Y")

    # Multiply sentiment score by 100 to show as percentage
    sqlite_df["sentiment_score"] = sqlite_df["sentiment_score"] * 100

    fig = go.Figure()

    # CSV trace
    fig.add_trace(go.Scatter(
        x=csv_df["date"],
        y=csv_df["csv_value"],
        mode="lines+markers",
        name="Series Values"
    ))

    # SQLite trace
    fig.add_trace(go.Scatter(
        x=sqlite_df["date"],
        y=sqlite_df["sentiment_score"],
        mode="lines+markers",
        name="Sentiment Score (%)" 
    ))

    # Format dates on x-axis 
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

# st.title("Series Values vs Sentiment Score Over Time")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# csv_path = os.path.join(BASE_DIR, "../../data/dataset.csv")
# sqlite_path = os.path.join(BASE_DIR, "../../data/emis_docs.sqlite")

# plot_values_over_time(csv_path, sqlite_path)


