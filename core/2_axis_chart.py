import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_sources_comparison(source1_data, source2_data):
    df1 = pd.DataFrame(source1_data)
    df2 = pd.DataFrame(source2_data)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")  
    ax1.set_facecolor("#1a1a2e")        

    ax1.set_xlabel("Time", color="white")
    ax1.set_ylabel("Sentiment Value", color="white")
    ax1.plot(df1['time'], df1['sentiment'], marker='o', label='Sentiment - Source 1', color='yellow', linewidth=2)
    ax1.tick_params(axis='y', colors="white")
    ax1.tick_params(axis='x', colors="white")
    ax1.grid(True, color="gray", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_facecolor("#1a1a2e")  
    ax2.set_ylabel("Series ID", color="white")
    ax2.plot(df2['time'], df2['series_id'], marker='x', label='Series ID - Source 2', color='orange', linestyle="--", linewidth=2)
    ax2.tick_params(axis='y', colors="white")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", facecolor="#16213e", edgecolor="white", fontsize=10)
    for text in legend.get_texts():
        text.set_color("white")

    plt.title("Source 1 Sentiment vs Source 2 Series ID over Time", color="white", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    source1_data = {
        "time": pd.date_range("2025-01-01", periods=10, freq="ME"),
        "sentiment": np.random.uniform(-1, 1, size=10),
        "series_id": np.random.randint(1000, 1100, size=10)
    }

    source2_data = {
        "time": pd.date_range("2025-01-01", periods=10, freq="ME"),
        "sentiment": np.random.uniform(-1, 1, size=10),
        "series_id": np.random.randint(1000, 1100, size=10)
    }

    plot_sources_comparison(source1_data, source2_data)
