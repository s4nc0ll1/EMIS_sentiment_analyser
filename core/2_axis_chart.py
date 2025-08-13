import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_sources_comparison(source1_data, source2_data):
    df1 = pd.DataFrame(source1_data)
    df2 = pd.DataFrame(source2_data)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Sentiment from Source 1
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Sentiment Value", color="tab:blue")
    ax1.plot(df1['time'], df1['sentiment'], marker='o', label='Sentiment - Source 1', color='blue')
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.grid(True)

    # Plot Series ID from Source 2 on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Series ID", color="tab:red")
    ax2.plot(df2['time'], df2['series_id'], marker='x', label='Series ID - Source 2', color='red', linestyle="--")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Source 1 Sentiment vs Source 2 Series ID over Time")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    source1_data = {
        "time": pd.date_range("2024-01-01", periods=10, freq="ME"),
        "sentiment": np.random.uniform(-1, 1, size=10),
        "series_id": np.random.randint(1000, 1100, size=10)
    }

    source2_data = {
        "time": pd.date_range("2024-01-01", periods=10, freq="ME"),
        "sentiment": np.random.uniform(-1, 1, size=10),
        "series_id": np.random.randint(1000, 1100, size=10)
    }

    plot_sources_comparison(source1_data, source2_data)
