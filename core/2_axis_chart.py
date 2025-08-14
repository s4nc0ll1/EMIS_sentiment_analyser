import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplcursors

def plot_sources_comparison(source1_data, source2_data):
    df1 = pd.DataFrame(source1_data)
    df2 = pd.DataFrame(source2_data)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")  
    ax1.set_facecolor("#1a1a2e")        

    ax1.set_xlabel("Time", color="white")
    ax1.set_ylabel("Sentiment Value", color="white")

    line1, = ax1.plot(df1['time'], df1['sentiment'], marker='x', label='Sentiment - Source 1',
                    color='orange', linestyle="--", linewidth=2)
    
    ax1.fill_between(df1['time'], df1['sentiment'], 0,
                    where=(df1['sentiment'] >= 0),
                    interpolate=True, color='orange', alpha=0.2)
    ax1.fill_between(df1['time'], df1['sentiment'], 0,
                    where=(df1['sentiment'] < 0),
                    interpolate=True, color='orange', alpha=0.2)

    ax1.tick_params(axis='y', colors="white")
    ax1.tick_params(axis='x', colors="white")
    ax1.grid(True, color="gray", linestyle="--", alpha=0.5)

    ax1.axhline(y=0, color="red", linestyle="-", linewidth=1.5, alpha=0.8)

    series_min = df2['series_id'].min()
    series_max = df2['series_id'].max()
    series_norm = (df2['series_id'] - series_min) / (series_max - series_min)
    vertical_offset = max(df1['sentiment'].max(), 1.0) + 0.5
    series_shifted = series_norm * 2 + vertical_offset

    line2, = ax1.plot(df2['time'], series_shifted, marker='o', color='yellow', linewidth=2,
                    label="Series ID (visual offset)")

    ax2 = ax1.twinx()
    ax2.set_facecolor("#1a1a2e")  
    ax2.set_ylabel("Series ID", color="white")
    ax2.tick_params(axis='y', colors="white")
    ax2.set_ylim(series_min, series_max)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    legend = ax1.legend(lines_1, labels_1, loc="upper left", facecolor="#16213e", edgecolor="white", fontsize=10)
    for text in legend.get_texts():
        text.set_color("white")

    plt.title("Source 1 Sentiment vs Source 2 Series ID over Time", color="white", fontsize=14)

    cursor = mplcursors.cursor([line1, line2], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        idx = int(round(sel.index))  
        if sel.artist == line1:
            sel.annotation.set_text(
                f"Time: {df1['time'].iloc[idx].date()}\nSentiment: {df1['sentiment'].iloc[idx]:.2f}"
            )
        elif sel.artist == line2:
            sel.annotation.set_text(
                f"Time: {df2['time'].iloc[idx].date()}\nSeries ID: {df2['series_id'].iloc[idx]}"
            )

        sel.annotation.get_bbox_patch().set(fc="#16213e", ec="white")
        sel.annotation.set_color("white")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    source1_data = {
        "time": pd.date_range("2025-01-01", periods=20, freq="ME"),
        "sentiment": np.random.uniform(-1, 1, size=20),
        "series_id": np.random.randint(3000, 4000, size=20)
    }

    source2_data = {
        "time": pd.date_range("2025-01-01", periods=20, freq="ME"),
        "sentiment": np.random.uniform(-1, 1, size=20),
        "series_id": np.random.randint(3000, 4000, size=20)
    }

    plot_sources_comparison(source1_data, source2_data)
