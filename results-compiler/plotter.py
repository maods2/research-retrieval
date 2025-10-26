import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_metric_series(model_metrics, metric_name="map_compiled"):
    """Extract (k, value) pairs for a given metric from a model's metrics."""
    data = model_metrics.get(metric_name, {})
    pattern = re.compile(rf"{metric_name.replace('_compiled', '')}At(\d+)", re.IGNORECASE)
    ks, values = [], []
    for k_str, v in data.items():
        m = pattern.match(k_str)
        if m:
            ks.append(int(m.group(1)))
            values.append(float(v))
    if ks:
        ks, values = zip(*sorted(zip(ks, values)))
    return np.array(ks), np.array(values)


def style_for_model(model_name):
    """Assign consistent color/marker/line style for each model family."""
    styles = {
        "dino": {"color": "#2ca02c", "marker": "s"},
        "dinov2": {"color": "#1f77b4", "marker": "o"},
        "vit": {"color": "#ff7f0e", "marker": "^"},
        "clip": {"color": "#d62728", "marker": "v"},
        "phikon-v2": {"color": "#bcbd22", "marker": "h"},
        "phikon": {"color": "#7f7f7f", "marker": "x"},
        "virchow2": {"color": "#e377c2", "marker": "P"},
        "uni2": {"color": "#17becf", "marker": "X"},
        "uni": {"color": "#8c564b", "marker": "*"},
        "resnet": {"color": "#9467bd", "marker": "D"},
        "supcon": {"color": "#0b4c8c", "linestyle": "--"},
        "liu_dsh": {"color": "#1a75ff", "linestyle": ":"},
        "autoencoder": {"color": "#471be8", "linestyle": "-."},
        "triplet": {"color": "#fc6c6e", "marker": "8"},
    }
    for key, style in styles.items():
        if key.lower() in model_name.lower():
            return style
    return {"color": "gray", "marker": "x"}


def format_label(model_name, values):
    """Format model label with mean ± std for publication plots."""
    mean = np.mean(values) * 100
    std = np.std(values) * 100
    label = model_name.replace("_", "-").capitalize()
    return f"{label} ({mean:.2f}% ± {std:.2f})"


def plot_metric_comparison_from_json(
    json_path,
    dataset_name,
    metric_key="map_compiled",
    title=None,
    legend_location="lower right",
    save_as=None,
):
    """
    Plot comparison of a metric (e.g., mAP) across models for a given dataset.
    Compatible with new JSON structure.
    """
    results = load_json(json_path)
    dataset = results.get(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in JSON.")

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.2,
        "axes.labelsize": 13,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })

    handles, labels = [], []

    for model_name, info in dataset["models"].items():
        metrics = info["metrics"]
        ks, values = extract_metric_series(metrics, metric_name=metric_key)
        if len(ks) == 0:
            continue

        style = style_for_model(model_name)
        line, = plt.plot(
            ks,
            values,
            label=format_label(model_name, values),
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=1.8,
            markersize=5,
        )
        handles.append(line)
        labels.append(model_name)

    plt.xlabel("k")
    plt.ylabel(metric_key.replace("_compiled", "").upper())
    plt.ylim(0, 1)
    plt.xlim(0, max(ks) + 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    if title:
        plt.title(title, fontsize=14, pad=10)
    plt.legend(loc=legend_location, frameon=False, fontsize="small")

    if save_as:
        plt.tight_layout()
        plt.savefig(save_as, format="pdf", dpi=300, bbox_inches="tight")

    plt.show()
