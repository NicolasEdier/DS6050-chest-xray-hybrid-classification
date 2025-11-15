import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
master_json_path = Path("evaluation_results/all_models.json")
output_dir = Path("evaluation/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

output_csv_path = output_dir / "model_summary.csv"

# ------------------------------------------------------------
# Load master JSON
# ------------------------------------------------------------
with open(master_json_path, "r") as f:
    data = json.load(f)

rows = []

for model_name, info in data.items():
    metrics = info["metrics"]

    row = {
        "model": model_name,
        "auroc_mean": round(metrics["auroc_mean"], 3),
        "f1_mean": round(metrics["f1_mean"], 3),
    }

    # AUROC per class: round to 3 decimals
    for idx, value in enumerate(metrics["auroc_per_class"]):
        row[f"auroc_class_{idx}"] = round(value, 3)

    rows.append(row)

# Convert to a DataFrame and save summary table
df_summary = pd.DataFrame(rows)
df_summary.to_csv(output_csv_path, index=False)
print(f"Saved model summary to {output_csv_path}")

def plot_loss_curves(all_models, output_dir):
    """Create one PNG with a training/validation loss subplot per model."""
    model_names = list(all_models.keys())
    num_models = len(model_names)

    fig, axes = plt.subplots(
        nrows=num_models, ncols=1,
        figsize=(10, 4 * num_models),
        sharex=False
    )

    if num_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):

        hist_path = Path(all_models[model_name]["training_history_path"])

        if not hist_path.exists():
            print(f"WARNING: training history not found for {model_name}")
            continue

        with open(hist_path, "r") as f:
            hist = json.load(f)

        epochs = np.arange(1, len(hist["train_losses"]) + 1)

        ax.plot(epochs, hist["train_losses"], label="Train Loss", linewidth=2)
        ax.plot(epochs, hist["val_losses"], label="Val Loss", linewidth=2)

        ax.set_title(f"{model_name} — Training vs Validation Loss", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / "comparison_loss_curves.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss comparison figure: {output_path}")


def plot_auroc_curves(all_models, output_dir):
    """Create one PNG with a validation AUROC subplot per model."""
    model_names = list(all_models.keys())
    num_models = len(model_names)

    fig, axes = plt.subplots(
        nrows=num_models, ncols=1,
        figsize=(10, 4 * num_models),
        sharex=False
    )

    if num_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):

        hist_path = Path(all_models[model_name]["training_history_path"])

        if not hist_path.exists():
            print(f"WARNING: training history not found for {model_name}")
            continue

        with open(hist_path, "r") as f:
            hist = json.load(f)

        epochs = np.arange(1, len(hist["val_aurocs"]) + 1)

        ax.plot(epochs, hist["val_aurocs"], label="Val AUROC", linewidth=2)

        ax.axhline(
            y=hist["best_auroc"],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Best AUROC = {hist['best_auroc']:.3f}"
        )

        ax.set_title(f"{model_name} — Validation AUROC", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUROC")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / "comparison_auroc_curves.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved AUROC comparison figure: {output_path}")

def plot_confusion_heatmaps(all_models_data, output_dir):
    """
    Option 2:
    Create heatmaps comparing FP, FN, TP across models (per disease).
    Produces three heatmaps: FP, FN, TP.
    """

    diseases = list(next(iter(all_models_data.values()))["error_analysis"].keys())
    models = list(all_models_data.keys())

    # Prepare matrices
    fp_matrix = []
    fn_matrix = []
    tp_matrix = []

    for disease in diseases:
        fp_row, fn_row, tp_row = [], [], []
        for model in models:
            ea = all_models_data[model]["error_analysis"][disease]
            fp_row.append(ea["num_fp"])
            fn_row.append(ea["num_fn"])
            tp_row.append(ea["num_tp"])
        fp_matrix.append(fp_row)
        fn_matrix.append(fn_row)
        tp_matrix.append(tp_row)

    # Convert to DataFrames
    fp_df = pd.DataFrame(fp_matrix, index=diseases, columns=models)
    fn_df = pd.DataFrame(fn_matrix, index=diseases, columns=models)
    tp_df = pd.DataFrame(tp_matrix, index=diseases, columns=models)

    # --- Plot heatmaps ---
    for metric_name, df in zip(["fp", "fn", "tp"], [fp_df, fn_df, tp_df]):
        plt.figure(figsize=(10, 12))
        sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{metric_name.upper()} Comparison Across Models")
        plt.xlabel("Model")
        plt.ylabel("Disease")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{metric_name}_comparison.png", dpi=150)
        plt.close()

    print("Saved FP/FN/TP comparison heatmaps.")

def plot_disease_confusion_bars(all_models_data, output_dir):
    """
    Option 3:
    Create one large figure with per-disease bar charts.
    Each subplot: FP, FN, TP for all models.
    All subplots share the same y-axis max for easy comparison.
    """

    diseases = list(next(iter(all_models_data.values()))["error_analysis"].keys())
    models = list(all_models_data.keys())
    num_diseases = len(diseases)

    # ------------------------------------------------------------
    # Compute global max across all FP/FN/TP for all diseases/models
    # ------------------------------------------------------------
    global_max = 0
    for model in models:
        for disease in diseases:
            ea = all_models_data[model]["error_analysis"][disease]
            vals = [ea["num_fp"], ea["num_fn"], ea["num_tp"]]
            global_max = max(global_max, max(vals))

    # Add a small buffer so bars don’t hit the top
    global_max = int(global_max * 1.10)

    # ------------------------------------------------------------
    # Create subplots
    # ------------------------------------------------------------
    fig, axes = plt.subplots(7, 2, figsize=(18, 28))
    axes = axes.flatten()

    for idx, disease in enumerate(diseases):
        ax = axes[idx]

        fp_vals = []
        fn_vals = []
        tp_vals = []

        for model in models:
            ea = all_models_data[model]["error_analysis"][disease]
            fp_vals.append(ea["num_fp"])
            fn_vals.append(ea["num_fn"])
            tp_vals.append(ea["num_tp"])

        x = range(len(models))

        ax.bar(x, fp_vals, width=0.25, label="FP", color="tab:red")
        ax.bar([i + 0.25 for i in x], fn_vals, width=0.25, label="FN", color="tab:orange")
        ax.bar([i + 0.50 for i in x], tp_vals, width=0.25, label="TP", color="tab:green")

        ax.set_xticks([i + 0.25 for i in x])
        ax.set_xticklabels(models, rotation=45, ha="right")

        ax.set_ylim(0, global_max)
        ax.set_title(disease)
        ax.grid(alpha=0.2)

        if idx == 0:
            ax.legend()

    # Turn off unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "disease_confusion_barplots.png", dpi=150)
    plt.close()

    print("Saved disease confusion bar plot grid (shared y-axis).")



plot_loss_curves(data, output_dir)
plot_auroc_curves(data, output_dir)
plot_confusion_heatmaps(data, output_dir)
plot_disease_confusion_bars(data, output_dir)
