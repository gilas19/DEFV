import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Get the parent directory of src (anlp-project)
PROJECT_ROOT = Path(__file__).parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results():
    """Load results from JSON files (now organized by model type)"""
    all_results = []

    # Process both baseline and fine_tuned directories
    for model_type in ['baseline', 'fine_tuned']:
        model_dir = RESULTS_DIR / model_type
        if not model_dir.exists():
            continue

        for result_file in model_dir.glob('*.json'):
            with open(result_file, 'r') as f:
                data = json.load(f)

                # Extract configuration and metrics
                config = data['config']
                stats = data['stats']

                all_results.append({
                    "config_id": config['config_id'],
                    "model": model_type,
                    "with_debate": config['with_debate'],
                    "initiator": config.get('initiator'),
                    "num_turns": config.get('num_turns', 0),
                    "accuracy": stats['accuracy'],
                    "avg_length": stats['avg_length'],
                    "avg_time": stats['avg_time'],
                    "num_claims": len(data['results'])  # Track how many claims were processed
                })

    return pd.DataFrame(all_results)


def load_detailed_results():
    """Load all individual claim results for error analysis"""
    all_claims = []

    for model_type in ['baseline', 'fine_tuned']:
        model_dir = RESULTS_DIR / model_type
        if not model_dir.exists():
            continue

        for result_file in model_dir.glob('*.json'):
            with open(result_file, 'r') as f:
                data = json.load(f)
                config = data['config']

                for result in data['results']:
                    all_claims.append({
                        'config_id': config['config_id'],
                        'model': model_type,
                        'with_debate': config['with_debate'],
                        'num_turns': config.get('num_turns', 0),
                        'gold_label': result['gold_label'],
                        'prediction': result['final_prediction']
                    })

    return pd.DataFrame(all_claims)


def plot_confusion_matrices(df):
    """Generate confusion matrices for each configuration"""
    os.makedirs(PLOTS_DIR / "confusion_matrices", exist_ok=True)
    labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']

    # Plot for each configuration
    for (config_id, model_type) in df[['config_id', 'model']].drop_duplicates().itertuples(index=False):
        config_df = df[(df['config_id'] == config_id) & (df['model'] == model_type)]

        # Create confusion matrix
        cm = confusion_matrix(
            config_df['gold_label'],
            config_df['prediction'],
            labels=labels
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )

        model_type = config_df['model'].iloc[0]
        debate_type = "debate" if config_df['with_debate'].iloc[0] else "no_debate"
        turns = config_df['num_turns'].iloc[0]

        plt.title(
            f"Confusion Matrix\n{model_type} ({debate_type}), {turns} turns\nConfig: {config_id}",
            pad=20
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        filename = f"confusion_matrix_{model_type}_{config_id}.png"
        plt.savefig(PLOTS_DIR / "confusion_matrices" / filename, dpi=300)
        plt.close()


def plot_label_errors(df):
    """Plot misclassification patterns by label"""
    os.makedirs(PLOTS_DIR / "label_errors", exist_ok=True)

    # Calculate error types for each configuration
    error_data = []
    for (config_id, model_type) in df[['config_id', 'model']].drop_duplicates().itertuples(index=False):
        config_df = df[(df['config_id'] == config_id) & (df['model'] == model_type)]

        for gold_label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
            label_df = config_df[config_df['gold_label'] == gold_label]
            total = len(label_df)

            if total == 0:
                continue

            for pred_label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                count = len(label_df[label_df['prediction'] == pred_label])
                error_data.append({
                    'config_id': config_id,
                    'model': config_df['model'].iloc[0],
                    'with_debate': config_df['with_debate'].iloc[0],
                    'num_turns': config_df['num_turns'].iloc[0],
                    'gold_label': gold_label,
                    'pred_label': pred_label,
                    'count': count,
                    'percentage': count / total * 100
                })

    error_df = pd.DataFrame(error_data)

    # Plot error patterns for each model type
    for model_type in ['baseline', 'fine_tuned']:
        model_df = error_df[error_df['model'] == model_type]

        # Plot 1: Error types by gold label (no debate)
        plt.figure(figsize=(12, 6))
        no_debate_df = model_df[model_df['with_debate'] == False]

        if not no_debate_df.empty:
            sns.barplot(
                x='gold_label',
                y='percentage',
                hue='pred_label',
                data=no_debate_df,
                palette=['#2ca02c', '#d62728', '#9467bd']  # Green, red, purple
            )
            plt.title(f"Prediction Distribution by Gold Label\n{model_type} (No Debate)")
            plt.xlabel("Gold Label")
            plt.ylabel("Percentage of Predictions")
            plt.legend(title="Predicted Label")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "label_errors" / f"{model_type}_no_debate_errors.png", dpi=300)
            plt.close()

        # Plot 2: Error types by debate turns
        debate_df = model_df[model_df['with_debate'] == True]
        if not debate_df.empty:
            plt.figure(figsize=(14, 8))
            sns.catplot(
                x='num_turns',
                y='percentage',
                hue='pred_label',
                col='gold_label',
                data=debate_df,
                kind='bar',
                palette=['#2ca02c', '#d62728', '#9467bd'],
                height=5,
                aspect=0.7
            )
            plt.suptitle(f"Prediction Patterns by Gold Label and Debate Turns\n{model_type}", y=1.05)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "label_errors" / f"{model_type}_debate_errors.png", dpi=300)
            plt.close()

def format_accuracy_table(df):
    # Filter for relevant configs only
    relevant = df.copy()

    # Assign readable row labels
    def label_row(row):
        if not row['with_debate']:
            return f"no_debate_{row['model']}"
        starter = "Agent1" if row['initiator'] == "agent_1" else "Agent2"
        return f"{starter}_{row['model']}"

    relevant["row_label"] = relevant.apply(label_row, axis=1)

    # Map turn counts to string for columns
    relevant["col_label"] = relevant.apply(
        lambda row: "no_debate" if not row['with_debate'] else str(row['num_turns']),
        axis=1
    )

    # Pivot table: rows are initiator+model, cols are turns (or no_debate), values = accuracy
    pivot = relevant.pivot_table(
        index="row_label",
        columns="col_label",
        values="accuracy",
        aggfunc="mean"
    )

    # Sort rows and columns to expected order
    row_order = [
        "no_debate_baseline",
        "no_debate_fine_tuned",
        "Agent1_baseline",
        "Agent1_fine_tuned",
        "Agent2_baseline",
        "Agent2_fine_tuned"
    ]
    col_order = ["no_debate", "1", "2", "4", "6"]

    pivot = pivot.reindex(index=row_order, columns=col_order)

    # Save to CSV
    pivot.to_csv(PLOTS_DIR / "accuracy_summary_table.csv")
    print("\n✅ accuracy_summary_table.csv saved.")

    return pivot
def generate_plots(df):
    """Generate all analysis plots with improved formatting and sorted config order"""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ----- Create custom sort key for accuracy plot -----
    def sort_key(row):
        if not row['with_debate']:
            return (0, '', 0)  # No debate first
        initiator_rank = 1 if row['initiator'] == 'agent1' else 2
        return (1, initiator_rank, row['num_turns'])

    df['sort_key'] = df.apply(sort_key, axis=1)
    df_sorted = df.sort_values('sort_key')

    # 1. Main Accuracy Comparison Plot
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        x="config_id",
        y="accuracy",
        hue="model",
        data=df_sorted,
        palette=['#1f77b4', '#ff7f0e']  # Blue for baseline, orange for fine-tuned
    )
    plt.title("Fact Verification Accuracy by Configuration", pad=20, fontsize=14)
    plt.xlabel("Configuration")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode='anchor'
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_all_configs.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2–4. Debate-Specific Plots
    debate_df = df[df["with_debate"]].copy()
    if not debate_df.empty:
        # Convert num_turns to categorical string for x-axis
        debate_df['num_turns'] = debate_df['num_turns'].astype(str) + " turns"

        # 2. Accuracy by Turns
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x="num_turns",
            y="accuracy",
            hue="model",
            style="initiator",
            markers=True,
            dashes=False,
            data=debate_df,
            linewidth=2.5,
            markersize=10,
            palette=['#1f77b4', '#ff7f0e']
        )
        plt.title("Debate Accuracy by Number of Turns")
        plt.xlabel("Number of Turns per Agent")
        plt.ylabel("Accuracy")
        plt.legend(title="Model/Initiator", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "accuracy_vs_turns.png", dpi=300)
        plt.close()

        # 3. Response Length Analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x="num_turns",
            y="avg_length",
            hue="model",
            data=debate_df,
            palette=['#1f77b4', '#ff7f0e']
        )
        plt.title("Response Length Distribution by Turns")
        plt.xlabel("Number of Turns per Agent")
        plt.ylabel("Average Response Length (tokens)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "length_vs_turns.png", dpi=300)
        plt.close()

        # 4. Time vs Accuracy
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="avg_time",
            y="accuracy",
            hue="model",
            style="initiator",
            data=debate_df,
            s=150,
            palette=['#1f77b4', '#ff7f0e']
        )
        plt.title("Response Time vs Accuracy")
        plt.xlabel("Average Response Time (seconds)")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "time_vs_accuracy.png", dpi=300)
        plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_df = df[["num_turns", "accuracy", "avg_length", "avg_time"]].corr()
    sns.heatmap(
        corr_df,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=.5
    )
    plt.title("Metric Correlations", pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=300)
    plt.close()


def calculate_correlations(df):
    """Enhanced correlation analysis with more metrics"""
    debate_df = df[df["with_debate"]]

    if debate_df.empty:
        print("No debate configurations found for correlation analysis")
        return

    print("\nEnhanced Correlation Analysis:")
    print("=" * 50)

    metrics = [
        ("Turns vs Accuracy", "num_turns", "accuracy"),
        ("Turns vs Length", "num_turns", "avg_length"),
        ("Turns vs Time", "num_turns", "avg_time"),
        ("Length vs Accuracy", "avg_length", "accuracy"),
        ("Time vs Accuracy", "avg_time", "accuracy"),
        ("Length vs Time", "avg_length", "avg_time")
    ]

    for name, x, y in metrics:
        r, p = pearsonr(debate_df[x], debate_df[y])
        stars = "*" * min(3, int(-np.log10(p)))  # Add significance stars
        print(f"{name:<20}: r = {r:.3f}{stars}, p = {p:.3f}")


def main():
    # Load both aggregated and detailed results
    df = load_results()
    detailed_df = load_detailed_results()

    if df.empty or detailed_df.empty:
        print("No results found - please run experiments first")
        return
        # 🚨 Add this for debugging


    # Save raw data
    df.to_csv(PLOTS_DIR / "all_results.csv", index=False)
    detailed_df.to_csv(PLOTS_DIR / "detailed_results.csv", index=False)
    format_accuracy_table(df)
    # Generate original plots
    generate_plots(df)
    calculate_correlations(df)

    # Generate new error analysis plots
    plot_confusion_matrices(detailed_df)
    plot_label_errors(detailed_df)

    # Save summary stats
    summary = df.groupby(['model', 'with_debate']).agg({
        'accuracy': ['mean', 'std'],
        'avg_length': 'mean',
        'avg_time': 'mean',
        'num_claims': 'sum'
    })
    summary.to_csv(PLOTS_DIR / "experiment_summary.csv")

    # Calculate label-specific accuracy
    label_acc = detailed_df.groupby(['model', 'with_debate', 'gold_label']).apply(
        lambda x: (x['gold_label'] == x['prediction']).mean()
    ).reset_index(name='accuracy')
    label_acc.to_csv(PLOTS_DIR / "label_accuracy.csv", index=False)

    print("\nAnalysis complete - results saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()