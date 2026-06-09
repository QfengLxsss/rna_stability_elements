from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from rna_stability_elements.config import load_config
from rna_stability_elements.features import (
    codon_feature_table,
    compact_sequence_model_features,
    synonymous_recoded_sequence,
)
from rna_stability_elements.models.baselines import regression_metrics
from rna_stability_elements.models.evaluation import numeric_feature_columns


LABELS = (
    "gene_sense_late_chase_6h_2h",
    "gene_sense_total_chase_6h_0h",
    "exon_sense_late_chase_6h_2h",
    "exon_sense_total_chase_6h_0h",
)
LABEL_DISPLAY = {
    "gene_sense_late_chase_6h_2h": "gene 6h/2h",
    "gene_sense_total_chase_6h_0h": "gene 6h/0h",
    "exon_sense_late_chase_6h_2h": "exon 6h/2h",
    "exon_sense_total_chase_6h_0h": "exon 6h/0h",
}
FEATURE_SETS = ("engineered", "codon_only", "engineered_plus_codon")
MUTAGENESIS_MODES = ("min_gc", "max_gc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run codon-aware features, permutation importance, and synonymous recoding interpretation."
    )
    parser.add_argument("--labels", nargs="*", default=list(LABELS))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument(
        "--split-scope",
        choices=["representative", "all"],
        default="representative",
        help="Use representative fixed splits for mechanism analysis, or all fixed splits.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-permutation-repeats", type=int, default=5)
    parser.add_argument("--max-mutagenesis-genes", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    figure_dir = root / "docs/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source = processed / "figure_source_data"
    source.mkdir(parents=True, exist_ok=True)
    motifs = load_config(root / "configs/project.yaml").get("sequence_features", {}).get("motifs", {})

    all_metrics = []
    all_permutation = []
    all_mutagenesis = []
    for label_id in args.labels:
        print(f"[mechanistic] {label_id}", flush=True)
        augmented, feature_sets = build_augmented_feature_table(processed, label_id)
        sequence_table = pd.read_csv(
            processed / f"parallel_modeling_master_with_sequences_{label_id}.tsv", sep="\t"
        )
        augmented.to_csv(
            processed / f"mechanistic_codon_augmented_features_{label_id}.tsv",
            sep="\t",
            index=False,
        )
        manifest = pd.read_csv(processed / f"fair_benchmark_splits_{label_id}.tsv", sep="\t")
        all_metrics.append(
            evaluate_feature_sets(
                augmented,
                manifest,
                feature_sets=feature_sets,
                label_id=label_id,
                device=args.device,
                n_estimators=args.n_estimators,
                split_scope=args.split_scope,
            )
        )
        all_permutation.append(
            permutation_importance(
                augmented,
                manifest,
                feature_columns=feature_sets["engineered_plus_codon"],
                label_id=label_id,
                device=args.device,
                n_estimators=args.n_estimators,
                n_repeats=args.n_permutation_repeats,
            )
        )
        all_mutagenesis.append(
            synonymous_mutagenesis(
                augmented,
                sequence_table,
                manifest,
                feature_columns=feature_sets["engineered_plus_codon"],
                label_id=label_id,
                motifs=motifs,
                device=args.device,
                n_estimators=args.n_estimators,
                max_genes=args.max_mutagenesis_genes,
            )
        )

    metrics = pd.concat(all_metrics, ignore_index=True)
    permutation = pd.concat(all_permutation, ignore_index=True)
    mutagenesis = pd.concat(all_mutagenesis, ignore_index=True)
    summary = summarize_metrics(metrics)
    mutation_summary = summarize_mutagenesis(mutagenesis)

    outputs = {
        "mechanistic_codon_xgboost_metrics.tsv": metrics,
        "mechanistic_codon_xgboost_summary.tsv": summary,
        "mechanistic_permutation_importance.tsv": permutation,
        "mechanistic_synonymous_mutagenesis.tsv": mutagenesis,
        "mechanistic_synonymous_mutagenesis_summary.tsv": mutation_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(processed / name, sep="\t", index=False)
        frame.to_csv(source / name, sep="\t", index=False)

    make_figures(figure_dir, summary, permutation, mutation_summary)
    write_report(root, summary, permutation, mutation_summary)


def build_augmented_feature_table(
    processed: Path, label_id: str
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    base = pd.read_csv(processed / f"parallel_sequence_model_features_{label_id}.tsv", sep="\t")
    sequence_table = pd.read_csv(
        processed / f"parallel_modeling_master_with_sequences_{label_id}.tsv", sep="\t"
    )
    codon = codon_feature_table(sequence_table)
    codon_columns = [
        column
        for column in codon.select_dtypes(include=[np.number]).columns
        if column != "target_label"
    ]
    augmented = base.merge(codon[["gene_id"] + codon_columns], on="gene_id", how="left")
    engineered = numeric_feature_columns(base, target_column="target_label")
    feature_sets = {
        "engineered": engineered,
        "codon_only": codon_columns,
        "engineered_plus_codon": engineered + codon_columns,
    }
    return augmented, feature_sets


def evaluate_feature_sets(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    feature_sets: dict[str, list[str]],
    label_id: str,
    device: str,
    n_estimators: int,
    split_scope: str,
) -> pd.DataFrame:
    rows = []
    selected_manifest = select_mechanistic_splits(manifest, split_scope=split_scope)
    for feature_set, columns in feature_sets.items():
        print(f"[mechanistic] {label_id} / {feature_set} / {split_scope}", flush=True)
        for split_name, split_manifest in selected_manifest.groupby("split_name", sort=False):
            pipeline = make_xgboost_pipeline(device=device, n_estimators=n_estimators)
            train_index = split_manifest.loc[split_manifest["role"] == "train", "row_index"].to_numpy()
            test_index = split_manifest.loc[split_manifest["role"] == "test", "row_index"].to_numpy()
            pipeline.fit(data.loc[train_index, columns], data.loc[train_index, "target_label"])
            prediction = pipeline.predict(data.loc[test_index, columns])
            metrics = regression_metrics(data.loc[test_index, "target_label"].to_numpy(), prediction)
            metrics.update(
                {
                    "label_id": label_id,
                    "label_display": LABEL_DISPLAY[label_id],
                    "model": "xgboost_mechanistic",
                    "feature_set": feature_set,
                    "evaluation": split_manifest["evaluation"].iloc[0],
                    "split_name": split_name,
                    "holdout_group": split_manifest["holdout_group"].iloc[0],
                    "repeat": int(split_manifest["repeat"].iloc[0]),
                    "n_features": len(columns),
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def permutation_importance(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_id: str,
    device: str,
    n_estimators: int,
    n_repeats: int,
) -> pd.DataFrame:
    split_manifest = manifest[manifest["split_name"] == "random_repeat_0"]
    train_index = split_manifest.loc[split_manifest["role"] == "train", "row_index"].to_numpy()
    test_index = split_manifest.loc[split_manifest["role"] == "test", "row_index"].to_numpy()
    pipeline = make_xgboost_pipeline(device=device, n_estimators=n_estimators)
    pipeline.fit(data.loc[train_index, feature_columns], data.loc[train_index, "target_label"])
    x_test = data.loc[test_index, feature_columns].copy()
    y_test = data.loc[test_index, "target_label"].to_numpy()
    baseline = regression_metrics(y_test, pipeline.predict(x_test))["pearson"]
    rng = np.random.default_rng(13)
    rows = []
    for group_name, columns in feature_groups(feature_columns).items():
        if not columns:
            continue
        for repeat in range(n_repeats):
            perturbed = x_test.copy()
            for column in columns:
                perturbed[column] = rng.permutation(perturbed[column].to_numpy())
            score = regression_metrics(y_test, pipeline.predict(perturbed))["pearson"]
            rows.append(
                {
                    "label_id": label_id,
                    "label_display": LABEL_DISPLAY[label_id],
                    "split_name": "random_repeat_0",
                    "group": group_name,
                    "repeat": repeat,
                    "n_features": len(columns),
                    "baseline_pearson": baseline,
                    "permuted_pearson": score,
                    "pearson_drop": baseline - score,
                }
            )
    return pd.DataFrame(rows)


def synonymous_mutagenesis(
    data: pd.DataFrame,
    sequence_table: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_id: str,
    motifs: dict[str, str],
    device: str,
    n_estimators: int,
    max_genes: int,
) -> pd.DataFrame:
    split_manifest = manifest[manifest["split_name"] == "random_repeat_0"]
    train_index = split_manifest.loc[split_manifest["role"] == "train", "row_index"].to_numpy()
    test_index = split_manifest.loc[split_manifest["role"] == "test", "row_index"].to_numpy()
    pipeline = make_xgboost_pipeline(device=device, n_estimators=n_estimators)
    pipeline.fit(data.loc[train_index, feature_columns], data.loc[train_index, "target_label"])

    test_data = data.loc[test_index].copy()
    baseline_prediction = pipeline.predict(test_data[feature_columns])
    test_data["baseline_prediction"] = baseline_prediction
    selected = test_data.reindex(
        test_data["baseline_prediction"].sub(test_data["target_label"]).abs().sort_values().index
    ).head(max_genes)
    selected_sequences = sequence_table.set_index("gene_id").loc[selected["gene_id"]].reset_index()
    rows = []
    for mode in MUTAGENESIS_MODES:
        mutated_sequences = selected_sequences.copy()
        mutated_sequences["sequence_cds"] = mutated_sequences["sequence_cds"].map(
            lambda value: synonymous_recoded_sequence(value, mode=mode)
        )
        mutated_sequences["sequence_full"] = (
            mutated_sequences["sequence_5utr"].fillna("")
            + mutated_sequences["sequence_cds"].fillna("")
            + mutated_sequences["sequence_3utr"].fillna("")
        )
        engineered = compact_sequence_model_features(mutated_sequences, motifs=motifs)
        codon = codon_feature_table(mutated_sequences)
        codon_columns = [
            column
            for column in codon.select_dtypes(include=[np.number]).columns
            if column != "target_label"
        ]
        mutated = engineered.merge(codon[["gene_id"] + codon_columns], on="gene_id", how="left")
        mutated = mutated.set_index("gene_id").reindex(selected["gene_id"]).reset_index()
        prediction = pipeline.predict(mutated[feature_columns])
        for idx, gene in enumerate(selected.itertuples(index=False)):
            rows.append(
                {
                    "label_id": label_id,
                    "label_display": LABEL_DISPLAY[label_id],
                    "gene_id": gene.gene_id,
                    "gene_symbol": getattr(gene, "gene_symbol", ""),
                    "mode": mode,
                    "baseline_prediction": float(gene.baseline_prediction),
                    "mutated_prediction": float(prediction[idx]),
                    "prediction_delta": float(prediction[idx] - gene.baseline_prediction),
                    "target_label": float(gene.target_label),
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["label_id", "label_display", "feature_set", "evaluation"]):
        row = dict(zip(["label_id", "label_display", "feature_set", "evaluation"], keys))
        row["n_splits"] = len(group)
        row["n_features"] = int(group["n_features"].median())
        for metric in ["pearson", "spearman", "r2", "rmse"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_median"] = group[metric].median()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_mutagenesis(mutagenesis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in mutagenesis.groupby(["label_id", "label_display", "mode"]):
        row = dict(zip(["label_id", "label_display", "mode"], keys))
        row["n_genes"] = group["gene_id"].nunique()
        row["prediction_delta_mean"] = group["prediction_delta"].mean()
        row["prediction_delta_median"] = group["prediction_delta"].median()
        row["prediction_delta_abs_mean"] = group["prediction_delta"].abs().mean()
        row["positive_fraction"] = (group["prediction_delta"] > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    return {
        "length": [c for c in feature_columns if c.endswith("_length")],
        "composition": [
            c
            for c in feature_columns
            if c.endswith("_gc_fraction") or c.endswith("_au_fraction") or c.endswith("_u_fraction")
        ],
        "motif": [c for c in feature_columns if "_motif_" in c],
        "kmer3": [c for c in feature_columns if "_kmer_" in c and len(c.rsplit("_kmer_", 1)[-1]) == 3],
        "kmer4": [c for c in feature_columns if "_kmer_" in c and len(c.rsplit("_kmer_", 1)[-1]) == 4],
        "codon_frequency": [c for c in feature_columns if c.startswith("cds_codon_") and len(c) == len("cds_codon_") + 3],
        "amino_acid": [c for c in feature_columns if c.startswith("cds_aa_")],
        "codon_position_gc": [c for c in feature_columns if c.startswith("cds_codon_pos")],
        "codon_structure": [
            c
            for c in feature_columns
            if c.startswith("cds_")
            and not c.startswith("cds_codon_")
            and not c.startswith("cds_aa_")
            and "_kmer_" not in c
            and "_motif_" not in c
        ],
    }


def select_mechanistic_splits(manifest: pd.DataFrame, *, split_scope: str) -> pd.DataFrame:
    if split_scope == "all":
        return manifest
    keep = {
        "random_repeat_0",
        "holdout_chr1",
        "holdout_chr7",
        "holdout_chr14",
        "holdout_chr19",
        "holdout_chr22",
        "holdout_chrX",
    }
    return manifest[manifest["split_name"].isin(keep)].copy()


def make_xgboost_pipeline(*, device: str, n_estimators: int) -> Pipeline:
    from xgboost import XGBRegressor

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=n_estimators,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    tree_method="hist",
                    device=device,
                    n_jobs=4,
                    random_state=13,
                ),
            ),
        ]
    )


def make_figures(
    figure_dir: Path,
    summary: pd.DataFrame,
    permutation: pd.DataFrame,
    mutation_summary: pd.DataFrame,
) -> None:
    plot_codon_performance(figure_dir, summary)
    plot_permutation_importance(figure_dir, permutation)
    plot_synonymous_mutagenesis(figure_dir, mutation_summary)


def plot_codon_performance(figure_dir: Path, summary: pd.DataFrame) -> None:
    data = summary[summary["evaluation"] == "chromosome_holdout"].pivot_table(
        index="label_display", columns="feature_set", values="pearson_mean"
    )
    data = data.reindex([LABEL_DISPLAY[label] for label in LABELS])
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(len(data.index))
    width = 0.24
    colors = {"engineered": "#3E7C59", "codon_only": "#C9793D", "engineered_plus_codon": "#4A6FA5"}
    for idx, feature_set in enumerate(FEATURE_SETS):
        ax.bar(x + (idx - 1) * width, data[feature_set], width=width, label=feature_set, color=colors[feature_set])
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=25, ha="right")
    ax.set_ylabel("Mean chromosome-holdout Pearson")
    ax.set_title("Codon-aware XGBoost feature comparison")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "mechanistic_codon_feature_performance")


def plot_permutation_importance(figure_dir: Path, permutation: pd.DataFrame) -> None:
    table = permutation.pivot_table(index="group", columns="label_display", values="pearson_drop", aggfunc="mean")
    order = table.mean(axis=1).sort_values(ascending=False).index
    table = table.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = ax.imshow(table, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for row in range(len(table.index)):
        for col in range(len(table.columns)):
            ax.text(col, row, f"{table.iloc[row, col]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Group permutation importance")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson drop")
    save_figure(fig, figure_dir, "mechanistic_permutation_importance")


def plot_synonymous_mutagenesis(figure_dir: Path, mutation_summary: pd.DataFrame) -> None:
    data = mutation_summary.pivot_table(index="label_display", columns="mode", values="prediction_delta_median")
    data = data.reindex([LABEL_DISPLAY[label] for label in LABELS])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(len(data.index))
    width = 0.32
    for idx, mode in enumerate(MUTAGENESIS_MODES):
        ax.bar(x + (idx - 0.5) * width, data[mode], width=width, label=mode)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=25, ha="right")
    ax.set_ylabel("Median prediction delta")
    ax.set_title("Synonymous CDS recoding perturbation")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "mechanistic_synonymous_mutagenesis")


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    root: Path,
    summary: pd.DataFrame,
    permutation: pd.DataFrame,
    mutation_summary: pd.DataFrame,
) -> None:
    chromosome = summary[summary["evaluation"] == "chromosome_holdout"]
    comparison = chromosome.pivot_table(index="label_id", columns="feature_set", values="pearson_mean")
    comparison["codon_gain"] = comparison["engineered_plus_codon"] - comparison["engineered"]
    best_gain = comparison["codon_gain"].idxmax()
    perm = permutation.groupby("group")["pearson_drop"].mean().sort_values(ascending=False)
    mut = mutation_summary.groupby("mode")["prediction_delta_median"].mean()
    mutation_table = mutation_summary.pivot_table(
        index="label_id", columns="mode", values="prediction_delta_median"
    )
    lines = [
        "# Mechanistic Interpretation: Codons, Permutation Importance, and In-Silico Recoding",
        "",
        "This report extends the biological interpretation with codon-aware features, group-level "
        "XGBoost permutation importance, and synonymous CDS recoding perturbations.",
        "",
        "The default run is a representative fixed-split mechanism screen: `random_repeat_0` plus "
        "six chromosome-holdout splits (`chr1`, `chr7`, `chr14`, `chr19`, `chr22`, `chrX`). "
        "It is intended for interpretation, not as a replacement for the full fair benchmark.",
        "",
        "## Main Findings",
        "",
        f"- Adding codon-aware features gives the largest mean chromosome-holdout Pearson change for "
        f"`{best_gain}` ({comparison.loc[best_gain, 'codon_gain']:+.3f}).",
        f"- The strongest average permutation group is `{perm.index[0]}` "
        f"(mean Pearson drop {perm.iloc[0]:.3f}), followed by `{perm.index[1]}` "
        f"({perm.iloc[1]:.3f}).",
        f"- Synonymous CDS GC-min recoding changes predictions by a median of "
        f"{mut.get('min_gc', np.nan):+.3f}; GC-max recoding changes predictions by "
        f"{mut.get('max_gc', np.nan):+.3f}. These are feature-model perturbations, not direct "
        "deep-model attribution.",
        "- SHAP was not run because the current environment does not provide the `shap` package; "
        "permutation importance is the primary model-agnostic importance estimate here.",
        "- Existing GPU-full deep runs did not save checkpoints, so true Transformer/Saluki "
        "position-level in-silico mutagenesis requires a checkpoint-saving re-run.",
        "",
        "## Codon-Aware Feature Performance",
        "",
        "| Label | Engineered | Codon-only | Engineered + codon | Codon gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label_id in LABELS:
        if label_id not in comparison.index:
            continue
        row = comparison.loc[label_id]
        lines.append(
            f"| {LABEL_DISPLAY[label_id]} | {row['engineered']:.3f} | "
            f"{row['codon_only']:.3f} | {row['engineered_plus_codon']:.3f} | "
            f"{row['codon_gain']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "![Codon performance](figures/mechanistic_codon_feature_performance.png)",
            "",
            "## Group Permutation Importance",
            "",
            "| Rank | Feature group | Mean Pearson drop |",
            "| ---: | --- | ---: |",
        ]
    )
    for index, (group, value) in enumerate(perm.items(), start=1):
        lines.append(f"| {index} | `{group}` | {value:.3f} |")
    lines.extend(
        [
            "",
            "![Permutation importance](figures/mechanistic_permutation_importance.png)",
            "",
            "## Synonymous CDS Recoding",
            "",
            "| Label | GC-min median delta | GC-max median delta |",
            "| --- | ---: | ---: |",
        ]
    )
    for label_id in LABELS:
        if label_id not in mutation_table.index:
            continue
        row = mutation_table.loc[label_id]
        lines.append(
            f"| {LABEL_DISPLAY[label_id]} | {row.get('min_gc', np.nan):+.3f} | "
            f"{row.get('max_gc', np.nan):+.3f} |"
        )
    lines.extend(
        [
            "",
            "GC-min and GC-max recoding preserve amino-acid sequence but deliberately change synonymous "
            "codon choices. Large prediction shifts therefore support a codon-usage-sensitive CDS "
            "signal, while still remaining model-based perturbations.",
        "",
        "![Synonymous mutagenesis](figures/mechanistic_synonymous_mutagenesis.png)",
        "",
        "## Outputs",
        "",
        "- `data/processed/mechanistic_codon_augmented_features_<label>.tsv`",
        "- `data/processed/mechanistic_codon_xgboost_metrics.tsv`",
        "- `data/processed/mechanistic_codon_xgboost_summary.tsv`",
        "- `data/processed/mechanistic_permutation_importance.tsv`",
        "- `data/processed/mechanistic_synonymous_mutagenesis.tsv`",
        "- `data/processed/mechanistic_synonymous_mutagenesis_summary.tsv`",
        "- `docs/figures/mechanistic_codon_feature_performance.{png,svg,pdf}`",
        "- `docs/figures/mechanistic_permutation_importance.{png,svg,pdf}`",
        "- `docs/figures/mechanistic_synonymous_mutagenesis.{png,svg,pdf}`",
        ]
    )
    (root / "docs/mechanistic_interpretation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
