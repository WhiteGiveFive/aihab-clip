"""Reproduce a descriptive confusion review from authenticated saved predictions.

No model inference, policy selection, bootstrap resampling, or test-registry
write is performed. The source bundles remain immutable.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness"
GEOMETRY = BASE / "joint_geometry_adapted_encoder_v1"
RF = BASE / "relaxed_guardrails_rf_v1"
STUDY = "geometry_router_confusion_review_v1"
PARENT = REPO / "dev_plans/reliability_aware_geo_fusion/reports/2026-09-15_geo_helpfulness_router/report.md"
RF_POLICY = "relaxed__rf__primary_full727__plots_20__harm_0p7"
SEEDS = (1, 2, 3, 4)
KEY = ["training_seed", "row_uid"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def policy(k: int) -> str:
    return f"k{k}__plots_20__harm_0p7"


def aligned(rows: pd.DataFrame) -> pd.DataFrame:
    assert not rows.duplicated(KEY).any(), "Duplicate seed/image identities"
    return rows.sort_values(KEY).reset_index(drop=True)


def load_sources():
    hashes = {}

    def bind(path, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        hashes[str(path.relative_to(REPO))] = actual
        return path

    receipt = read_json(bind(GEOMETRY / "receipt.json"))

    def geometry_file(name):
        return bind(GEOMETRY / name, receipt["output_sha256"][name])

    policies = read_json(geometry_file("frozen_policies.json"))
    selection = read_json(geometry_file("selection_manifest.json"))
    assert digest(GEOMETRY / "frozen_policies.json") == selection["frozen_policies_sha256"]
    assert selection["selection_cohort"] == "development-validation"
    assert not selection["test_outcomes_used"]
    banks = {bank: aligned(pd.read_parquet(geometry_file(f"test_{bank}_scored.parquet")))
             for bank in (3378, 4200)}
    saved_metrics = pd.read_csv(geometry_file("performance_by_seed.csv"))
    intervals = pd.read_csv(geometry_file("paired_intervals.csv"))
    geometry_file("paired_bootstrap.json")
    for name in ("declaration.json", "predictions/manifest.json", "scores/manifest.json"):
        path = RF / name
        bind(path, receipt["source_sha256"][str(path)])
    ontology = read_json(RF / "declaration.json")["ontology"]
    labels = [row["label_name"] for row in ontology]
    assert [row["dense_id"] for row in ontology] == list(range(18))
    class_path = bind(REPO / "data/__init__.py")
    class_map = next(ast.literal_eval(node.value) for node in ast.parse(class_path.read_text()).body
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                     and t.id == "REASSIGN_LABEL_NAME_L3" for t in node.targets))
    assert labels == [class_map[row["canonical_l3_id"]] for row in ontology]
    rf_manifest = read_json(RF / "scores/manifest.json")
    rf_receipt = read_json(bind(RF / "scores/receipt.json",
                               rf_manifest["artifacts"]["receipt.json"]["file_sha256"]))

    def rf_file(name):
        return bind(RF / "scores" / name, rf_manifest["artifacts"][name]["file_sha256"])

    classification = read_json(rf_file("best_model/classification.json"))
    assert digest(RF / "scores/best_model/classification.json") == rf_receipt["output_file_sha256"]["best_model/classification.json"]
    historical = aligned(pd.read_parquet(rf_file("best_model/actions_with_outcomes.parquet"),
                                        filters=[("policy_id", "=", RF_POLICY)]))
    assert len(historical) == 5388
    match = KEY + ["file", "plot_idx", "label_id_dense", "canonical_l3_id", "label_name", "raw_pred", "geo_pred"]
    first = banks[3378]
    for bank, rows in banks.items():
        assert len(rows) == 5388 and set(rows.training_seed) == set(SEEDS)
        assert rows.row_uid.nunique() == 1347 and rows.plot_idx.nunique() == 531
        assert rows.groupby("training_seed").size().eq(1347).all()
        assert rows[match].equals(historical[match]), "Historical RF cohort/expert mismatch"
        assert np.array_equal(rows.rf_pred, historical.final_pred)
        assert np.array_equal(rows.rf_acted, historical.acted)
        assert rows[match + ["image_pred", "rf_pred"]].equals(first[match + ["image_pred", "rf_pred"]])
        identities = ["row_uid", "file", "plot_idx", "label_id_dense", "canonical_l3_id", "label_name"]
        seed_one = rows.loc[rows.training_seed.eq(1), identities].reset_index(drop=True)
        for seed in SEEDS:
            assert seed_one.equals(rows.loc[rows.training_seed.eq(seed), identities].reset_index(drop=True))
        for k in (5, 10):
            selected = policies[policy(k)]
            threshold = float.fromhex(selected["threshold_specification"]["value_hex"])
            assert selected["selected_feasible"] and selected["threshold_specification"]["comparison"] == "strict_gt"
            score = np.minimum(rows[f"k{k}_A_image"], rows[f"k{k}_A_geo"])
            np.testing.assert_allclose(rows[f"k{k}_score"], score, equal_nan=True)
            acted = rows.raw_pred.ne(rows.geo_pred) & rows[f"k{k}_eligible"] & score.gt(threshold)
            assert np.array_equal(acted, rows[policy(k) + "_acted"])
            assert np.array_equal(np.where(acted, rows.geo_pred, rows.raw_pred), rows[policy(k) + "_pred"])
            for alternate in policies:
                if policies[alternate]["k"] == k:
                    assert np.array_equal(rows[alternate + "_pred"], rows[policy(k) + "_pred"])
    bind(PARENT)
    return banks, policies, labels, ontology, saved_metrics, classification, intervals, hashes


def confusion(rows, column):
    truth = rows.label_id_dense.to_numpy(dtype=int)
    pred = rows[column].to_numpy(dtype=int)
    assert np.isin(truth, np.arange(18)).all() and np.isin(pred, np.arange(18)).all()
    return np.bincount(18 * truth + pred, minlength=18 * 18).reshape(18, 18)


def rates(cm):
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    recall = np.divide(tp, support, out=np.zeros(18, dtype=float), where=support != 0)
    precision = np.divide(tp, predicted, out=np.zeros(18, dtype=float), where=predicted != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(18, dtype=float), where=(support + predicted) != 0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": tp.sum() / cm.sum(), "weighted_f1": (f1 * support).sum() / cm.sum(),
            "macro_f1": f1.mean()}


def outcomes(rows, pred, baseline, mask=None):
    if mask is None:
        mask = pred != baseline
    correct, was_correct = pred == rows.label_id_dense, baseline == rows.label_id_dense
    rescued = int((mask & correct & ~was_correct).sum())
    harmed = int((mask & ~correct & was_correct).sum())
    return {"actions": int(mask.sum()), "rescues": rescued, "harms": harmed,
            "both_wrong": int((mask & ~correct & ~was_correct).sum()),
            "net_correct": rescued - harmed, "harm_rescue_ratio": harmed / rescued if rescued else None}


def compute(banks, saved_metrics, classification, labels):
    models = {"fusion": (4200, "raw_pred", "Fusion baseline"),
              "rf": (4200, "rf_pred", "Fusion + RF router")}
    for bank in (3378, 4200):
        for k in (5, 10):
            models[f"geometry_{bank}_k{k}"] = (bank, policy(k) + "_pred", f"Adapted geometry: {bank:,} references, K={k}")
    matrices, seed_matrices, summaries, classes, seeds, interventions = {}, {}, [], [], [], []
    for model, (bank, column, name) in models.items():
        rows = banks[bank]
        cms = np.stack([confusion(rows.loc[rows.training_seed.eq(seed)], column) for seed in SEEDS])
        cm = cms.sum(axis=0)
        matrices[model], seed_matrices[model] = cm, cms
        assert cm.sum() == 5388 and np.array_equal(cms.sum(axis=2), np.tile(cm.sum(axis=1) / 4, (4, 1)))
        per_seed = [rates(m) for m in cms]
        avg = {metric: float(np.mean([s[metric] for s in per_seed])) for metric in ("accuracy", "weighted_f1", "macro_f1")}
        saved_model = column.removesuffix("_pred") if model.startswith("geometry_") else model
        check = saved_metrics.loc[saved_metrics.cohort.eq(f"test_{bank}") & saved_metrics.model.eq(saved_model)].sort_values("training_seed")
        assert len(check) == 4
        for i, seed in enumerate(SEEDS):
            values = {key: float(per_seed[i][key]) for key in avg}
            for metric, value in values.items():
                np.testing.assert_allclose(value, check.iloc[i][metric], rtol=0, atol=1e-12)
            seeds.append({"model": model, "training_seed": seed, **values})
        if model in ("fusion", "rf"):
            old = classification["experts"]["raw"] if model == "fusion" else classification["policies"][RF_POLICY]
            assert np.array_equal(cm, old["pooled_seed_realizations"]["confusion_matrix"])
            for i, seed in enumerate(SEEDS):
                assert np.array_equal(cms[i], old["per_seed"][str(seed)]["confusion_matrix"])
            for metric, value in avg.items():
                np.testing.assert_allclose(value, old["seed_summary"][metric]["mean"], atol=1e-12, rtol=0)
        summaries.append({"model": model, "name": name, **avg})
        for label, habitat in enumerate(labels):
            classes.append({"model": model, "class_id": label, "habitat": habitat,
                            "unique_images": int(cm[label].sum() / 4),
                            "pooled_correct": int(cm[label, label]),
                            **{f"mean_{metric}": float(np.mean([s[metric][label] for s in per_seed]))
                               for metric in ("precision", "recall", "f1")}})
        if model != "fusion":
            interventions.append({"model": model, **outcomes(rows, rows[column], rows.raw_pred)})
    return models, matrices, seed_matrices, pd.DataFrame(summaries), pd.DataFrame(classes), pd.DataFrame(seeds), pd.DataFrame(interventions)


def combined_rules(rows, geometry_column, labels):
    """Evaluate fixed OR/AND actions on aligned, already scored predictions."""
    columns = KEY + ["file", "plot_idx", "label_id_dense", "canonical_l3_id",
                     "label_name", "raw_pred", "geo_pred", "rf_pred"]
    predictions = rows[columns].copy()
    predictions["geometry_pred"] = rows[geometry_column]
    for model in ("rf", "geometry"):
        acted = predictions[f"{model}_pred"].ne(predictions.raw_pred)
        predictions[f"{model}_acted"] = acted
        assert np.array_equal(predictions.loc[acted, f"{model}_pred"],
                              predictions.loc[acted, "geo_pred"])
    predictions["disagreement"] = predictions.raw_pred.ne(predictions.geo_pred)
    predictions["or_acted"] = predictions.rf_acted | predictions.geometry_acted
    predictions["and_acted"] = predictions.rf_acted & predictions.geometry_acted
    for rule in ("or", "and"):
        acted = predictions[f"{rule}_acted"]
        assert not (acted & ~predictions.disagreement).any()
        predictions[f"{rule}_pred"] = np.where(acted, predictions.geo_pred, predictions.raw_pred)
    assert int(predictions.or_acted.sum() + predictions.and_acted.sum()) == int(
        predictions.rf_acted.sum() + predictions.geometry_acted.sum())
    methods = {"fusion": ("raw_pred", "Fusion baseline"),
               "rf": ("rf_pred", "RF router"),
               "geometry": ("geometry_pred", "Adapted geometry router"),
               "or": ("or_pred", "OR: either router"),
               "and": ("and_pred", "AND: both routers")}
    summaries, seeds, classes, matrices = [], [], [], {}
    metric_names = ("accuracy", "weighted_f1", "macro_f1")

    def intervention_stats(part, column):
        counts = outcomes(part, part[column], part.raw_pred)
        disagreements = int(part.disagreement.sum())
        return {**counts, "test_cases": len(part), "disagreements": disagreements,
                "action_coverage": counts["actions"] / len(part) if len(part) else 0.0,
                "actions_per_disagreement": counts["actions"] / disagreements if disagreements else 0.0}

    for model, (column, name) in methods.items():
        cms, seed_rates = [], []
        for seed in SEEDS:
            part = predictions.loc[predictions.training_seed.eq(seed)]
            cm = confusion(part, column)
            cms.append(cm)
            result = rates(cm)
            seed_rates.append(result)
            counts = intervention_stats(part, column)
            assert int(np.trace(cm) - part.raw_pred.eq(part.label_id_dense).sum()) == counts["net_correct"]
            seeds.append({"model": model, "training_seed": seed,
                          **{key: float(result[key]) for key in metric_names}, **counts})
        avg = {key: float(np.mean([result[key] for result in seed_rates])) for key in metric_names}
        summaries.append({"model": model, "name": name, **avg,
                          **intervention_stats(predictions, column)})
        matrices[model] = {str(seed): cm.tolist() for seed, cm in zip(SEEDS, cms)}
        pooled_cm = np.stack(cms).sum(axis=0)
        for label, habitat in enumerate(labels):
            part = predictions.loc[predictions.label_id_dense.eq(label)]
            classes.append({"model": model, "class_id": label, "habitat": habitat,
                            "unique_images": part.row_uid.nunique(),
                            "pooled_correct": int(pooled_cm[label, label]),
                            **{f"mean_{key}": float(np.mean([result[key][label] for result in seed_rates]))
                               for key in ("precision", "recall", "f1")},
                            **intervention_stats(part, column)})
    return (pd.DataFrame(summaries), pd.DataFrame(seeds), pd.DataFrame(classes),
            predictions, matrices)


def draw_cm(cm, label_list, sink=None):
    """Canonical CS raw/normalized plotting behavior with a local figure sink."""
    if not isinstance(cm, np.ndarray):
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(cm, torch.Tensor):
            cm = cm.detach().cpu().numpy()
        else:
            try:
                cm = np.asarray(cm)
            except Exception:
                return
    assert cm.shape == (len(label_list), len(label_list))
    support = cm.sum(axis=1, keepdims=True)
    for normalized in (False, True):
        values = cm / np.where(support == 0, 1, support) if normalized else cm
        annot = np.array([["0" if v == 0 else f"{v:.2f}" for v in row] for row in values]) if normalized else values.astype(int)
        fig, ax = plt.subplots(figsize=(15, 12))
        sns.heatmap(values, annot=annot, fmt="" if normalized else "d", cmap="Blues",
                    xticklabels=label_list, yticklabels=label_list, ax=ax, vmin=0,
                    vmax=1 if normalized else None, annot_kws={"fontsize": 7})
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix L3" + (" (Normalized)" if normalized else ""))
        fig.tight_layout()
        if sink is not None:
            sink(fig, values, normalized)
        plt.close(fig)


def figures(output, models, matrices, summary, labels, main_model):
    root = output / "figures"
    root.mkdir(exist_ok=True)
    support = matrices["fusion"].sum(axis=1)
    ticks = [name + (" [no test examples]" if support[i] == 0 else "") for i, name in enumerate(labels)]
    stats = summary.set_index("model")
    for model, cm in matrices.items():
        def sink(fig, values, normalized, model=model):
            fig.suptitle(models[model][2], fontsize=15, y=1.005)
            suffix = "normalized" if normalized else "counts"
            fig.savefig(root / f"{model}_{suffix}.png", dpi=130, bbox_inches="tight")
        draw_cm(cm, ticks, sink)
    row_labels = [f"{i}: " + "\n".join(textwrap.wrap(name, 25)) + ("\n[no test examples]" if support[i] == 0 else "") for i, name in enumerate(labels)]
    norm = {key: value / np.where(support[:, None] == 0, 1, support[:, None]) for key, value in matrices.items()}
    for difference in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(30, 11))
        keys = ("fusion", "rf", main_model)
        pairs = (("rf", "fusion"), (main_model, "fusion"), (main_model, "rf"))
        for j, ax in enumerate(axes):
            if difference:
                a, b = pairs[j]
                values = 100 * (norm[a] - norm[b])
                title = f"{models[a][2]}\nminus {models[b][2]}"
                opts = dict(cmap="RdBu_r", center=0, vmin=-15, vmax=15)
            else:
                key = keys[j]
                values = norm[key]
                title = f"{models[key][2]}\nAcc@1 {100 * stats.loc[key, 'accuracy']:.2f}%"
                opts = dict(cmap="Blues", vmin=0, vmax=1)
            sns.heatmap(values, ax=ax, cbar=False, square=True,
                        xticklabels=list(range(18)), yticklabels=row_labels if j == 0 else False, **opts)
            for r in range(18):
                for c in range(18):
                    v = values[r, c]
                    if abs(v) >= (0.1 if difference else 0.005) or (r == c and support[r] > 0):
                        ax.text(c + 0.5, r + 0.5, f"{v:+.1f}" if difference else f"{v:.2f}",
                                ha="center", va="center", fontsize=7,
                                color="white" if abs(v) > (8 if difference else 0.55) else "#172033")
            ax.set(title=title, xlabel="Predicted habitat (ID matches row labels)", ylabel="True habitat" if j == 0 else "")
        caption = ("Cell change (pp): red = increased, blue = decreased. Increased diagonal is better; increased off-diagonal is worse. Colour limits ±15 pp; annotations exact."
                   if difference else "Row-normalized confusion matrices; shared 0–1 colour scale. Four seeds on the same 1,347 images; not an ensemble.")
        fig.suptitle(caption, fontsize=14, y=0.96)
        # Reserve a separate colour bar so all three matrix axes stay equal.
        fig.subplots_adjust(left=0.14, right=0.955, bottom=0.08, top=0.87, wspace=0.06)
        color_ax = fig.add_axes([0.968, 0.14, 0.009, 0.67])
        fig.colorbar(axes[-1].collections[0], cax=color_ax,
                     extend="both" if difference else "neither")
        stem = "confusion_differences" if difference else "confusion_comparison"
        for extension in ("png", "pdf"):
            fig.savefig(root / f"{stem}.{extension}", dpi=150, bbox_inches="tight")
        plt.close(fig)
    assert not plt.get_fignums()


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def relative_link(path, output):
    import os
    return Path(os.path.relpath(path, output)).as_posix()


def report(output, models, matrices, summary, classes, interventions, intervals, policies, banks, main_model, overlap, combined):
    stats, per_class = summary.set_index("model"), classes.set_index(["model", "class_id"])
    bank, column, main_name = models[main_model]
    k = policies[column.removesuffix("_pred")]["k"]
    threshold = policies[policy(k)]["threshold_display"]
    main_keys = ("fusion", "rf", main_model)
    scores = table(["Method", "Acc@1 (%)", "Weighted F1", "Macro F1"],
                   [[models[key][2], f"{100 * stats.loc[key, 'accuracy']:.4f}", f"{stats.loc[key, 'weighted_f1']:.5f}", f"{stats.loc[key, 'macro_f1']:.5f}"] for key in main_keys])
    support = matrices["fusion"].sum(axis=1) // 4
    labels = [per_class.loc[("fusion", i), "habitat"] for i in range(18)]
    recalls = table(["True habitat", "Images", "Fusion recall (%)", "RF recall (%)", "Geometry recall (%)", "Geometry − fusion (pp)", "Geometry − RF (pp)"],
                    [[labels[i], support[i], *[f"{100 * per_class.loc[(key, i), 'mean_recall']:.2f}" for key in main_keys],
                      f"{100 * (per_class.loc[(main_model, i), 'mean_recall'] - per_class.loc[('fusion', i), 'mean_recall']):+.2f}",
                      f"{100 * (per_class.loc[(main_model, i), 'mean_recall'] - per_class.loc[('rf', i), 'mean_recall']):+.2f}"] for i in range(18) if support[i]])
    # Explicit bidirectional habitat confusions discussed in the parent report.
    pairs = [(4, 5), (5, 4), (9, 11), (11, 9), (11, 7), (7, 11), (7, 10), (10, 5), (6, 5)]
    directions = table(["True → predicted habitat", "Fusion", "RF", "Geometry"],
                       [[f"{labels[a]} → {labels[b]}", *[f"{matrices[key][a,b]} ({100 * matrices[key][a,b] / (4 * support[a]):.2f}%)" for key in main_keys]] for a, b in pairs])
    action_rows = interventions.set_index("model")
    actions = table(["Correction", "Actions", "Rescues R", "Harms H", "Both wrong", "Net R−H", "Observed H/R"],
                    [[models[key][2], *[int(action_rows.loc[key, field]) for field in ("actions", "rescues", "harms", "both_wrong", "net_correct")], f"{action_rows.loc[key, 'harm_rescue_ratio']:.3f}"] for key in ("rf", main_model)])
    overlap_table = table(["Action subset", "Actions", "Rescues", "Harms", "Both wrong", "Net"],
                          [[name, *[row[field] for field in ("actions", "rescues", "harms", "both_wrong", "net_correct")]] for name, row in overlap.items()])
    combined_stats = combined.set_index("model")
    combined_scores = table(["Method", "Acc@1 (%)", "Weighted F1", "Macro F1"],
                            [[row['name'], f"{100 * row['accuracy']:.4f}", f"{row['weighted_f1']:.5f}", f"{row['macro_f1']:.5f}"]
                             for _, row in combined.iterrows()])
    combined_actions = table(["Correction", "Actions", "% of disagreements", "Rescues", "Harms", "Both wrong", "Net R−H"],
                             [[row['name'], int(row['actions']), f"{100 * row['actions_per_disagreement']:.2f}",
                               *[int(row[field]) for field in ("rescues", "harms", "both_wrong", "net_correct")]]
                              for _, row in combined.loc[combined.model.ne("fusion")].iterrows()])
    union, intersection = combined_stats.loc["or"], combined_stats.loc["and"]
    combined_differences = "; ".join(
        f"{row['name']} − geometry: accuracy {100 * (row['accuracy'] - combined_stats.loc['geometry', 'accuracy']):+.4f} pp, "
        f"weighted F1 {row['weighted_f1'] - combined_stats.loc['geometry', 'weighted_f1']:+.5f}, "
        f"macro F1 {row['macro_f1'] - combined_stats.loc['geometry', 'macro_f1']:+.5f}"
        for row in (union, intersection))
    sensitivity = table(["Reference images", "Strict K", "Acc@1 (%)", "Weighted F1", "Macro F1", "Rescues", "Harms"],
                        [[b, kk, f"{100 * stats.loc[f'geometry_{b}_k{kk}', 'accuracy']:.4f}", f"{stats.loc[f'geometry_{b}_k{kk}', 'weighted_f1']:.5f}", f"{stats.loc[f'geometry_{b}_k{kk}', 'macro_f1']:.5f}", int(action_rows.loc[f'geometry_{b}_k{kk}', 'rescues']), int(action_rows.loc[f'geometry_{b}_k{kk}', 'harms'])] for b in (3378, 4200) for kk in (5, 10)])
    selected_intervals = intervals.loc[intervals.cohort.eq(f"test_{bank}") & intervals.target.eq(policy(k)) & intervals.baseline.isin(["fusion", "rf"]) & intervals.metric.eq("accuracy")]
    ci_table = table(["Comparison", "Accuracy difference (pp)", "Conditional 95% interval (pp)"],
                    [[f"Geometry − {row.baseline}", f"{100 * row.point_estimate:+.3f}", f"[{100 * row.lower:+.3f}, {100 * row.upper:+.3f}]"] for row in selected_intervals.itertuples()])
    # Rank diagonal movements descriptively, without significance claims.
    delta = np.diag(matrices[main_model] - matrices["fusion"])
    gains = sorted(np.flatnonzero(delta > 0), key=lambda i: -delta[i])
    losses = sorted(np.flatnonzero(delta < 0), key=lambda i: delta[i])
    gain_text = ", ".join(f"{labels[i]} ({int(delta[i]):+d})" for i in gains)
    loss_text = ", ".join(f"{labels[i]} ({int(delta[i]):+d})" for i in losses)
    f1_delta = np.array([per_class.loc[(main_model, i), "mean_f1"] - per_class.loc[("fusion", i), "mean_f1"] for i in range(18)])
    f1_losses = sorted(np.flatnonzero(f1_delta < -1e-12), key=lambda i: f1_delta[i])
    f1_text = ", ".join(f"{labels[i]} ({f1_delta[i]:+.3f})" for i in f1_losses)
    precision_table = table(["Habitat", "Fusion precision / recall / F1", "RF precision / recall / F1", "Geometry precision / recall / F1"],
                            [[labels[i], *[f"{100 * per_class.loc[(key, i), 'mean_precision']:.2f}% / {100 * per_class.loc[(key, i), 'mean_recall']:.2f}% / {per_class.loc[(key, i), 'mean_f1']:.3f}" for key in main_keys]] for i in (7, 10, 11)])
    calcareous = banks[bank].loc[banks[bank].label_id_dense.eq(6) & banks[bank].raw_pred.eq(6)
                                 & banks[bank][column].ne(6)]
    rare_note = (f"Calcareous Grassland has only {support[6]} test photos. Its {len(calcareous)} lost correct seed–image predictions "
                 f"come from {calcareous.row_uid.nunique()} unique photo(s) in {calcareous.plot_idx.nunique()} plot(s), "
                 "repeated across seeds; they are not independent examples. Strict reference support alone does not prevent this harm.")
    habitat_sensitivity = table(["Habitat", "Fusion recall (%)", "RF recall (%)", "Geometry recall range across four configurations (%)"],
                                [[labels[i], f"{100 * per_class.loc[('fusion', i), 'mean_recall']:.2f}", f"{100 * per_class.loc[('rf', i), 'mean_recall']:.2f}",
                                  f"{100 * classes.loc[classes.model.str.startswith('geometry_') & classes.class_id.eq(i), 'mean_recall'].min():.2f}–{100 * classes.loc[classes.model.str.startswith('geometry_') & classes.class_id.eq(i), 'mean_recall'].max():.2f}"] for i in (11, 7, 4, 5, 10, 6)])
    links = " · ".join(f"[{models[key][2]}](figures/{key}_normalized.png)" for key in models)
    content = f"""# Geo-helpfulness router: adapted geometry versus RF and fusion

This report extends the [September 15 progress summary]({relative_link(PARENT, output)}) with a habitat-level comparison of adapted geometry, its existing RF router, and the unchanged fusion baseline. It reuses already scored predictions for the same **1,347 cleaned-test images from 531 plots**, across **four training seeds**. No expert or router is retrained and no threshold is selected here.

## 1. Adapted geometry raises accuracy but does not improve every habitat

The main illustration uses **{bank:,} development references, strict K={k}**, and the validation-selected **H≤0.7R, at least 20 acted plots** policy. This is a displayed configuration from the completed exploratory study, not a newly confirmed or promoted winner. All four bank/K configurations appear below.

{scores}

Geometry changes accuracy by **{100 * (stats.loc[main_model, 'accuracy'] - stats.loc['fusion', 'accuracy']):+.2f} percentage points versus fusion** and **{100 * (stats.loc[main_model, 'accuracy'] - stats.loc['rf', 'accuracy']):+.2f} points versus RF**. Its lower macro F1 shows that these aggregate gains do not amount to uniformly better class performance.

### Confusion matrices

![Fusion, RF, and adapted-geometry row-normalized confusion matrices](figures/confusion_comparison.png)

Rows are true habitats and columns are predictions. The diagonal is habitat recall; every panel uses the same 0–1 colour scale. The matrices sum four seed runs, giving **5,388 seed–image predictions**, not 5,388 independent images and not an ensemble. Because every seed sees the same test images, pooled row-normalized recalls equal the mean per-seed recalls. F1 in the tables is computed per seed and then averaged; it is not F1 recomputed from the pooled matrix. Absent-class rows are marked and contribute zero to the fixed 18-class macro F1.

Full-size matrices for all configurations: {links}. Raw count matrices use the corresponding `_counts.png` filename; all exact per-seed and pooled matrices are in [metrics.json](metrics.json).

## 2. The gains and losses concern different habitat confusions

Compared with fusion, geometry adds correct seed–image predictions for **{gain_text}**. It loses correct predictions for **{loss_text}**. These are net diagonal changes, not the separate rescue/harm counts in Section 3.

{recalls}

The main comparisons highlighted in the earlier report remain informative:

{directions}

Each cell reports **pooled error count (percentage of its true-habitat row)**. Opposite directions have different denominators. Improved–Neutral Grassland remains a trade-off: both corrections reduce Improved→Neutral errors while increasing Neutral→Improved errors. Geometry reduces Bog→Dwarf Shrub Heath confusion more strongly than RF, but increases Dwarf Shrub Heath→Bog errors relative to fusion. It also reduces Acid→Fen/Marsh/Swamp confusion while increasing Acid→Bog and Fen/Marsh/Swamp→Neutral errors. Thus geometry changes where errors occur; it does not generally separate all confusing habitat pairs.

![Changes in confusion-matrix cells](figures/confusion_differences.png)

Differences are percentage points of each true-habitat row. Red means a cell increased and blue means it decreased: an increase on the diagonal is beneficial, whereas an increase off the diagonal is an error increase. Small off-diagonal movements are easier to see here than in the original matrices. [All cell values and differences](confusion_cells.csv) retain exact counts and rates.

**Why macro F1 falls.** The largest per-class F1 reductions versus fusion are {f1_text}. Macro F1 gives every ontology class equal weight; weighted F1 gives more weight to habitats with more test images. Recall and F1 need not move together because routing also changes how many false positives enter each predicted habitat. For rare habitats, a few predictions can cause a large rate change. [Per-class precision, recall and F1](per_class.csv) includes all 18 habitats for all six methods.

{precision_table}

Acid Grassland improves in both precision and recall. Bog gains recall while precision falls slightly. Fen/Marsh/Swamp loses recall but gains precision; its F1 is lower than fusion's, yet slightly higher than RF's. The recall matrix alone therefore does not establish which router has better F1 for a habitat.

{rare_note}

## 3. Geometry and RF intervene on substantially different cases

Rescue means fusion was wrong and the corrected prediction is right; harm means fusion was right and the correction is wrong. Both-wrong interventions change the predicted habitat without correcting the error.

{actions}

Counts pool the four seeds. Dividing net rescue by 5,388 reproduces each router's accuracy gain over fusion. The empirical test H/R is descriptive: a constraint fitted on development validation does not guarantee the same ratio on new queries.

{overlap_table}

Both methods replace fusion only with geo, so on jointly acted cases they make the same corrected prediction. The exclusive-action rows expose why their accuracies differ: geometry finds some rescues RF misses and also makes some harms RF avoids.

### 3.1 Combining the fixed routers with OR and AND

Added 2026-10-06. Let r and g be the intervention decisions from the saved RF and the displayed **{bank:,}-reference, strict-K{k}** geometry policy. **OR** switches fusion to geo when r OR g is true; **AND** switches only when r AND g is true. Otherwise the final prediction is fusion's. Neither rule uses the true label to decide, and both reuse the original component thresholds. OR accepts RF-only interventions even when geometry's support or score condition is not satisfied; AND requires both component decisions.

There are **{int(union['disagreements']):,} fusion–geo disagreements** across the four seeds. The union is **{int(action_rows.loc['rf', 'actions'])} + {int(action_rows.loc[main_model, 'actions'])} − {int(intersection['actions'])} = {int(union['actions'])}** interventions. The intersection has **{int(intersection['actions'])}** interventions. Overlap and outcome counts refer to matched seed–image decisions, not a union of images across different seeds.

{combined_scores}

{combined_actions}

{combined_differences}. OR adds the RF-only decisions to geometry: **{overlap['RF only']['rescues']} additional rescues and {overlap['RF only']['harms']} additional harms**, for **{overlap['RF only']['net_correct']:+d}** net correct decisions. AND removes geometry-only decisions: **{overlap['Geometry only']['harms']} harms avoided and {overlap['Geometry only']['rescues']} rescues lost**. Requiring both routers still leaves **{int(intersection['harms'])} harms**.

These two combined rules were proposed after examining cleaned-test overlap. Their results are exploratory; the combinations have not been selected or confirmed on development validation, and the component validation harm constraints do not automatically transfer to a combination. No uncertainty interval is claimed for these new comparisons. F1 remains the mean of four per-seed scores over the fixed 18-class ontology.

[Combined-rule summary](combined_rules_summary.csv) · [Per-seed performance and interventions](combined_rules_per_seed.csv) · [Per-habitat performance and interventions](combined_rules_per_class.csv) · [Matched predictions and actions](combined_rules_predictions.parquet). The `combined_rules` entry in [metrics.json](metrics.json) records the exact component configuration, rule definitions and per-seed confusion matrices. These outputs follow `--reference-bank` and `--k`, with the requested 4,200/K5 comparison as the default.

## 4. The other adapted-geometry configurations retain the same aggregate trade-off

{sensitivity}

Every row uses the same development-validation threshold-selection guard (H≤0.7R, at least 20 acted plots), 1% minimum coverage, and positive net rescue. The saved thresholds are **{policies[policy(5)]['threshold_display']:.9f} for K=5** and **{policies[policy(10)]['threshold_display']:.9f} for K=10** and transfer unchanged to both reference banks. In the adapted study all nine guard combinations at each K selected identical predictions; these are four distinct bank/K comparisons, not 36 independent improvements. Every adapted row has higher observed accuracy and lower macro F1 than both baselines. [Summary](summary.csv), [per-seed metrics](per_seed_metrics.csv), and the full-size matrices above retain all four configurations.

{habitat_sensitivity}

The principal Bog/Acid gains and Fen/Neutral/Calcareous losses relative to fusion persist across both reference banks and both neighborhood sizes.

## 5. Interpretation, provenance and reproduction

{ci_table}

These intervals are copied from the authenticated completed adapted study: 2,000 habitat-stratified whole-plot bootstrap draws retain all photos and four seed realizations together. They condition on the fitted experts, embeddings and selected thresholds; thresholds are not refitted within each draw. Both displayed accuracy intervals include zero. Habitat-specific differences in this report are descriptive, without separate significance tests. The test set has already informed several exploratory comparisons, so these results do not establish independent superiority.

The expert heads are the September 15 `best_model` family (fixed epoch 50, fitted on all 4,200 development images), not the test-selected `oracle_test_best_model` family. The geometry image towers were fitted on **3,378 development-training images**; the same seed-specific tower embeds references and queries. Using 4,200 reference images expands the labeled bank without refitting the encoder. Thresholds were fitted on the 822 development-validation images, with development-train references and development-train expert heads.

RF is `{RF_POLICY}`. Geometry is `{policy(k)}` with score S=min(A_image,A_geo) and strict **S>{threshold:.9f}** on fusion/geo disagreements. Each class distance averages cosine distances to K distinct reference plots after separate L2 normalization of the two modalities; each plot contributes its closest photo. Insufficient support for either candidate class retains fusion. The saved actions are checked against this frozen rule.

The generator authenticates every consumed scientific table against its saved receipt/manifest, matches row identities, truth, fusion and RF predictions against the historical scored RF rows, checks class ordering against the CS mapping, and reproduces the archived per-seed metrics and RF/fusion confusion matrices. [Provenance](provenance.json) records source/code hashes and confirms the registry and consumed source files remain unchanged. This is a derived review of existing scored predictions, not a new evaluation event; no labels are newly joined to unscored predictions.

From the repository root, using the existing `habcls` environment:

```bash
python tools/plot_multimodal_geo_helpfulness_confusion_review.py --reference-bank {bank} --k {k}
```

This regenerates figures, CSV tables, `metrics.json`, `provenance.json`, and this report in the canonical review directory. It requires the existing adapted and RF artifact bundles and the parent report; no photos, GPU or encoder inference are needed. To reproduce in a fresh destination:

```bash
python tools/plot_multimodal_geo_helpfulness_confusion_review.py --reference-bank {bank} --k {k} \\
  --output-dir multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/geometry_router_confusion_reproduction_v1
```

The main figure can be changed using `--reference-bank 3378` or `--k 10`; all four adapted configurations remain exported. Register any retained reproduction directory in the study catalog. [Original adapted study](../joint_geometry_adapted_encoder_v1/report.md) describes the earlier evaluation; [development record]({relative_link(REPO / 'dev_plans/reliability_aware_geo_fusion/23_geometry_router_confusion_review.md', output)}) indexes this review.
"""
    (output / "report.md").write_text(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", default=str(BASE / STUDY))
    parser.add_argument("--reference-bank", type=int, choices=(3378, 4200), default=4200)
    parser.add_argument("--k", type=int, choices=(5, 10), default=5)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output = (REPO / output).resolve() if not output.is_absolute() else output.resolve()
    if output.parent != BASE.resolve() or not output.name.startswith("geometry_router_confusion_"):
        parser.error("Output must be a geometry_router_confusion_* review directory directly under the canonical artifact root")
    if output.exists() and any(output.iterdir()):
        if not (output / "provenance.json").exists() or read_json(output / "provenance.json").get("study_id") != STUDY:
            parser.error("Refusing to overwrite a directory not owned by this review")
    registry = REPO / "multimodal_artifacts/locked_test_registry/cs/gse_100m_cleaned_test/events"
    registry_before = {str(p.relative_to(REPO)): digest(p) for p in registry.rglob("*") if p.is_file()}
    banks, policies, labels, ontology, saved, historical, intervals, hashes = load_sources()
    models, matrices, seed_matrices, summary, classes, seeds, interventions = compute(banks, saved, historical, labels)
    main_model = f"geometry_{args.reference_bank}_k{args.k}"
    rows = banks[args.reference_bank]
    geometry_pred, rf_pred = rows[policy(args.k) + "_pred"], rows.rf_pred
    ga, ra = geometry_pred.ne(rows.raw_pred), rf_pred.ne(rows.raw_pred)
    overlap = {"Both routers": outcomes(rows, geometry_pred, rows.raw_pred, ga & ra),
               "Geometry only": outcomes(rows, geometry_pred, rows.raw_pred, ga & ~ra),
               "RF only": outcomes(rows, rf_pred, rows.raw_pred, ra & ~ga)}
    combined, combined_seeds, combined_classes, combined_predictions, combined_matrices = combined_rules(
        rows, policy(args.k) + "_pred", labels)
    for model, original in (("fusion", "fusion"), ("rf", "rf"), ("geometry", main_model)):
        for metric in ("accuracy", "weighted_f1", "macro_f1"):
            np.testing.assert_allclose(combined.set_index("model").loc[model, metric],
                                       summary.set_index("model").loc[original, metric], rtol=0, atol=1e-12)
    output.mkdir(exist_ok=True)
    for name, frame in (("summary", summary), ("per_class", classes), ("per_seed_metrics", seeds), ("interventions", interventions)):
        frame.to_csv(output / f"{name}.csv", index=False)
    for name, frame in (("summary", combined), ("per_seed", combined_seeds), ("per_class", combined_classes)):
        frame.to_csv(output / f"combined_rules_{name}.csv", index=False)
    combined_predictions.to_parquet(output / "combined_rules_predictions.parquet", index=False)
    cells = []
    for i in range(18):
        for j in range(18):
            n = matrices["fusion"][i].sum()
            row = {"true_id": i, "true_habitat": labels[i], "predicted_id": j, "predicted_habitat": labels[j], "unique_images": int(n / 4)}
            for model, cm in matrices.items():
                row[model + "_count"] = int(cm[i, j])
                row[model + "_row_rate"] = float(cm[i, j] / n) if n else 0.0
                for baseline in ("fusion", "rf"):
                    row[f"{model}_minus_{baseline}_pp"] = float(100 * (cm[i,j] - matrices[baseline][i,j]) / n) if n else 0.0
            cells.append(row)
    pd.DataFrame(cells).to_csv(output / "confusion_cells.csv", index=False)
    write_json(output / "metrics.json", {"study_id": STUDY, "main_model": main_model,
               "aggregation": "Mean per-seed metrics; matrices pool seeds, not ensemble predictions",
               "unique_images": 1347, "unique_plots": 531, "seed_image_rows": 5388,
               "ontology": ontology, "summary": summary.to_dict("records"),
               "confusion_matrices": {key: value.tolist() for key, value in matrices.items()},
               "per_seed_confusion_matrices": {key: {str(seed): value[i].tolist() for i, seed in enumerate(SEEDS)} for key, value in seed_matrices.items()},
               "action_overlap": overlap,
               "combined_rules": {"geometry_model": main_model, "reference_bank": args.reference_bank,
                                  "k": args.k, "rf_policy": RF_POLICY, "geometry_policy": policy(args.k),
                                  "definitions": {"or": "Switch to geo if RF or geometry intervenes; otherwise retain fusion",
                                                  "and": "Switch to geo if both RF and geometry intervene; otherwise retain fusion"},
                                  "proposed_after_test_overlap_inspection": True,
                                  "summary": combined.astype(object).where(pd.notna(combined), None).to_dict("records"),
                                  "per_seed_confusion_matrices": combined_matrices}})
    figures(output, models, matrices, summary, labels, main_model)
    report(output, models, matrices, summary, classes, interventions, intervals, policies, banks, main_model, overlap, combined)
    registry_after = {str(p.relative_to(REPO)): digest(p) for p in registry.rglob("*") if p.is_file()}
    assert registry_before == registry_after
    assert all(digest(REPO / name) == expected for name, expected in hashes.items())
    code = [Path(__file__), REPO / "tools/plot_multimodal_geo_helpfulness_confusion_review.py"]
    provenance = {"study_id": STUDY, "source_sha256": hashes,
                  "implementation_sha256": {str(p.relative_to(REPO)): digest(p) for p in code},
                  "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                  "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                                  "seaborn": sns.__version__},
                  "main_model": main_model, "rf_policy": RF_POLICY,
                  "geometry_policies": {policy(k): policies[policy(k)] for k in (5, 10)},
                  "registry_files": len(registry_before), "registry_unchanged": True,
                  "source_files_unchanged": True, "independent_confirmation": False,
                  "model_fitting": False, "threshold_selection": False,
                  "combined_rules": ["or", "and"], "combined_rules_proposed_after_test_overlap_inspection": True,
                  "output_sha256": {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file() and p.name != "provenance.json"}}
    write_json(output / "provenance.json", provenance)
    print(summary.to_string(index=False))
    print(f"Report: {output / 'report.md'}")
    return 0
