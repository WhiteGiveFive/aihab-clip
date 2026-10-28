"""Selection, artifact authentication and whole-plot uncertainty contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from multimodal.attention_fusion import GeoPreprocessor
from multimodal.attention_fusion_comparison import (
    ComparisonError, METRICS, analyze_study, authenticate_runs, compare_selected,
    declare_contrasts, freeze_contrasts, interpretation_category,
    load_reference_inputs, paired_plot_bootstrap, select_learning_rates,
)
from multimodal.attention_fusion_training import (
    _array_hash, _source_hashes, _validated_data, classification_metrics, configuration_hash, sha256_file,
)
from tools.prepare_attention_fusion_protocol import build_matrix


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reseal(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["files"] = {p.name: sha256_file(p) for p in directory.iterdir()
                         if p.is_file() and p.name != "manifest.json"}
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = configuration_hash(manifest)
    write_json(directory / "manifest.json", manifest)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    repo = tmp_path_factory.mktemp("comparison")
    config = yaml.safe_load((ROOT / "configs/multimodal_attention_fusion.yaml").read_text())
    matrix = build_matrix(config)
    config["data"].update(train_rows=8, train_plots=4, validation_rows=6,
                          validation_plots=3, image_dim=4, geo_dim=2, geo_columns=["A00", "A01"],
                          num_classes=4, dense_to_canonical_l3=[0, 1, 2, 4])
    config["training"].update(epochs=2, checkpoint_epoch=2)
    config["metrics"]["bootstrap"]["replicates"] = 31
    for candidate in config["candidates"]:
        candidate["expected_trainable_parameters"] = 2
    for fit in matrix["development_fits"]:
        fit.update(epochs=2, checkpoint_epoch=2, fit_rows=8)
    declaration_path = repo / "contrasts.json"
    freeze_contrasts(config, matrix, declaration_path, repo_root=repo)
    arrays = {}
    for role, n in (("train", 8), ("validation", 6)):
        values = {"row_uid": np.array([f"{role}_{i:03d}" for i in range(n)]),
                  "plot_idx": np.array([f"{role}_p{i//2}" for i in range(n)]),
                  "label": np.arange(n, dtype=np.int64) % 3,
                  "image_embedding": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
                  "geo_embedding": np.random.default_rng(42).normal(size=(n, 2))}
        arrays[role] = values
    references = {}
    for seed in config["data"]["training_seeds"]:
        for kind in ("image", "geo_only"):
            data = copy.deepcopy(arrays)
            if kind == "geo_only":
                for values in data.values():
                    values["image_embedding"].fill(0)
                    values["geo_embedding"] = np.asfortranarray(values["geo_embedding"])
            references[(seed, kind)] = {"arrays": data, "provenance": {"seed": seed, "kind": kind},
                "input_hashes": {role: {k: _array_hash(v) for k, v in sorted(values.items())}
                                 for role, values in data.items()}}
    environment = {"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8", "device": "cpu"}
    for index, fit in enumerate(matrix["development_fits"]):
        directory = repo / fit["output_dir"]
        directory.mkdir(parents=True)
        ref = references[(fit["seed"], "geo_only" if fit["candidate_id"] == "geo_only" else "image")]
        spec = {"config_sha256": configuration_hash(config), "study_id": config["study_id"],
                "regime": config["regime"], "candidate_id": fit["candidate_id"], "geo_mode": fit["geo_preprocessing"],
                "learning_rate": fit["learning_rate"], "seed": fit["seed"], "fit_role": "train",
                "verification": False, "epochs": 2, "source_hashes": _source_hashes(),
                "input_hashes": ref["input_hashes"], "input_provenance": ref["provenance"], "environment": environment}
        identity = configuration_hash(spec)
        run = {"run_identity_sha256": identity, "completed_epoch": 2, "status": "complete", "verification": False,
               "trainable_parameters": 2, "elapsed_seconds": 0.5, "peak_cuda_bytes": None,
               "peak_process_rss_bytes": 1000, "peak_process_rss_scope": "process_lifetime_high_water_not_isolated_run_peak"}
        metrics = {}
        for role, values in ref["arrays"].items():
            rng = np.random.default_rng(index)
            logits = rng.normal(size=(len(values["label"]), 4)).astype(np.float32)
            metrics[role], predictions = classification_metrics(values["label"], logits)
            np.savez_compressed(directory / f"{role}_predictions.npz",
                                **{k: values[k] for k in ("row_uid", "plot_idx", "label")}, **predictions)
        history = [{"epoch": i, "optimization_loss": 2.0 / i, **metrics} for i in (1, 2)]
        processor = GeoPreprocessor("native" if fit["geo_preprocessing"] == "not_applicable" else fit["geo_preprocessing"],
                                    config["data"]["geo_columns"]).fit(ref["arrays"]["train"]["geo_embedding"], ref["arrays"]["train"]["row_uid"])
        for name, value in (("config.json", config), ("run_spec.json", spec), ("run.json", run),
                            ("metrics.json", metrics), ("history.json", history),
                            ("geo_preprocessing.json", processor.to_dict())):
            write_json(directory / name, value)
        torch.save({"epoch": 2, "model_state": {"weight": torch.ones(2)}}, directory / "final.pt")
        torch.save({"completed_epoch": 2}, directory / "resume.pt")
        manifest = {**run, "class_map": [0, 1, 2, 4],
                    "boundary": {"test_data_loaded": False, "encoder_executed": False}}
        write_json(directory / "manifest.json", manifest)
        reseal(directory)
    audit_dir = repo / config["paths"]["artifact_root"] / "protocol"
    audit_dir.mkdir(parents=True)
    write_json(audit_dir / "input_audit.json", {"assignments": {"class_map": [
        {"label_id_dense": i, "canonical_l3_id": canonical, "label_name": f"Habitat {canonical}"}
        for i, canonical in enumerate([0, 1, 2, 4])]}})
    return repo, config, matrix, references, declaration_path


@pytest.fixture(scope="module")
def records(study):
    repo, config, matrix, references, _ = study
    return authenticate_runs(config, matrix, references, repo_root=repo)


def test_declared_contrasts_are_complete_unique_and_primary_is_fixed(study):
    _, config, matrix, _, _ = study
    result = declare_contrasts(config, matrix)
    assert len(result["contrasts"]) == 63
    primary = [c for c in result["contrasts"] if c["kind"] == "primary"]
    assert len(primary) == 1
    assert primary[0]["lhs_arm_id"] == "attention_two_token__geo_native"
    assert primary[0]["rhs_arm_id"] == "projected_concat__geo_native"
    assert len([c for c in result["contrasts"] if c["kind"] == "preprocessing"]) == 12


def test_declaration_is_immutable_and_cannot_be_created_after_fitting(study):
    repo, config, matrix, _, path = study
    before = path.read_bytes()
    freeze_contrasts(config, matrix, path, repo_root=repo)
    assert path.read_bytes() == before
    with pytest.raises(ComparisonError, match="before development fits"):
        freeze_contrasts(config, matrix, repo / "late.json", repo_root=repo)
    changed = copy.deepcopy(config)
    changed["interpretation"]["minimum_practical_effect"] = 0.1
    with pytest.raises(ComparisonError, match="declaration differs"):
        freeze_contrasts(changed, matrix, path, repo_root=repo)


def test_complete_matrix_authentication_and_selection(study, records):
    _, config, matrix, _, _ = study
    assert len(records) == 228
    summaries, selected = select_learning_rates(config, matrix, records)
    assert len(summaries) == 57 and len(selected) == 19
    assert sum(len(a["per_seed"]) for a in selected) == 76
    assert all(set(a["per_seed"]) == {"1", "2", "3", "4"} for a in selected)
    for a in selected:
        assert a["sample_std"]["acc1"] == pytest.approx(np.std(
            [s["metrics"]["acc1"] for s in a["per_seed"].values()], ddof=1))


def test_equal_geo_values_with_different_layouts_use_their_own_exact_fitted_state(study, records):
    repo, _, matrix, references, _ = study
    image = references[(1, "image")]["arrays"]["train"]["geo_embedding"]
    geo = references[(1, "geo_only")]["arrays"]["train"]["geo_embedding"]
    assert image.flags.c_contiguous and geo.flags.f_contiguous
    assert np.array_equal(image, geo) and _array_hash(image) == _array_hash(geo)
    states = {}
    for candidate in ("geo_only", "raw_concat"):
        fit = next(f for f in matrix["development_fits"] if f["candidate_id"] == candidate
                   and f["geo_preprocessing"] == "standardize" and f["seed"] == 1)
        states[candidate] = json.loads((repo / fit["output_dir"] / "geo_preprocessing.json").read_text())
    assert states["geo_only"]["fit_row_sha256"] == states["raw_concat"]["fit_row_sha256"]
    assert states["geo_only"] != states["raw_concat"]
    assert np.allclose(states["geo_only"]["mean"], states["raw_concat"]["mean"], atol=1e-15, rtol=0)
    # Authentication must pass with exact own-producer states, not a relaxed
    # tolerance or a memoized state borrowed from the first candidate.
    assert len(records) == 228


def test_reference_loader_matches_runner_bindings_and_trainer_dtype_hashes(study, monkeypatch):
    import multimodal.attention_fusion_inputs as inputs

    _, config, _, references, _ = study
    source = copy.deepcopy(references[(1, "image")]["arrays"])
    for role, values in source.items():
        values["image_embedding"] = values["image_embedding"].astype(np.float64)
        values["geo_embedding"] = values["geo_embedding"].astype(np.float32)
        values["label"] = values["label"].astype(np.int32)
        values["plot_idx"] = np.arange(len(values["label"]), dtype=np.int32)
        values["row_uid"] = values["row_uid"].astype("S32")
    provenance = {"protocol_config_sha256": "stale", "train_role": "not_bound", "native_geo": "example"}
    monkeypatch.setattr(inputs, "load_prepared_inputs", lambda config, seed, role, **kwargs: copy.deepcopy(source[role]))
    monkeypatch.setattr(inputs, "load_input_provenance", lambda *args, **kwargs: provenance.copy())

    def geo_loader(*args, **kwargs):
        values = copy.deepcopy(source)
        for arrays in values.values():
            arrays["image_embedding"].fill(0)
        return values["train"], values["validation"], {"image_placeholder": "zeros_unused_by_geo_only"}

    monkeypatch.setattr(inputs, "load_geo_only_inputs", geo_loader)
    loaded = load_reference_inputs(config)
    for (seed, kind), reference in loaded.items():
        assert reference["provenance"]["protocol_config_sha256"] == configuration_hash(config)
        assert reference["provenance"]["train_role"] == "train"
        assert reference["provenance"]["validation_role"] == "validation"
        for role in source:
            expected = _validated_data(source[role], config, role)
            if kind == "geo_only":
                expected["image_embedding"].fill(0)
            assert reference["input_hashes"][role] == {key: _array_hash(value) for key, value in sorted(expected.items())}
    assert provenance["protocol_config_sha256"] == "stale"  # Do not mutate loader state.


@pytest.mark.parametrize("violation", ["missing_seed", "duplicate_seed", "missing_lr", "duplicate_fit"])
def test_incomplete_or_duplicate_accounting_never_selects(study, records, violation):
    _, config, matrix, _, _ = study
    changed = copy.deepcopy(matrix)
    supplied = list(records)
    if violation == "missing_seed":
        supplied = supplied[1:]
    elif violation == "duplicate_seed":
        supplied = [supplied[1], *supplied[1:]]
    elif violation == "missing_lr":
        changed["development_fits"] = changed["development_fits"][1:]
    else:
        changed["development_fits"][0] = changed["development_fits"][1]
    with pytest.raises(ComparisonError):
        select_learning_rates(config, changed, supplied)


@pytest.mark.parametrize("winner_metric", ["acc1", "weighted_f1", "mcc", "smaller_lr"])
def test_unrounded_global_lr_selection_and_tie_order(study, records, winner_metric):
    _, config, matrix, _, _ = study
    supplied = copy.deepcopy(records)
    arm_id = matrix["candidate_geo_arms"][0]["arm_id"]
    for r in supplied:
        if r["arm_id"] == arm_id:
            r["metrics"].update(acc1=0.5, weighted_f1=0.4, mcc=0.2, acc3=0.8)
            # One seed prefers the largest LR, but the across-seed mean does not.
            if winner_metric == "acc1" and r["learning_rate"] == 0.001:
                r["metrics"]["acc1"] = 0.9 if r["seed"] == 1 else 0.1
            if winner_metric != "smaller_lr" and r["learning_rate"] == 0.0003:
                r["metrics"][winner_metric] += 1 / 6 if winner_metric == "acc1" else 1e-10
    _, selected = select_learning_rates(config, matrix, supplied)
    winner = next(a for a in selected if a["arm_id"] == arm_id)
    assert winner["selected_learning_rate"] == (0.0001 if winner_metric == "smaller_lr" else 0.0003)


@pytest.mark.parametrize("violation", ["checksum", "predictions", "metrics", "inputs", "verification", "epoch", "scaler"])
def test_authentication_rejects_tampering_even_with_resealed_payloads(study, violation):
    repo, config, matrix, references, _ = study
    directory = repo / matrix["development_fits"][0]["output_dir"]
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    try:
        if violation in {"checksum", "metrics"}:
            metrics = json.loads((directory / "metrics.json").read_text())
            metrics["validation"]["acc1"] += 0.1
            write_json(directory / "metrics.json", metrics)
        elif violation == "predictions":
            with np.load(directory / "validation_predictions.npz", allow_pickle=False) as data:
                values = {k: data[k].copy() for k in data.files}
            values["row_uid"] = values["row_uid"][::-1]
            np.savez_compressed(directory / "validation_predictions.npz", **values)
        elif violation == "scaler":
            value = json.loads((directory / "geo_preprocessing.json").read_text())
            value["fit_row_count"] += 1
            write_json(directory / "geo_preprocessing.json", value)
        elif violation == "epoch":
            torch.save({"epoch": 1, "model_state": {"weight": torch.ones(2)}}, directory / "final.pt")
        else:
            spec = json.loads((directory / "run_spec.json").read_text())
            if violation == "inputs":
                spec["input_hashes"]["validation"]["label"] = "wrong"
            else:
                spec["verification"] = True
            write_json(directory / "run_spec.json", spec)
            for name in ("run.json", "manifest.json"):
                value = json.loads((directory / name).read_text())
                value["run_identity_sha256"] = configuration_hash(spec)
                write_json(directory / name, value)
        if violation != "checksum":
            reseal(directory)
        with pytest.raises(ValueError):
            authenticate_runs(config, matrix, references, repo_root=repo)
    finally:
        for name, data in original.items():
            (directory / name).write_bytes(data)


def test_bootstrap_matches_literal_whole_plot_resampling_for_all_metrics():
    labels = np.array([0, 1, 2, 0, 1, 2, 0])
    plots = np.array(["a", "b", "b", "c", "c", "c", "c"])
    rng = np.random.default_rng(97)
    logits = {arm: rng.normal(size=(4, len(labels), 4)).astype(np.float32) for arm in ("left", "right")}
    predictions = {arm: np.argsort(-values, axis=-1, kind="stable")[:, :, :3] for arm, values in logits.items()}
    contrasts = [{"contrast_id": "paired", "lhs_arm_id": "left", "rhs_arm_id": "right"}]
    intervals, draws, weights = paired_plot_bootstrap(labels, plots, predictions, contrasts,
        num_classes=4, replicates=73, seed=111, chunk_size=7)
    brute = []
    for row in weights:
        indices = np.concatenate([np.tile(np.flatnonzero(plots == plot), int(count))
                                  for plot, count in zip(sorted(set(plots)), row)])
        differences = []
        for seed in range(4):
            a = classification_metrics(labels[indices], logits["left"][seed, indices])[0]
            b = classification_metrics(labels[indices], logits["right"][seed, indices])[0]
            differences.append([a[k] - b[k] for k in METRICS])
        brute.append(np.mean(differences, axis=0))
    assert np.allclose(draws[:, 0], brute, atol=1e-15, rtol=0)
    assert weights.shape == (73, 3) and np.all(weights.sum(axis=1) == 3)
    again = paired_plot_bootstrap(labels, plots, predictions, contrasts,
                                  num_classes=4, replicates=73, seed=111, chunk_size=19)
    assert np.array_equal(draws, again[1]) and np.array_equal(weights, again[2])
    for i, metric in enumerate(METRICS):
        assert intervals["paired"][metric] == pytest.approx(np.percentile(np.array(brute)[:, i], [2.5, 97.5]))


def test_paired_seed_cancellations_have_exactly_zero_accuracy_differences():
    labels, plots = np.array([0, 1]), np.array(["a", "b"])
    right = np.tile(np.array([[[0], [0]]]), (4, 1, 1))
    left = right.copy()
    left[0, 0, 0], left[0, 1, 0] = 1, 1
    left[1, 0, 0], left[1, 1, 0] = 0, 0
    # Swap identical two methods between seed pairs; the four-seed contrast is zero.
    right[1] = left[0]
    contrasts = [{"contrast_id": "equal", "lhs_arm_id": "a", "rhs_arm_id": "b"}]
    _, draws, _ = paired_plot_bootstrap(labels, plots, {"a": left, "b": right}, contrasts,
                                       num_classes=2, replicates=17, seed=7)
    assert np.array_equal(draws[:, 0, :2], np.zeros((17, 2)))


@pytest.mark.parametrize("deltas,lower,guard,expected", [
    ([0.1, 0.1, 0.1, -0.01], 0.001, 0, "supported_improvement"),
    ([0.1, 0.1, 0.1, -0.01], 0, 0, "inconclusive"),
    ([0.1, 0.1, 0.1, -0.01], 0.001, -0.001, "inconclusive"),
    ([0.1, 0.1, -0.01, -0.01], 0.001, 0, "inconclusive"),
    ([0, 0, 0, 0], 0, 0, "no_improvement"),
    ([-0.1, -0.1, -0.1, 0.01], -0.2, 0.1, "no_improvement"),
])
def test_predeclared_categories_do_not_promote_uncertainty_or_metric_harm(deltas, lower, guard, expected):
    per_seed = {str(i): {"acc1": delta} for i, delta in enumerate(deltas)}
    mean = {"acc1": np.mean(deltas), "acc3": guard, "weighted_f1": 0, "mcc": 0}
    assert interpretation_category(per_seed, mean, {"acc1": [lower, 0.2]}) == expected


def test_selected_comparisons_preserve_rescue_harm_counts(study, records):
    _, config, matrix, _, _ = study
    _, selected = select_learning_rates(config, matrix, records)
    contrasts, draws, weights = compare_selected(config, selected, records, declare_contrasts(config, matrix))
    assert draws.shape == (31, 63, 4) and weights.shape == (31, 3)
    for contrast in contrasts:
        for seed, counts in contrast["rescues_harms_by_seed"].items():
            assert counts["net"] == counts["rescued"] - counts["harmed"]
            assert contrast["per_seed"][seed]["acc1"] == counts["net"] / 6


def test_swapped_seed_secondary_metric_means_remain_exact_ties(study, records, monkeypatch):
    import multimodal.attention_fusion_comparison as comparison

    _, config, matrix, _, _ = study
    _, selected = select_learning_rates(config, matrix, records)
    selected = copy.deepcopy(selected)
    declaration = declare_contrasts(config, matrix)
    primary = next(c for c in declaration["contrasts"] if c["kind"] == "primary")
    by_arm = {a["arm_id"]: a for a in selected}
    left, right = by_arm[primary["lhs_arm_id"]], by_arm[primary["rhs_arm_id"]]
    lhs, rhs = [0.7, 0.001, 0.3, 0.002], [0.3, 0.002, 0.7, 0.001]
    # This real floating-point cancellation previously produced -2.1684e-19
    # with np.mean, incorrectly failing the strict no-secondary-harm guard.
    assert np.mean(np.array(lhs) - np.array(rhs)) < 0
    for index, seed in enumerate(map(str, config["data"]["training_seeds"])):
        for metric in ("weighted_f1", "mcc"):
            left["per_seed"][seed]["metrics"][metric] = lhs[index]
            right["per_seed"][seed]["metrics"][metric] = rhs[index]
    intervals = {c["contrast_id"]: {k: [0.01, 0.2] for k in METRICS} for c in declaration["contrasts"]}
    monkeypatch.setattr(comparison, "paired_plot_bootstrap", lambda *args, **kwargs: (intervals, np.empty(0), np.empty(0)))
    contrasts, _, _ = compare_selected(config, selected, records, declaration)
    result = next(c for c in contrasts if c["contrast_id"] == primary["contrast_id"])
    assert result["mean_difference"]["weighted_f1"] == 0.0
    assert result["mean_difference"]["mcc"] == 0.0
    positive = {str(seed): {"acc1": 0.1} for seed in range(1, 5)}
    mean = {**result["mean_difference"], "acc1": 0.1, "acc3": 0.0}
    assert interpretation_category(positive, mean, result["interval95"]) == "supported_improvement"


def test_atomic_reference_publication_never_leaves_partial_destination(tmp_path, monkeypatch):
    import multimodal.attention_fusion_comparison as comparison

    path = tmp_path / "selection/selected.json"
    value = {"fit_id": "example", "checkpoint_sha256": "hash"}
    original_link = comparison.os.link

    def interrupt_before_publish(source, destination):
        assert json.loads(Path(source).read_text()) == value
        assert not Path(destination).exists()
        raise KeyboardInterrupt("synthetic publication interruption")

    monkeypatch.setattr(comparison.os, "link", interrupt_before_publish)
    with pytest.raises(KeyboardInterrupt):
        comparison._write_new_json(path, value)
    assert not path.exists() and not list(path.parent.iterdir())
    monkeypatch.setattr(comparison.os, "link", original_link)
    comparison._write_new_json(path, value)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        comparison._write_new_json(path, {"different": "cannot overwrite"})
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == ["selected.json"]


def test_analysis_writes_authenticated_outputs_and_76_immutable_references(study, monkeypatch):
    repo, config, matrix, references, declaration_path = study
    import multimodal.attention_fusion_comparison as comparison
    import tools.run_multimodal_attention_fusion as runner

    monkeypatch.setattr(runner, "read_frozen_plan", lambda *a, **k: (config, matrix, {"manifest_payload_sha256": "synthetic"}))
    monkeypatch.setattr(comparison, "load_reference_inputs", lambda *a, **k: references)
    output = repo / config["paths"]["artifact_root"] / "reports/m3"
    result = analyze_study(config, matrix, declaration_path, output, repo_root=repo)
    assert result["counts"] == {"development_fits": 228, "selected_arms": 19, "selected_checkpoints": 76, "contrasts": 63}
    assert result["class_labels"] == ["Habitat 0", "Habitat 1", "Habitat 2", "Habitat 4"]
    assert result["validation_class_support_images"][-1] == 0
    assert result["validation_class_support_plots"][-1] == 0
    selection = json.loads((output / "selection.json").read_text())
    for reference in selection["selected_references"]:
        assert json.loads((repo / reference["selection_path"]).read_text()) == reference
        assert reference["retrain"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(sha256_file(output / name) == value for name, value in manifest["files"].items())
    with pytest.raises(ComparisonError, match="analysis exists"):
        analyze_study(config, matrix, declaration_path, output, repo_root=repo)
