"""M3J curves and selected-checkpoint comparisons retain their actual evidence."""

import copy
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from multimodal import attention_fusion_joint_reporting as reporting
from multimodal.attention_fusion_training import classification_metrics


def metric_fixture(prediction):
    labels = np.array([0, 0, 1, 1, 2, 2])
    logits = np.zeros((len(labels), 18), dtype=np.float32)
    logits[np.arange(len(labels)), prediction] = 3
    metrics, arrays = classification_metrics(labels, logits)
    arrays.update({"label": labels, "row_uid": np.array([f"image_{index}" for index in range(len(labels))]),
                   "plot_idx": np.array(["a", "a", "b", "c", "c", "d"])})
    return metrics, arrays


def history_fixture(final=None, best=None):
    if final is None:
        final, _ = metric_fixture([0, 1, 1, 1, 2, 0])
    if best is None:
        best, _ = metric_fixture([0, 0, 1, 1, 2, 0])
    return [{"epoch": epoch, "optimizer_steps": 212 * epoch,
             "learning_rate": 5e-5 / epoch, "next_learning_rate": 5e-5 / (epoch + 1),
             "train": {**copy.deepcopy(metrics), "optimization_loss": 0.25 * epoch},
             "validation": copy.deepcopy(metrics), "train_seconds": 80 + epoch,
             "validation_seconds": 12, "epoch_seconds": 95 + epoch,
             "peak_cuda_bytes": 12 * 2**30}
            for epoch, metrics in enumerate((best, final), 1)]


def write_run(root, candidate, initialization, *, seed=1, complete=True, regime="joint_partial", predictions=None):
    final_prediction = [0, 1, 1, 1, 2, 0 if initialization == "pretrained" else 2]
    best_prediction = [0, 0, 1, 1, 2, 0 if initialization == "pretrained" else 2]
    if predictions is not None:
        final_prediction, best_prediction = predictions
    final, final_arrays = metric_fixture(final_prediction)
    best, best_arrays = metric_fixture(best_prediction)
    directory = root / "runs" / f"seed{seed}" / regime / candidate / initialization
    directory.mkdir(parents=True)
    spec = {"run_id": directory.relative_to(root / "runs").as_posix(), "candidate_id": candidate,
            "initialization": initialization, "regime": regime, "seed": seed,
            "class_labels": [f"Habitat {i}" for i in range(18)], "expected_epochs": 2,
            "status": "complete" if complete else "running", "elapsed_seconds": 200.0}
    (directory / "run.json").write_text(json.dumps(spec))
    history = history_fixture(final, best)
    (directory / "history.json").write_text(json.dumps(history if complete else history[:1]))
    if complete:
        (directory / "metrics.json").write_text(json.dumps({
            "final": {"epoch": 2, "metrics": final}, "best": {"epoch": 1, "metrics": best}}))
        np.savez_compressed(directory / "final_predictions.npz", **final_arrays)
        np.savez_compressed(directory / "best_predictions.npz", **best_arrays)
    return directory


def write_loader_config(directory, *, seed=1, regime="joint_partial", policy=None):
    prefix = {"joint_partial": "attention_fusion_joint",
              "frozen_online": "attention_fusion_joint_frozen_online"}[regime]
    study_id = f"{prefix}_direct_v1" if seed == 1 else f"{prefix}_seed{seed}_direct_v1"
    training = {f"{stream}_seed_namespace": "attention_fusion_joint_v1"
                for stream in ("shuffle", "augmentation")}
    if policy is not None:
        training["loader_rng_policy"] = policy
    (directory / "config.json").write_text(json.dumps({"study_id": study_id, "training": training}))


def test_best_epoch_uses_one_unrounded_checkpoint_and_earliest_exact_tie():
    history = history_fixture()
    assert reporting.best_epoch(history) == 1
    # Acc@3 does not override the declared Acc@1 -> F1 -> MCC ranking.
    history[1]["validation"]["acc3"] = 1.0
    assert reporting.best_epoch(history) == 1
    history[1]["validation"] = copy.deepcopy(history[0]["validation"])
    assert reporting.best_epoch(history) == 1
    history[1]["validation"]["weighted_f1"] += 1e-10
    assert reporting.best_epoch(history) == 2
    history[1]["validation"]["weighted_f1"] = history[0]["validation"]["weighted_f1"]
    history[1]["validation"]["mcc"] += 1e-10
    assert reporting.best_epoch(history) == 2


@pytest.mark.parametrize("change", ["nonfinite", "duplicate_epoch", "step_regression"])
def test_history_rejects_corrupt_or_replayed_epochs(change):
    history = history_fixture()
    if change == "nonfinite":
        history[1]["validation"]["acc1"] = float("nan")
    elif change == "duplicate_epoch":
        history[1]["epoch"] = 1
    else:
        history[1]["optimizer_steps"] = 212
    with pytest.raises(ValueError):
        reporting.history_rows(history)


def test_live_curves_plot_exact_history_and_preserve_optimization_loss(tmp_path, monkeypatch):
    history = history_fixture()
    before = plt.get_fignums()
    recorded = {}

    def capture(fig, stem, pdf=True):
        recorded["axes"] = fig.axes
        recorded["pdf"] = pdf
        return [str(stem.with_suffix(".png"))]

    monkeypatch.setattr(reporting, "_save", capture)
    files = reporting.plot_history(history, tmp_path, "Seed 1 <joint>", save_pdf=False)
    assert not recorded["pdf"]
    axes = recorded["axes"]
    assert len(axes) == 9
    assert axes[0].lines[0].get_label() == "Training pass (augmented)"
    assert np.array_equal(axes[0].lines[0].get_ydata(), [0.25, 0.5])
    assert np.array_equal(axes[0].lines[1].get_ydata(), [row["validation"]["loss"] for row in history])
    assert np.array_equal(axes[1].lines[1].get_ydata(), [100 * row["validation"]["acc1"] for row in history])
    assert np.array_equal(axes[5].lines[0].get_ydata(), [row["learning_rate"] for row in history])
    assert np.array_equal(axes[5].lines[1].get_ydata(), [row["next_learning_rate"] for row in history])
    assert np.array_equal(axes[7].lines[0].get_ydata(), [12, 12])
    assert np.array_equal(axes[8].lines[0].get_ydata(), [212, 424])
    assert plt.get_fignums() == before
    page = (tmp_path / "monitoring.html").read_text()
    assert "Seed 1 &lt;joint&gt;" in page and "content='30'" in page
    assert str(tmp_path / "history.csv") in files
    with (tmp_path / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert float(rows[1]["validation_acc1"]) == history[1]["validation"]["acc1"]


@pytest.mark.parametrize("read_from_disk", [False, True])
def test_geo_only_curves_describe_dropout_without_image_augmentation(tmp_path, monkeypatch, read_from_disk):
    metadata = {"candidate_id": "geo_only"}
    if read_from_disk:
        (tmp_path / "run.json").write_text(json.dumps(metadata))
    recorded = {}

    def capture(fig, stem, pdf=True):
        recorded["label"] = fig.axes[0].lines[0].get_label()
        recorded["footer"] = fig.texts[-1].get_text()
        return []

    monkeypatch.setattr(reporting, "_save", capture)
    reporting.plot_history(history_fixture(), tmp_path / "curves",
                           metadata=None if read_from_disk else metadata)
    assert recorded["label"] == "Training pass (dropout)"
    assert "Training uses dropout and changing weights" in recorded["footer"]
    assert "augmentation" not in recorded["footer"]


def test_complete_package_keeps_both_policies_pairs_initializations_and_all_class_axes(tmp_path, monkeypatch):
    candidates = [name for name in reporting.MODEL_LABELS if name != "geo_only"]
    for candidate in candidates:
        for initialization in ("pretrained", "cs_adapted"):
            write_run(tmp_path, candidate, initialization)
    confusions = []

    def draw(cm, labels, sink):
        confusions.append((np.asarray(cm), labels))

    monkeypatch.setattr(reporting, "draw_cm", draw)
    output = tmp_path / "reports" / "seed1_joint_partial"
    summary = reporting.build_report(tmp_path, output)
    assert summary["completed_fits"] == 16 and summary["incomplete_fits"] == 0
    assert summary["seeds"] == [1]
    assert len(confusions) == 32
    assert all(cm.shape == (18, 18) and len(labels) == 18 for cm, labels in confusions)
    for policy, expected_epoch in (("final", 2), ("best", 1)):
        rows = summary["policies"][policy]["runs"]
        assert len(rows) == 16 and {row["epoch"] for row in rows} == {expected_epoch}
        contrasts = summary["policies"][policy]["contrasts"]
        initialization = [item for item in contrasts if item["kind"] == "cs_adapted_minus_pretrained"]
        assert len(initialization) == 8
        assert all(item["difference_acc1"] == pytest.approx(1 / 6) for item in initialization)
        assert all((item["rescued"], item["harmed"], item["net_correct"]) == (1, 0, 1)
                   for item in initialization)
        assert len([item for item in contrasts if item["kind"] == "primary_attention_vs_projected"]) == 2
        assert len([item for item in contrasts if item["kind"] == "bypass_vs_cls"]) == 4
        with (output / "tables" / f"{policy}_confusion_by_class.csv").open() as stream:
            classes = list(csv.DictReader(stream))
        assert len(classes) == 16 * 18
        assert classes[3]["support_images"] == "0" and classes[3]["recall"] == ""
    final = summary["policies"]["final"]["runs"][0]
    best = summary["policies"]["best"]["runs"][0]
    assert final["run_id"] == best["run_id"]
    assert best["acc1"] > final["acc1"]
    text = (output / "report.md").read_text()
    assert "provisional" in text and "conditional on checkpoint" in text
    assert "## Final epoch" in text and "## Best validation" in text
    assert "five CS adaptation epochs" in text


def test_incomplete_run_is_not_reported_as_final_and_other_seeds_are_filtered(tmp_path):
    write_run(tmp_path, "image_only", "pretrained", complete=False)
    write_run(tmp_path, "image_only", "pretrained", seed=2)
    summary = reporting.build_report(tmp_path, render_confusions=False)
    assert summary["completed_fits"] == 0 and summary["incomplete_fits"] == 1
    assert summary["seeds"] == [1]
    assert summary["policies"]["final"]["runs"] == []


def test_replay_pending_run_is_not_yet_marked_complete(tmp_path):
    directory = write_run(tmp_path, "image_only", "pretrained")
    path = directory / "run.json"
    spec = json.loads(path.read_text())
    spec["status"] = "running"
    path.write_text(json.dumps(spec))
    summary = reporting.build_report(tmp_path, render_confusions=False)
    assert summary["completed_fits"] == 0 and summary["incomplete_fits"] == 1


def test_selection_provenance_mismatch_is_rejected(tmp_path):
    directory = write_run(tmp_path, "image_only", "pretrained")
    path = directory / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["best"]["epoch"] = 2
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Incorrect best checkpoint epoch"):
        reporting.build_report(tmp_path, render_confusions=False)


def test_paired_prediction_identity_mismatch_is_rejected(tmp_path):
    write_run(tmp_path, "image_only", "pretrained")
    directory = write_run(tmp_path, "image_only", "cs_adapted")
    for policy in ("final", "best"):
        path = directory / f"{policy}_predictions.npz"
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        data["row_uid"] = data["row_uid"][::-1]
        np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="row_uid are not aligned"):
        reporting.build_report(tmp_path, render_confusions=False)


def test_prediction_confusion_disagreement_is_rejected(tmp_path):
    directory = write_run(tmp_path, "image_only", "pretrained")
    path = directory / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["final"]["metrics"]["confusion_matrix"][0][0] += 1
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Confusion/prediction disagreement"):
        reporting.build_report(tmp_path, render_confusions=False)


@pytest.mark.parametrize("source", ["config", "run_spec", "run"])
def test_exploratory_assurance_propagates_to_reports_tables_and_audits(tmp_path, source):
    directory = write_run(tmp_path, "image_only", "pretrained")
    write_run(tmp_path, "raw_concat", "pretrained")
    path = directory / f"{source}.json"
    value = json.loads(path.read_text()) if path.exists() else {}
    if source == "config":
        value["numerical_policy"] = {"mode": "exploratory"}
    else:
        value["result_assurance"] = reporting.PROVISIONAL_ASSURANCE
    path.write_text(json.dumps(value))
    audit_summary = {"mode": "exploratory", "comparisons": 4,
                     "accepted_numerical_comparisons": 1, "rejected_comparisons": 0,
                     "frozen_changed_coordinates": 1, "max_absolute_error": 3.725e-9}
    (directory / "numerical_audit.json").write_text(json.dumps({"summary": audit_summary, "events": []}))
    output = tmp_path / "report"
    summary = reporting.build_report(tmp_path, output, render_confusions=False)
    assert summary["result_assurance"] == reporting.PROVISIONAL_ASSURANCE
    assert summary["numerical_audits"][0]["summary"] == audit_summary
    assert summary["numerical_audits"][0]["audit_present"]
    for policy in reporting.POLICIES:
        rows = summary["policies"][policy]["runs"]
        assert next(row for row in rows if row["candidate_id"] == "image_only")["result_assurance"] == reporting.PROVISIONAL_ASSURANCE
        assert next(row for row in rows if row["candidate_id"] == "raw_concat")["result_assurance"] != reporting.PROVISIONAL_ASSURANCE
        # A comparison involving either provisional participant stays provisional.
        assert all(row["result_assurance"] == reporting.PROVISIONAL_ASSURANCE
                   for row in summary["policies"][policy]["contrasts"])
        with (output / "tables" / f"{policy}_metrics.csv").open() as stream:
            assert reporting.PROVISIONAL_ASSURANCE in {row["result_assurance"] for row in csv.DictReader(stream)}
    report = (output / "report.md").read_text()
    assert reporting.PROVISIONAL_TITLE in report
    assert "do not establish machine health or bit-for-bit reproducibility" in report
    assert "Recorded audit" in report and "3.725e-09" in report


def test_exploratory_incomplete_run_reports_missing_audit(tmp_path):
    directory = write_run(tmp_path, "image_only", "pretrained", complete=False)
    (directory / "config.json").write_text(json.dumps({"numerical_policy": {"mode": "exploratory"}}))
    summary = reporting.build_report(tmp_path, render_confusions=False)
    assert summary["completed_fits"] == 0
    assert summary["result_assurance"] == reporting.PROVISIONAL_ASSURANCE
    assert not summary["numerical_audits"][0]["audit_present"]
    assert "**Missing audit**" in (tmp_path / "report/report.md").read_text()


@pytest.mark.parametrize("read_from_disk", [False, True])
def test_exploratory_live_curves_are_visibly_provisional(tmp_path, monkeypatch, read_from_disk):
    titles = []
    metadata = {"numerical_policy": {"mode": "exploratory"}}
    if read_from_disk:
        (tmp_path / "config.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(reporting, "_save", lambda fig, stem, pdf=True: titles.append(fig._suptitle.get_text()) or [])
    output = tmp_path / "curves"
    reporting.plot_history(history_fixture(), output, metadata=None if read_from_disk else metadata)
    assert reporting.PROVISIONAL_TITLE in titles[0]
    assert reporting.PROVISIONAL_NOTICE in (output / "monitoring.html").read_text()
    with (output / "history.csv").open() as stream:
        assert {row["result_assurance"] for row in csv.DictReader(stream)} == {reporting.PROVISIONAL_ASSURANCE}


def test_exploratory_confusion_exports_keep_provisional_caption(tmp_path, monkeypatch):
    directory = write_run(tmp_path, "image_only", "pretrained")
    (directory / "config.json").write_text(json.dumps({"numerical_policy": {"mode": "exploratory"}}))
    titles = []

    def draw(cm, labels, sink):
        fig = plt.figure()
        try:
            sink(fig, np.asarray(cm), False)
        finally:
            plt.close(fig)

    monkeypatch.setattr(reporting, "draw_cm", draw)
    monkeypatch.setattr(reporting, "_save", lambda fig, stem, pdf=True: titles.append(fig._suptitle.get_text()) or [])
    reporting.build_report(tmp_path)
    assert len(titles) == 2
    assert all(reporting.PROVISIONAL_TITLE in title for title in titles)


def test_exploratory_replay_loss_bound_keeps_checkpoint_selection_and_metadata_separate(tmp_path):
    directory = write_run(tmp_path, "image_only", "pretrained")
    policy = {"mode": "exploratory"}
    (directory / "config.json").write_text(json.dumps({"numerical_policy": policy}))
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update(result_assurance=reporting.PROVISIONAL_ASSURANCE, numerical_policy=policy)
    metrics_path.write_text(json.dumps(metrics))
    history_path = directory / "history.json"
    history = json.loads(history_path.read_text())
    # Original epoch loss and regenerated replay loss can differ within the
    # declared bound; saved prediction/metric consistency remains mandatory.
    history[1]["validation"]["loss"] += 5e-7
    history_path.write_text(json.dumps(history))
    summary = reporting.build_report(tmp_path, render_confusions=False)
    assert summary["completed_fits"] == 1
    assert set(summary["policies"]) == {"final", "best"}
    history[1]["validation"]["loss"] += 0.001
    history_path.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="metrics do not match selected epoch"):
        reporting.build_report(tmp_path, render_confusions=False)
    history[1]["validation"]["loss"] = metrics["final"]["metrics"]["loss"]
    history[1]["validation"]["acc1"] += 1e-10
    history_path.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="metrics do not match selected epoch"):
        reporting.build_report(tmp_path, render_confusions=False)


def test_direct_policy_reports_reloaded_metrics_without_replay_gate_or_missing_audit(tmp_path):
    directory = write_run(tmp_path, "image_only", "pretrained")
    (directory / "config.json").write_text(json.dumps({
        "verification_policy": reporting.DIRECT_VERIFICATION_POLICY,
        "result_assurance": reporting.PROVISIONAL_ASSURANCE,
    }))
    history_path = directory / "history.json"
    history = json.loads(history_path.read_text())
    history[0]["validation"]["acc1"] = 0.99
    history[1]["validation"]["loss"] += 0.01
    history_path.write_text(json.dumps(history))
    summary = reporting.build_report(tmp_path, render_confusions=False)
    assert summary["completed_fits"] == 1
    assert summary["result_assurance"] == reporting.PROVISIONAL_ASSURANCE
    assert "numerical_audits" not in summary
    final = summary["policies"]["final"]["runs"][0]
    assert final["metrics_source"] == "reloaded_checkpoint_evaluation"
    assert final["original_epoch_loss"] == history[1]["validation"]["loss"]
    assert final["loss"] != final["original_epoch_loss"]
    best = summary["policies"]["best"]["runs"][0]
    assert best["epoch"] == 1 and best["original_epoch_acc1"] == 0.99
    assert best["acc1"] != best["original_epoch_acc1"]
    report = (tmp_path / "report/report.md").read_text()
    assert "evaluating those saved checkpoints after loading" in report
    assert "Missing audit" not in report
    assert "tolerance is enabled" not in report


def test_frozen_report_pairs_reference_by_model_initialization_seed_and_policy_without_writing_it(tmp_path):
    reference, frozen = tmp_path / "joint", tmp_path / "frozen"
    for candidate in ("image_only", "attention_two_token", "projected_concat"):
        for initialization in ("pretrained", "cs_adapted"):
            write_run(reference, candidate, initialization)
            write_run(frozen, candidate, initialization, regime="frozen_online")
    # The joint reference corrects one extra image for this specific pair.
    target = frozen / "runs/seed1/frozen_online/attention_two_token/pretrained"
    for path in target.iterdir():
        path.unlink()
    target.rmdir()
    write_run(frozen, "attention_two_token", "pretrained", regime="frozen_online",
              predictions=([1, 1, 1, 1, 2, 0], [1, 0, 1, 1, 2, 0]))
    write_run(reference, "image_only", "pretrained", seed=2)
    before = {path: path.read_bytes() for path in reference.rglob("*") if path.is_file()}
    summary = reporting.build_report(frozen, render_confusions=False, regime="frozen_online",
                                     paired_reference_root=reference)
    assert summary["completed_fits"] == 6
    paired = summary["encoder_regime_comparison"]
    assert paired["direction"] == "joint_partial_minus_frozen_online"
    assert paired["reference_access"] == "read_only"
    assert paired["loader_rng_comparison"]["status"] == "unknown"
    assert "cannot be established" in summary["encoder_regime_rng_limitation"]
    for policy, epoch in (("final", 2), ("best", 1)):
        comparisons = paired["policies"][policy]
        assert comparisons["paired_complete"] == 6
        assert comparisons["unavailable_pairs"] == []
        assert all(item["seed"] == 1 and item["lhs_epoch"] == item["rhs_epoch"] == epoch
                   for item in comparisons["contrasts"])
        attention = next(item for item in comparisons["contrasts"]
                         if item["candidate_id"] == "attention_two_token" and item["initialization"] == "pretrained")
        assert attention["difference_acc1"] == pytest.approx(1 / 6)
        assert (attention["rescued"], attention["harmed"], attention["net_correct"]) == (1, 0, 1)
        assert "joint_partial" in attention["lhs_run_id"] and "frozen_online" in attention["rhs_run_id"]
        assert all(item["regime"] == "frozen_online" for item in summary["policies"][policy]["contrasts"])
        assert len([item for item in summary["policies"][policy]["contrasts"]
                    if item["kind"] == "primary_attention_vs_projected"]) == 2
        with (frozen / f"report/tables/{policy}_encoder_regime_contrasts.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 6
    after = {path: path.read_bytes() for path in reference.rglob("*") if path.is_file()}
    assert before == after
    report = (frozen / "report/report.md").read_text()
    assert "partial-joint minus frozen-online" in report
    assert "Each regime selects its own best-validation epoch" in report


def test_frozen_report_exposes_missing_and_incomplete_regime_pairs(tmp_path):
    reference, frozen = tmp_path / "joint", tmp_path / "frozen"
    for candidate in ("image_only", "raw_concat", "projected_concat"):
        write_run(reference, candidate, "pretrained")
    write_run(frozen, "image_only", "pretrained", regime="frozen_online")
    write_run(frozen, "raw_concat", "pretrained", regime="frozen_online", complete=False)
    write_run(frozen, "attention_cls", "pretrained", regime="frozen_online")
    summary = reporting.build_report(frozen, render_confusions=False, regime="frozen_online",
                                     paired_reference_root=reference)
    for policy in reporting.POLICIES:
        paired = summary["encoder_regime_comparison"]["policies"][policy]
        assert paired["paired_complete"] == 1 and len(paired["unavailable_pairs"]) == 3
        missing = {item["candidate_id"]: item for item in paired["unavailable_pairs"]}
        assert missing["raw_concat"]["frozen_online_status"] == "incomplete"
        assert missing["projected_concat"]["frozen_online_status"] == "missing"
        assert missing["attention_cls"]["joint_partial_status"] == "missing"
        assert len(paired["contrasts"]) == 1
    assert "1 complete pairs; 3 unavailable pairs" in (frozen / "report/report.md").read_text()


@pytest.mark.parametrize("identity", ["row_uid", "label", "plot_idx", "class_labels"])
def test_frozen_report_rejects_mismatched_regime_cohorts(tmp_path, identity):
    reference, frozen = tmp_path / "joint", tmp_path / "frozen"
    write_run(reference, "image_only", "pretrained")
    directory = write_run(frozen, "image_only", "pretrained", regime="frozen_online")
    if identity == "class_labels":
        path = directory / "run.json"
        spec = json.loads(path.read_text())
        spec[identity] = spec[identity][::-1]
        path.write_text(json.dumps(spec))
        message = "Paired class-label order differs"
    else:
        for policy in reporting.POLICIES:
            path = directory / f"{policy}_predictions.npz"
            with np.load(path, allow_pickle=False) as source:
                data = {key: source[key] for key in source.files}
            for key in (("label", "prediction", "logits") if identity == "label" else (identity,)):
                data[key] = data[key][::-1]
            np.savez_compressed(path, **data)
        message = f"Paired validation {identity} are not aligned"
    with pytest.raises(ValueError, match=message):
        reporting.build_report(frozen, render_confusions=False, regime="frozen_online",
                               paired_reference_root=reference)


def test_frozen_report_cannot_write_outputs_into_reference_package(tmp_path):
    reference, frozen = tmp_path / "joint", tmp_path / "frozen"
    write_run(reference, "image_only", "pretrained")
    write_run(frozen, "image_only", "pretrained", regime="frozen_online")
    with pytest.raises(ValueError, match="cannot modify the paired reference root"):
        reporting.build_report(frozen, reference / "reports", render_confusions=False,
                               regime="frozen_online", paired_reference_root=reference)


def test_direct_policy_still_rejects_wrong_selected_epoch(tmp_path):
    directory = write_run(tmp_path, "image_only", "pretrained")
    (directory / "config.json").write_text(json.dumps({
        "verification_policy": reporting.DIRECT_VERIFICATION_POLICY,
        "result_assurance": reporting.PROVISIONAL_ASSURANCE,
    }))
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["best"]["epoch"] = 2
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Incorrect best checkpoint epoch"):
        reporting.build_report(tmp_path, render_confusions=False)


def test_combined_seed_report_counts_one_shared_geo_fit_and_preserves_reference_packages(tmp_path):
    joint, frozen, geo = (tmp_path / name for name in ("joint", "frozen", "geo"))
    for candidate in reporting.MODEL_LABELS:
        if candidate == "geo_only":
            continue
        for initialization in ("pretrained", "cs_adapted"):
            joint_run = write_run(joint, candidate, initialization)
            frozen_run = write_run(frozen, candidate, initialization, regime="frozen_online")
            write_loader_config(joint_run)
            write_loader_config(frozen_run, regime="frozen_online")
    write_run(geo, "geo_only", "none", regime="geo_only",
              predictions=([1, 1, 1, 1, 2, 0], [1, 0, 1, 1, 2, 0]))
    # A reference source fingerprint is historical evidence, not a requirement
    # to reproduce the currently installed reporting implementation.
    (joint / "protocol.json").write_text(json.dumps({"source_sha256": "old-code"}))
    write_run(joint, "image_only", "pretrained", seed=2)
    before = {path: path.read_bytes() for root in (joint, frozen)
              for path in root.rglob("*") if path.is_file()}
    summary = reporting.build_report(geo, render_confusions=False, regime="geo_only",
                                     completed_reference_roots=[joint, frozen])
    assert summary["completed_fits"] == 33 and summary["incomplete_fits"] == 0
    assert sum(row["candidate_id"] == "geo_only" for row in summary["inventory"]) == 1
    assert summary["completed_reference_packages"]["access"] == "read_only"
    assert "not a strict trainability-only contrast" in summary["encoder_regime_rng_limitation"]
    rng = summary["encoder_regime_comparison"]["loader_rng_comparison"]
    assert rng["status"] == "different_namespaces"
    assert rng["pair_counts"]["different_namespaces"] == 16
    for policy, epoch in (("final", 2), ("best", 1)):
        rows = summary["policies"][policy]["runs"]
        assert len(rows) == 33 and {row["epoch"] for row in rows} == {epoch}
        shared = [item for item in summary["policies"][policy]["contrasts"]
                  if item["kind"] == "versus_shared_geo_only"]
        assert len(shared) == 32
        assert len({item["rhs_run_id"] for item in shared}) == 1
        assert {item["regime"] for item in shared} == {"joint_partial", "frozen_online"}
        assert all(item["rhs_epoch"] == epoch for item in shared)
        assert all(item["difference_acc1"] > 0 for item in shared)
        regime = summary["encoder_regime_comparison"]["policies"][policy]
        assert regime["paired_complete"] == 16 and regime["unavailable_pairs"] == []
    after = {path: path.read_bytes() for root in (joint, frozen)
             for path in root.rglob("*") if path.is_file()}
    assert before == after
    report = (geo / "report/report.md").read_text()
    assert "one shared geo-only fit, counted once" in report
    assert "training-order or image-crop randomness also differs" in report
    assert "image augmentation for image-based fits" in report
    assert "| Regime | Initialization |" in report


@pytest.mark.parametrize("seed", [2, 3, 4])
@pytest.mark.parametrize("geo_complete", [True, False])
def test_deferred_frozen_report_has_seventeen_active_fits_and_no_fabricated_regime_comparison(
        tmp_path, seed, geo_complete):
    joint, geo = tmp_path / "joint", tmp_path / "geo"
    for candidate in reporting.MODEL_LABELS:
        if candidate != "geo_only":
            for initialization in ("pretrained", "cs_adapted"):
                directory = write_run(joint, candidate, initialization, seed=seed)
                write_loader_config(directory, seed=seed, policy="declared_namespaces_v1")
    write_run(geo, "geo_only", "none", seed=seed, regime="geo_only", complete=geo_complete)
    before = {path: path.read_bytes() for path in joint.rglob("*") if path.is_file()}
    summary = reporting.build_report(
        geo, render_confusions=False, seed=seed, regime="geo_only",
        completed_reference_roots=[joint], deferred_reference_regimes=["frozen_online"])
    assert summary["completed_fits"] == 16 + int(geo_complete)
    assert summary["incomplete_fits"] == int(not geo_complete)
    assert len(summary["inventory"]) == 17
    assert summary["active_scope_complete"] is geo_complete
    assert summary["active_planned_fits"] == 17
    assert summary["deferred_planned_fits"] == 16
    assert summary["full_planned_fits"] == 33
    assert summary["deferred_reference_packages"] == [
        {"seed": seed, "regime": "frozen_online", "status": "deferred", "planned_fits": 16}]
    assert summary["encoder_regime_comparison_status"] == "deferred"
    assert "encoder_regime_comparison" not in summary
    assert "encoder_regime_rng_interpretation" not in summary
    assert "encoder_regime_rng_limitation" not in summary
    for policy in reporting.POLICIES:
        rows = summary["policies"][policy]["runs"]
        assert len(rows) == 16 + int(geo_complete)
        shared = [item for item in summary["policies"][policy]["contrasts"]
                  if item["kind"] == "versus_shared_geo_only"]
        assert len(shared) == (16 if geo_complete else 0)
        assert {item["regime"] for item in rows} <= {"joint_partial", "geo_only"}
        assert not (geo / "report/tables" / f"{policy}_encoder_regime_pairing.csv").exists()
    report = (geo / "report/report.md").read_text()
    assert "16 frozen-online fits deferred for later" in report
    assert "Deferred fits are not failed or incomplete runs" in report
    assert "provenance is missing" not in report
    assert "unavailable pairs" not in report
    assert before == {path: path.read_bytes() for path in joint.rglob("*") if path.is_file()}


@pytest.mark.parametrize("reference_regime,reference_seed,complete,expected", [
    ("frozen_online", 2, True, "deferred regime cannot also supply"),
    ("joint_partial", 3, True, "No matching runs in completed reference"),
    ("joint_partial", 2, False, "Completed reference contains an unfinished fit"),
])
def test_deferred_report_keeps_active_reference_identity_and_completion_checks(
        tmp_path, reference_regime, reference_seed, complete, expected):
    reference, geo = tmp_path / "reference", tmp_path / "geo"
    write_run(reference, "image_only", "pretrained", seed=reference_seed,
              regime=reference_regime, complete=complete)
    write_run(geo, "geo_only", "none", seed=2, regime="geo_only")
    with pytest.raises(ValueError, match=expected):
        reporting.build_report(geo, render_confusions=False, seed=2, regime="geo_only",
                               completed_reference_roots=[reference],
                               deferred_reference_regimes=["frozen_online"])


@pytest.mark.parametrize("seed,regime,deferred", [
    (1, "geo_only", ["frozen_online"]),
    (2, "joint_partial", ["frozen_online"]),
    (2, "geo_only", ["joint_partial"]),
])
def test_deferred_report_is_limited_to_later_seed_geo_packages(tmp_path, seed, regime, deferred):
    with pytest.raises(ValueError, match="Only seed 2–4 geo-only reports may defer"):
        reporting.build_report(tmp_path, seed=seed, regime=regime,
                               completed_reference_roots=[tmp_path / "reference"],
                               deferred_reference_regimes=deferred)


@pytest.mark.parametrize("seed", [2, 3, 4])
@pytest.mark.parametrize("combined", [False, True])
def test_later_seed_reports_use_saved_loader_policy_instead_of_seed1_rng_warning(tmp_path, seed, combined):
    joint, frozen, geo = (tmp_path / name for name in ("joint", "frozen", "geo"))
    for root, regime in ((joint, "joint_partial"), (frozen, "frozen_online")):
        directory = write_run(root, "image_only", "pretrained", seed=seed, regime=regime)
        write_loader_config(directory, seed=seed, regime=regime, policy="declared_namespaces_v1")
    before = {path: path.read_bytes() for path in joint.rglob("*") if path.is_file()}
    if combined:
        write_run(geo, "geo_only", "none", seed=seed, regime="geo_only")
        summary = reporting.build_report(geo, render_confusions=False, seed=seed, regime="geo_only",
                                         completed_reference_roots=[joint, frozen])
        report_root = geo
    else:
        summary = reporting.build_report(frozen, render_confusions=False, seed=seed, regime="frozen_online",
                                         paired_reference_root=joint)
        report_root = frozen
    comparison = summary["encoder_regime_comparison"]
    assert summary["seeds"] == [seed]
    assert comparison["loader_rng_comparison"]["status"] == "matched_namespaces"
    assert comparison["loader_rng_comparison"]["pair_counts"] == {
        "matched_namespaces": 1, "different_namespaces": 0, "unknown": 0}
    assert "encoder_regime_rng_limitation" not in summary
    for policy in reporting.POLICIES:
        pair = comparison["policies"][policy]["pairing"][0]
        assert pair["loader_rng_status"] == "matched_namespaces"
        assert pair["lhs_loader_rng_shuffle_namespace"] == pair["rhs_loader_rng_shuffle_namespace"]
        assert pair["lhs_loader_rng_policy"] == "declared_namespaces_v1"
    report = (report_root / "report/report.md").read_text()
    assert "matching shuffle and augmentation namespaces" in report
    assert "not bit-for-bit execution equivalence" in report
    assert "training-order or image-crop randomness also differs" not in report
    assert before == {path: path.read_bytes() for path in joint.rglob("*") if path.is_file()}


@pytest.mark.parametrize("alteration, expected", [("namespace", "different_namespaces"),
                                                  ("missing", "unknown"),
                                                  ("unsupported", "unknown")])
def test_declared_policy_does_not_imply_matching_when_recorded_namespaces_are_different_or_unknown(
        tmp_path, alteration, expected):
    joint, frozen = tmp_path / "joint", tmp_path / "frozen"
    for root, regime in ((joint, "joint_partial"), (frozen, "frozen_online")):
        directory = write_run(root, "image_only", "pretrained", seed=2, regime=regime)
        write_loader_config(directory, seed=2, regime=regime, policy="declared_namespaces_v1")
    path = directory / "config.json"
    config = json.loads(path.read_text())
    if alteration == "namespace":
        config["training"]["augmentation_seed_namespace"] = "different_augmentation"
    elif alteration == "missing":
        config["training"].pop("shuffle_seed_namespace")
    else:
        config["training"]["loader_rng_policy"] = "unknown_future_policy"
    path.write_text(json.dumps(config))
    summary = reporting.build_report(frozen, render_confusions=False, seed=2, regime="frozen_online",
                                     paired_reference_root=joint)
    assert summary["encoder_regime_comparison"]["loader_rng_comparison"]["status"] == expected
    assert "encoder_regime_rng_limitation" in summary


@pytest.mark.parametrize("policy", [None, "declared_namespaces_v1"])
@pytest.mark.parametrize("identity_field", ["candidate_id", "regime"])
def test_geo_only_loader_metadata_has_shared_shuffle_and_no_image_augmentation(policy, identity_field):
    config = {"study_id": "geo_package", "training": {
        "shuffle_seed_namespace": "shared_shuffle",
        "augmentation_seed_namespace": "unused_image_namespace"}}
    if policy is not None:
        config["training"]["loader_rng_policy"] = policy
    metadata = reporting._loader_rng_metadata([{identity_field: "geo_only"}, config])
    assert metadata == {"policy": policy or "legacy_study_id_v1",
                        "shuffle_namespace": "shared_shuffle", "augmentation_namespace": None}


def test_legacy_geo_loader_metadata_uses_study_id_only_when_shuffle_namespace_is_absent():
    metadata = reporting._loader_rng_metadata([
        {"candidate_id": "geo_only"}, {"study_id": "legacy_geo", "training": {}}])
    assert metadata["shuffle_namespace"] == "legacy_geo"
    assert metadata["augmentation_namespace"] is None


@pytest.mark.parametrize("identity", ["row_uid", "label", "plot_idx", "class_labels"])
def test_combined_geo_reference_rejects_mismatched_validation_cohort(tmp_path, identity):
    joint, frozen, geo = (tmp_path / name for name in ("joint", "frozen", "geo"))
    write_run(joint, "image_only", "pretrained")
    write_run(frozen, "image_only", "pretrained", regime="frozen_online")
    directory = write_run(geo, "geo_only", "none", regime="geo_only")
    if identity == "class_labels":
        path = directory / "run.json"
        spec = json.loads(path.read_text())
        spec[identity] = spec[identity][::-1]
        path.write_text(json.dumps(spec))
        message = "Class-label order differs between runs"
    else:
        for policy in reporting.POLICIES:
            path = directory / f"{policy}_predictions.npz"
            with np.load(path, allow_pickle=False) as source:
                data = {key: source[key] for key in source.files}
            for key in (("label", "prediction", "logits") if identity == "label" else (identity,)):
                data[key] = data[key][::-1]
            np.savez_compressed(path, **data)
        message = f"Paired validation {identity} are not aligned"
    with pytest.raises(ValueError, match=message):
        reporting.build_report(geo, render_confusions=False, regime="geo_only",
                               completed_reference_roots=[joint, frozen])


def test_combined_report_does_not_admit_unfinished_reference_fits(tmp_path):
    joint, geo = tmp_path / "joint", tmp_path / "geo"
    write_run(joint, "image_only", "pretrained", complete=False)
    write_run(geo, "geo_only", "none", regime="geo_only")
    with pytest.raises(ValueError, match="Completed reference contains an unfinished fit"):
        reporting.build_report(geo, render_confusions=False, regime="geo_only",
                               completed_reference_roots=[joint])


def test_combined_report_rejects_duplicate_reference_and_shared_geo_identities(tmp_path):
    joint, geo = tmp_path / "joint", tmp_path / "geo"
    write_run(joint, "image_only", "pretrained")
    write_run(geo, "geo_only", "none", regime="geo_only")
    with pytest.raises(ValueError, match="Duplicate completed reference roots"):
        reporting.build_report(geo, render_confusions=False, regime="geo_only",
                               completed_reference_roots=[joint, joint])
    write_run(geo, "geo_only", "duplicate", regime="geo_only")
    with pytest.raises(ValueError, match="Expected one shared geo-only fit per seed"):
        reporting.build_report(geo, render_confusions=False, regime="geo_only",
                               completed_reference_roots=[joint])


def test_combined_report_cannot_modify_reference_package(tmp_path):
    joint, geo = tmp_path / "joint", tmp_path / "geo"
    write_run(joint, "image_only", "pretrained")
    write_run(geo, "geo_only", "none", regime="geo_only")
    before = {path: path.read_bytes() for path in joint.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="cannot modify a completed reference root"):
        reporting.build_report(geo, joint / "report", render_confusions=False, regime="geo_only",
                               completed_reference_roots=[joint])
    assert before == {path: path.read_bytes() for path in joint.rglob("*") if path.is_file()}
