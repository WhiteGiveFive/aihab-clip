"""Exercise M3J checkpoint trajectories on a small CPU image/geo network."""

from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from multimodal import attention_fusion_joint_training as training
from multimodal.attention_fusion_joint_inputs import OnlineFusionModel
from multimodal.attention_fusion_joint_storage import JointCheckpointIntegrityError, JointCheckpointStore


class TinyOnlineModel(nn.Module):
    """Retain visual gradients, modality concatenation and stochastic dropout."""

    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(3, 4)
        self.visual.bias.requires_grad_(False)
        self.fusion = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Dropout(0.35), nn.Linear(8, 3))
        self.train_transform = self.prediction_transform = None
        self.provenance = {"fixture": "tiny_online_v1"}
        self.trainable_manifest = {"visual_tensors": 1, "trainable_visual_parameters": 12}

    def forward(self, image, geo):
        encoded = nn.functional.normalize(self.visual(image), dim=-1)
        return self.fusion(torch.cat((encoded, geo), dim=-1))


def make_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(283)
        return TinyOnlineModel()


def make_frozen_model(head_seed=283):
    """Exercise the production online wrapper with a small fully frozen tower."""
    class TinyVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Module()
            self.trunk.blocks = nn.ModuleList([nn.Identity() for _ in range(27)])
            self.trunk.norm = nn.LayerNorm(3)
            self.trunk.attn_pool = nn.Linear(3, 4)

        def forward(self, image):
            return self.trunk.attn_pool(self.trunk.norm(image))

    class TinyFusion(nn.Module):
        image_dim = 4
        candidate_id = "raw_concat"

        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Dropout(0.35), nn.Linear(8, 3))

        def forward(self, image, geo):
            return self.head(torch.cat((image, geo), dim=-1))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(283)
        visual = TinyVisual()
        torch.manual_seed(head_seed)
        return OnlineFusionModel(visual, TinyFusion(), regime="frozen_online", strict_scope=False,
                                 provenance={"fixture": "tiny_frozen_online_v1"})


def make_geo_model():
    class TinyGeoFusion(nn.Module):
        image_dim = 4
        candidate_id = "geo_only"

        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.1), nn.Linear(8, 3))

        def forward(self, image, geo):
            return self.head(geo)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(283)
        return OnlineFusionModel(None, TinyGeoFusion(), regime="geo_only",
                                 provenance={"fixture": "tiny_geo_only_v1"})


@pytest.fixture
def config(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "configs/multimodal_attention_fusion_joint.yaml"
    value = yaml.safe_load(path.read_text())
    value.pop("numerical_policy", None)
    value["verification_policy"] = "direct_freeze_checks_v1"
    value["result_assurance"] = "provisional_unresolved_memory_integrity"
    # These tests cover trainer mechanics; real-source pinning is checked by the
    # protocol tests. Keep identity stable while other implementation work runs.
    monkeypatch.setattr(training, "source_hashes", lambda: {"fixture": "a" * 64})
    monkeypatch.setattr(training, "_environment", lambda device: {"fixture_runtime": "cpu", "device": str(device)})
    return value


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(741)
    roles = {}
    for role, count in (("train", 19), ("validation", 7)):
        image, geo = torch.randn(count, 3, generator=generator), torch.randn(count, 2, generator=generator)
        label = torch.randint(0, 3, (count,), generator=generator)
        roles[role] = [{"image": image[index], "geo": geo[index], "label": label[index],
                        "row_uid": f"{role}-{index:03d}", "plot_idx": f"{role}-plot-{index:03d}"}
                       for index in range(count)]
    return {"roles": roles, "provenance": {"fixture": "aligned-development-v1", "source_role": "train"},
            "scaler": {"mode": "standardize", "fit_rows": 19},
            "source_config": {"class_ontology": {"classes": [
                {"dense_id": index, "label_name": f"class-{index}"} for index in range(3)]}}}


@pytest.fixture
def fit():
    return {"run_id": "seed1/joint_partial/raw_concat/pretrained", "seed": 1,
            "candidate_id": "raw_concat", "initialization": "pretrained", "regime": "joint_partial"}


def loader_factory(inputs, config, seed, epoch, **_):
    loaders = []
    for role in ("train", "validation"):
        generator = torch.Generator().manual_seed(seed * 10000 + epoch)
        rows = []
        for row in inputs["roles"][role]:
            item = dict(row)
            if role == "train":
                # Views change each epoch without consuming the dropout RNG.
                item["image"] = row["image"] + torch.randn(3, generator=generator) * 0.05
            rows.append(item)
        loaders.append(DataLoader(rows, batch_size=config["training"]["batch_size"],
                                  shuffle=role == "train", generator=generator,
                                  drop_last=False, num_workers=0))
    return tuple(loaders)


def run(config, fit, inputs, store, name="run", **kwargs):
    return training.train_fit(config, fit, inputs, store.root / "runs" / name, store,
                              model=make_model(), loader_factory=loader_factory,
                              device="cpu", verification=True, verification_epochs=2,
                              make_plots=False, **kwargs)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            assert_tree_equal(first, second)
    else:
        assert left == right


def without_timing(history):
    return [{key: value for key, value in entry.items()
             if key not in {"train_seconds", "validation_seconds", "epoch_seconds"}}
            for entry in history]


def test_fresh_fit_preserves_recipe_states_checkpoints_and_epoch_history(config, fit, inputs, tmp_path):
    store = JointCheckpointStore(tmp_path)
    result = run(config, fit, inputs, store)
    assert result["status"] == "complete"
    assert result["training_resume_supported"] is False
    assert result["metrics_source"] == "reloaded_checkpoint_evaluation"
    path = tmp_path / "runs/run/checkpoint_manifest.json"
    manifest = store.read_manifest(path)
    saved = store.load_state(manifest["last"]["state"])
    assert saved["completed_epoch"] == 2 and saved["next_epoch"] == 3
    assert saved["training_resume_supported"] is False
    assert saved["history"][-1]["optimizer_steps"] == 4
    assert saved["history"][0]["learning_rate"] == 5e-5
    assert saved["history"][0]["next_learning_rate"] < 5e-5
    group = saved["optimizer"]["param_groups"][0]
    assert group["weight_decay"] == 0 and group["betas"] == (0.9, 0.999)
    assert saved["scheduler"]["T_max"] == 50
    assert set(saved["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert "numerical_audit" not in manifest
    assert "numerical_policy" not in manifest["provenance"]
    assert manifest["freeze_checks"]["start"]["values"]["frozen_parameters_unchanged"]
    assert manifest["freeze_checks"]["end"]["values"]["frozen_parameters_unchanged"]
    assert manifest["freeze_checks"]["end"]["structure"]["trainable_gradients_checked"]
    initial = make_model()
    visual = store.load_weights(manifest["final"]["visual"])
    assert torch.equal(visual["bias"], initial.visual.bias)
    assert not torch.equal(visual["weight"], initial.visual.weight)
    for component in ("visual", "fusion"):
        assert manifest["final"][component] == manifest["last"][component]
    history = json.loads((tmp_path / "runs/run/history.json").read_text())
    assert history == saved["history"]
    selected = history[manifest["best"]["epoch"] - 1]["validation"]
    assert manifest["best"]["selection_metrics"] == selected
    assert manifest["final"]["training_metrics"] == history[-1]["validation"]
    for policy in ("best", "final"):
        assert (tmp_path / f"runs/run/{policy}_predictions.npz").exists()
        assert (tmp_path / f"runs/run/{policy}_training_predictions.npz").exists()
        assert manifest[policy]["metrics_source"] == "reloaded_checkpoint_evaluation"


def test_resume_and_existing_directory_are_rejected_without_mutation(config, fit, inputs, tmp_path):
    store = JointCheckpointStore(tmp_path)
    run(config, fit, inputs, store, stop_after_epoch=1)
    path = tmp_path / "runs/run/checkpoint_manifest.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="resume is disabled"):
        run(config, fit, inputs, store, resume=True)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        run(config, fit, inputs, store)
    assert path.read_bytes() == before
    assert run(config, fit, inputs, store, "fresh_restart")["status"] == "complete"


def test_geo_only_trains_without_images_exports_and_finalizes_without_retraining(
        config, inputs, tmp_path, monkeypatch):
    from multimodal import attention_fusion_joint_inputs as joint_inputs

    fit = {"run_id": "seed1/geo_only", "seed": 1, "candidate_id": "geo_only",
           "initialization": "not_applicable", "regime": "geo_only"}
    inputs = deepcopy(inputs)
    for rows in inputs["roles"].values():
        for row in rows:
            del row["image"]
    loader_calls = []

    def geo_loaders(inputs, config, seed, epoch, *, include_images,
                    train_transform, prediction_transform):
        assert include_images is False
        assert train_transform is prediction_transform is None
        loader_calls.append(epoch)
        return tuple(DataLoader(inputs["roles"][role], batch_size=16, shuffle=role == "train",
                     generator=torch.Generator().manual_seed(seed * 10000 + epoch))
                     for role in ("train", "validation"))

    model = make_geo_model()
    before = {name: value.clone() for name, value in model.fusion.state_dict().items()}
    store = JointCheckpointStore(tmp_path / "source")
    directory = store.root / "runs" / fit["run_id"]
    result = training.train_fit(config, fit, inputs, directory, store, model=model,
        loader_factory=geo_loaders, device="cpu", verification=True,
        verification_epochs=2, make_plots=False)
    assert result["status"] == "complete" and result["completed_epoch"] == 2
    assert loader_calls == [1, 2, 2]
    manifest = store.read_manifest(directory / "checkpoint_manifest.json")
    assert manifest["provenance"]["frozen_visual_integrity"]["reference"] is None
    for stage in ("start", "end"):
        check = manifest["freeze_checks"][stage]
        assert check["values"]["status"] == "not_applicable"
        assert check["values"]["visual_encoder_present"] is False
        assert check["values"]["checked_tensors"] == 0
        assert check["structure"]["frozen_tensors"] == 0
    for policy in ("last", "best", "final"):
        assert "visual" not in manifest[policy]
        assert manifest[policy]["fusion"]["kind"] == "weights"
    final = store.load_weights(manifest["final"]["fusion"])
    assert any(not torch.equal(value, before[name]) for name, value in final.items())
    saved_state = store.load_state(manifest["last"]["state"])
    assert len(saved_state["optimizer"]["param_groups"][0]["params"]) == len(list(model.fusion.parameters()))
    for policy in ("best", "final"):
        assert manifest[policy]["metrics_source"] == "reloaded_checkpoint_evaluation"
        with np.load(directory / f"{policy}_predictions.npz", allow_pickle=False) as predictions:
            assert predictions["row_uid"].tolist() == [row["row_uid"] for row in inputs["roles"]["validation"]]

    export_path = store.export_bundle(directory / "checkpoint_manifest.json", tmp_path / "export")
    exported = JointCheckpointStore(tmp_path / "export")
    exported_manifest = exported.read_manifest(export_path)
    restored = make_geo_model()
    training.load_weight_refs(exported, restored, exported_manifest["final"])
    assert restored.visual is None
    _, validation = geo_loaders(inputs, config, 1, 2, include_images=False,
                                train_transform=None, prediction_transform=None)
    metrics, predictions, _ = training.predict(restored, validation, torch.device("cpu"))
    assert np.isfinite(metrics["loss"]) and np.isfinite(predictions["logits"]).all()

    history = (directory / "history.json").read_bytes()
    original_epoch = training._epoch

    def only_evaluate(model, loader, device, optimizer=None):
        assert optimizer is None
        return original_epoch(model, loader, device, optimizer)

    def reconstruct(provenance, device):
        assert provenance == {"fixture": "tiny_geo_only_v1"}
        return make_geo_model().to(device)

    monkeypatch.setattr(joint_inputs, "make_model_from_manifest", reconstruct)
    monkeypatch.setattr(joint_inputs, "build_loaders", geo_loaders)
    monkeypatch.setattr(training, "_epoch", only_evaluate)
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("No training during finalization"))
    outcome = training.finalize_fit(config, fit, inputs, directory, store,
                                   device="cpu", make_plots=False)
    assert outcome["status"] == "complete"
    assert (directory / "history.json").read_bytes() == history
    restored_manifest = store.read_manifest(directory / "checkpoint_manifest.json")
    for policy in ("final", "best"):
        assert restored_manifest[policy]["fusion"] == manifest[policy]["fusion"]


def test_checkpoint_visual_component_must_match_model(tmp_path):
    store = JointCheckpointStore(tmp_path)
    geo_model = make_geo_model()
    geo_refs = training._weight_refs(store, geo_model)
    assert set(geo_refs) == {"fusion"}
    with pytest.raises(ValueError, match="visual component differs"):
        training.load_weight_refs(store, make_model(), geo_refs)
    with pytest.raises(ValueError, match="visual component differs"):
        training.load_weight_refs(store, geo_model, dict(geo_refs, visual=geo_refs["fusion"]))


def test_frozen_online_trains_heads_and_shares_exportable_visual_objects(
        config, fit, inputs, tmp_path, monkeypatch):
    """Final/best and independent fits share one visual object, with fresh heads."""
    store = JointCheckpointStore(tmp_path / "source")
    frozen_fit = dict(fit, regime="frozen_online", run_id="seed1/frozen_online/raw_concat/pretrained")
    visual_writes = []
    original_put_weights = store.put_weights

    def track_visual_writes(state):
        reference = original_put_weights(state)
        if any(name.startswith("trunk.") for name in state):
            visual_writes.append(reference)
        return reference

    monkeypatch.setattr(store, "put_weights", track_visual_writes)
    manifests = []
    for index, head_seed in enumerate((283, 284)):
        model = make_frozen_model(head_seed)
        initial_head = {name: value.clone() for name, value in model.fusion.state_dict().items()}
        directory = store.root / "runs" / f"frozen-{index}"
        outcome = training.train_fit(config, frozen_fit, inputs, directory, store,
            model=model, loader_factory=loader_factory, device="cpu", verification=True,
            verification_epochs=2, make_plots=False)
        assert outcome["status"] == "complete"
        manifest = store.read_manifest(directory / "checkpoint_manifest.json")
        manifests.append(manifest)
        assert manifest["provenance"]["trainable_manifest"]["trainable_visual_parameters"] == 0
        assert manifest["freeze_checks"]["end"]["values"]["checked_elements"] == sum(
            parameter.numel() for parameter in model.visual.parameters())
        assert manifest["freeze_checks"]["end"]["structure"]["frozen_gradients_absent"]
        assert all(parameter.grad is None for parameter in model.visual.parameters())
        saved_head = store.load_weights(manifest["final"]["fusion"])
        assert any(not torch.equal(value, initial_head[name]) for name, value in saved_head.items())
        visual_ref = manifest["provenance"]["frozen_visual_integrity"]["reference"]
        assert all(manifest[policy]["visual"] == visual_ref for policy in ("last", "best", "final"))
        # The initial reference and first epoch serialize once each; later epochs reuse the ref.
        assert len(visual_writes) == 2 * (index + 1)
        saved = store.load_state(manifest["last"]["state"])
        assert len(saved["optimizer"]["param_groups"][0]["params"]) == len(list(model.fusion.parameters()))

    assert all(reference == visual_writes[0] for reference in visual_writes)
    assert manifests[0]["final"]["fusion"] != manifests[1]["final"]["fusion"]
    exported_path = store.export_bundle(store.root / "runs/frozen-0/checkpoint_manifest.json",
                                       tmp_path / "export")
    exported = JointCheckpointStore(tmp_path / "export")
    manifest = exported.read_manifest(exported_path)
    restored = make_frozen_model()
    training.load_weight_refs(exported, restored, manifest["final"])
    comparison = training._verify_frozen_visual(dict(restored.visual.named_parameters()),
        manifest["provenance"]["frozen_visual_integrity"], "export", store=exported)
    assert comparison["frozen_parameters_unchanged"]
    _, validation = loader_factory(inputs, config, 1, 2)
    metrics, predictions, _ = training.predict(restored, validation, torch.device("cpu"))
    assert all(np.isfinite(predictions["logits"]).reshape(-1))
    assert np.isfinite(metrics["loss"])
    assert predictions["row_uid"].tolist() == [row["row_uid"] for row in inputs["roles"]["validation"]]


@pytest.mark.parametrize("with_buffer", [False, True])
def test_frozen_online_reuses_canonical_full_state_and_preserves_buffer_fallback(
        config, fit, inputs, tmp_path, monkeypatch, with_buffer):
    from safetensors.torch import save_file
    from multimodal.geo_helpfulness_protocol import sha256_file

    model = make_frozen_model()
    if with_buffer:
        model.visual.register_buffer("fixture_buffer", torch.tensor([7.0]))
    original = {name: value.clone() for name, value in model.visual.state_dict().items()}
    source = tmp_path / "original.safetensors"
    save_file(original, str(source))
    model._initial_visual_source = (source, "", sha256_file(source))
    state_cpu = training._state_cpu
    visual_captures = []

    def checked_state_cpu(state):
        if isinstance(state, dict) and any(str(name).startswith("trunk.") for name in state):
            assert with_buffer, "A complete canonical visual reference must not be copied from the device again"
            visual_captures.append(set(state))
        return state_cpu(state)

    monkeypatch.setattr(training, "_state_cpu", checked_state_cpu)
    store = JointCheckpointStore(tmp_path / "artifacts")
    directory = store.root / "runs/frozen"
    frozen_fit = dict(fit, regime="frozen_online", run_id="seed1/frozen_online/raw_concat/pretrained")
    training.train_fit(config, frozen_fit, inputs, directory, store, model=model,
        loader_factory=loader_factory, device="cpu", verification=True,
        verification_epochs=2, make_plots=False)
    manifest = store.read_manifest(directory / "checkpoint_manifest.json")
    reference = manifest["provenance"]["frozen_visual_integrity"]["reference"]
    visual_ref = manifest["final"]["visual"]
    assert all(manifest[policy]["visual"] == visual_ref for policy in ("last", "best"))
    assert (visual_ref == reference) is (not with_buffer)
    assert len(visual_captures) == int(with_buffer)
    assert_tree_equal(store.load_weights(visual_ref), original)
    exported_path = store.export_bundle(directory / "checkpoint_manifest.json", tmp_path / "export")
    exported = JointCheckpointStore(tmp_path / "export")
    exported_manifest = exported.read_manifest(exported_path)
    restored = make_frozen_model()
    if with_buffer:
        restored.visual.register_buffer("fixture_buffer", torch.tensor([0.0]))
    training.load_weight_refs(exported, restored, exported_manifest["final"])
    assert_tree_equal(restored.visual.state_dict(), original)


def test_best_checkpoint_order_ignores_acc3_loss_and_keeps_earliest_exact_tie():
    baseline = {"acc1": 0.6, "weighted_f1": 0.5, "mcc": 0.3, "acc3": 0.8, "loss": 0.9}
    assert training.is_better(baseline, None)
    assert not training.is_better(dict(baseline), baseline)
    assert not training.is_better({**baseline, "acc3": 1.0, "loss": 0.1}, baseline)
    assert training.is_better({**baseline, "mcc": 0.4}, baseline)
    assert training.is_better({**baseline, "weighted_f1": 0.6, "mcc": 0.0}, baseline)
    assert training.is_better({**baseline, "acc1": 0.7, "weighted_f1": 0.0, "mcc": -0.5}, baseline)
    assert not training.is_better({**baseline, "acc1": 0.5, "weighted_f1": 1.0, "mcc": 1.0}, baseline)


def test_direct_snapshot_is_independent_and_detects_tiny_actual_change(tmp_path):
    model = make_model()
    with torch.no_grad():
        model.visual.bias[0] = -0.0005629033548757434
    expected = training.frozen_visual_integrity(model)
    assert expected["reference_origin"] == "independent_initial_model_copy"
    assert expected["_snapshot"]["bias"].data_ptr() != model.visual.bias.data_ptr()
    with torch.no_grad():
        model.visual.bias.view(torch.int32)[0] ^= 0x40
    with pytest.raises(ValueError, match="Frozen parameter values changed") as error:
        training._verify_frozen_visual(dict(model.visual.named_parameters()), expected, "tiny drift")
    assert "first_index=(0,)" in str(error.value)
    assert "initial=-0.0005629033548757434" in str(error.value)
    assert "absolute_difference=3.725290298461914e-09" in str(error.value)
    assert expected["_snapshot"]["bias"][0].item() == -0.0005629033548757434


@pytest.mark.parametrize("stored", [False, True])
def test_production_frozen_reference_uses_original_checkpoint_not_changed_live_copy(tmp_path, stored):
    from safetensors.torch import save_file
    from multimodal.geo_helpfulness_protocol import sha256_file

    model = make_model()
    original_bias = model.visual.bias.detach().clone()
    source = tmp_path / "original_visual.safetensors"
    save_file({"visual." + name: value for name, value in model.visual.state_dict().items()}, str(source))
    model._initial_visual_source = (source, "visual.", sha256_file(source))
    with torch.no_grad():
        model.visual.bias.view(torch.int32)[0] ^= 0x40
    changed_bias = model.visual.bias.detach().clone()
    store = JointCheckpointStore(tmp_path / "artifacts") if stored else None
    expected = training.frozen_visual_integrity(model, store=store)
    assert expected["reference_origin"] == "authenticated_original_encoder_checkpoint"
    reference = store.load_weights(expected["reference"]) if stored else expected["_snapshot"]
    assert torch.equal(reference["bias"], original_bias)
    assert torch.equal(model.visual.bias, changed_bias)  # Capturing a reference does not repair the model.
    with pytest.raises(ValueError, match="Frozen parameter values changed"):
        training._verify_frozen_visual(dict(model.visual.named_parameters()), expected,
                                      "source-anchored check", store=store)


def test_direct_comparison_treats_signed_zero_as_equal():
    model = make_model()
    with torch.no_grad():
        model.visual.bias[0] = -0.0
    expected = training.frozen_visual_integrity(model)
    with torch.no_grad():
        model.visual.bias[0] = 0.0
    assert training._verify_frozen_visual(dict(model.visual.named_parameters()), expected, "zero")["status"] == "exact"


def test_stored_initial_reference_is_serializable_and_directly_compared(tmp_path):
    model = make_model()
    store = JointCheckpointStore(tmp_path)
    expected = training.frozen_visual_integrity(model, store=store)
    json.dumps(expected)
    assert expected["algorithm"] == "direct_frozen_values_v1"
    assert "sha256" not in expected  # Only the storage object, not live values, is hashed.
    assert expected["reference"]["kind"] == "weights"
    assert training._verify_frozen_visual(dict(model.visual.named_parameters()), expected,
                                           "stored", store=store)["checked_tensors"] == 1


def test_empty_frozen_scope_has_valid_comparison_without_empty_weight_object(tmp_path):
    model = make_model()
    model.visual.bias.requires_grad_(True)
    store = JointCheckpointStore(tmp_path)
    expected = training.frozen_visual_integrity(model, store=store)
    assert expected["parameters"] == [] and expected["reference"] is None
    assert training._verify_frozen_visual(dict(model.visual.named_parameters()), expected,
                                           "empty", store=store)["checked_elements"] == 0


def test_freezing_structure_checks_optimizer_exclusion_and_gradients():
    model = make_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    assert training.verify_freezing(model, optimizer)["frozen_gradients_absent"]
    model.visual.bias.grad = torch.ones_like(model.visual.bias)
    with pytest.raises(ValueError, match="Frozen parameter has a gradient"):
        training.verify_freezing(model, optimizer)
    model.visual.bias.grad = None
    with pytest.raises(ValueError, match="missing its gradient"):
        training.verify_freezing(model, optimizer, require_gradients=True)
    optimizer.add_param_group({"params": [model.visual.bias]})
    with pytest.raises(ValueError, match="Optimizer parameters differ"):
        training.verify_freezing(model, optimizer)


def test_freezing_structure_rejects_changed_requires_grad():
    model = make_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    model.visual.bias.requires_grad_(True)
    with pytest.raises(ValueError, match="Optimizer parameters differ"):
        training.verify_freezing(model, optimizer)


def test_frozen_change_fails_at_end_before_final_checkpoint(config, fit, inputs, tmp_path, monkeypatch):
    original_epoch = training._epoch
    changed = False

    def mutate_frozen(model, loader, device, optimizer=None):
        nonlocal changed
        result = original_epoch(model, loader, device, optimizer)
        if optimizer is not None and not changed:
            with torch.no_grad():
                model.visual.bias.view(torch.int32)[0] ^= 1
            changed = True
        return result

    monkeypatch.setattr(training, "_epoch", mutate_frozen)
    store = JointCheckpointStore(tmp_path)
    with pytest.raises(ValueError, match="at fit end before checkpoint publication"):
        run(config, fit, inputs, store)
    manifest = store.read_manifest(tmp_path / "runs/run/checkpoint_manifest.json")
    assert manifest["last"]["epoch"] == 1 and "final" not in manifest
    status = json.loads((tmp_path / "runs/run/run.json").read_text())
    assert status["status"] == "failed_or_interrupted" and status["completed_epoch"] == 1
    assert len(json.loads((tmp_path / "runs/run/history.json").read_text())) == 2


def test_frozen_comparison_occurs_only_at_start_and_end(config, fit, inputs, tmp_path, monkeypatch):
    original = training._verify_frozen_visual
    contexts = []

    def capture(snapshot, expected, context, **kwargs):
        contexts.append(context)
        return original(snapshot, expected, context, **kwargs)

    monkeypatch.setattr(training, "_verify_frozen_visual", capture)
    run(config, fit, inputs, JointCheckpointStore(tmp_path))
    assert contexts == ["at fit start", "at fit end before checkpoint publication"]


def test_authoritative_reload_metrics_do_not_require_identity_with_epoch_metrics(
        config, fit, inputs, tmp_path, monkeypatch):
    original = training.predict
    calls = 0

    def distinguish_reload(model, loader, device):
        nonlocal calls
        calls += 1
        metrics, predictions, duration = original(model, loader, device)
        if calls > 2:
            metrics = dict(metrics, loss=metrics["loss"] + 0.125)
        return metrics, predictions, duration

    monkeypatch.setattr(training, "predict", distinguish_reload)
    store = JointCheckpointStore(tmp_path)
    assert run(config, fit, inputs, store)["status"] == "complete"
    manifest = store.read_manifest(tmp_path / "runs/run/checkpoint_manifest.json")
    metrics = json.loads((tmp_path / "runs/run/metrics.json").read_text())
    history = json.loads((tmp_path / "runs/run/history.json").read_text())
    assert metrics["metrics_source"] == "reloaded_checkpoint_evaluation"
    assert metrics["final"]["metrics"] == manifest["final"]["metrics"]
    assert metrics["final"]["metrics"]["loss"] == pytest.approx(history[-1]["validation"]["loss"] + 0.125)
    assert manifest["final"]["training_metrics"] == history[-1]["validation"]
    assert not manifest["checkpoint_evaluation"]["final"]["compared_for_exact_identity"]


def test_loading_checks_file_integrity_structure_and_finiteness(tmp_path):
    store = JointCheckpointStore(tmp_path)
    model = make_model()
    refs = training._weight_refs(store, model)
    training.load_weight_refs(store, model, refs)
    bad = store.load_weights(refs["fusion"])
    name = next(iter(bad))
    bad[name] = torch.full_like(bad[name], float("nan"))
    changed = dict(refs, fusion=store.put_weights(bad))
    with pytest.raises(ValueError, match="Nonfinite state"):
        training.load_weight_refs(store, model, changed)
    path = store.resolve(refs["visual"])
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="size changed"):
        training.load_weight_refs(store, model, refs)


@pytest.mark.parametrize("mutation", ["keys", "shape", "dtype"])
def test_loading_rejects_incompatible_state(tmp_path, mutation):
    store = JointCheckpointStore(tmp_path)
    model = make_model()
    refs = training._weight_refs(store, model)
    bad = store.load_weights(refs["fusion"])
    name = next(iter(bad))
    if mutation == "keys":
        bad["wrong_name"] = bad.pop(name)
    elif mutation == "shape":
        bad[name] = bad[name].reshape(-1, 1)
    else:
        bad[name] = bad[name].double()
    changed = dict(refs, fusion=store.put_weights(bad))
    with pytest.raises(ValueError, match="keys changed|shape or dtype"):
        training.load_weight_refs(store, model, changed)


def test_inference_load_has_no_byte_identity_gate(tmp_path, monkeypatch):
    store = JointCheckpointStore(tmp_path)
    model = make_model()
    refs = training._weight_refs(store, model)
    original = model.fusion.load_state_dict

    def changed_finite_copy(state, **kwargs):
        result = original(state, **kwargs)
        with torch.no_grad():
            next(model.fusion.parameters()).reshape(-1).view(torch.int32)[0] ^= 1
        return result

    monkeypatch.setattr(model.fusion, "load_state_dict", changed_finite_copy)
    training.load_weight_refs(store, model, refs)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_exported_bundle_loads_with_initial_snapshot_dependency(config, fit, inputs, tmp_path):
    source = JointCheckpointStore(tmp_path / "source")
    run(config, fit, inputs, source)
    exported_path = source.export_bundle(source.root / "runs/run/checkpoint_manifest.json", tmp_path / "export")
    exported = JointCheckpointStore(tmp_path / "export")
    manifest = exported.read_manifest(exported_path)
    restored = make_model()
    training.load_weight_refs(exported, restored, manifest["final"])
    training._verify_frozen_visual(dict(restored.visual.named_parameters()),
                                  manifest["provenance"]["frozen_visual_integrity"], "export", store=exported)
    _, validation = loader_factory(inputs, config, 1, 2)
    metrics, predictions, _ = training.predict(restored, validation, torch.device("cpu"))
    assert metrics == manifest["final"]["metrics"]
    with np.load(source.root / "runs/run/final_predictions.npz", allow_pickle=False) as saved:
        assert all(np.array_equal(predictions[key], saved[key]) for key in predictions)


def test_archived_attempt_references_survive_active_pruning(config, fit, inputs, tmp_path):
    store = JointCheckpointStore(tmp_path)
    run(config, fit, inputs, store, stop_after_epoch=1)
    original = tmp_path / "runs/run"
    archived = tmp_path / "interrupted_attempts/seed1/run/attempt1"
    archived.parent.mkdir(parents=True)
    original.rename(archived)
    old = store.read_manifest(archived / "checkpoint_manifest.json")
    assert run(config, fit, inputs, store)["status"] == "complete"
    assert store.read_manifest(archived / "checkpoint_manifest.json")["last"] == old["last"]
    store.load_state(old["last"]["state"])


def test_nonfinite_gradient_stops_before_first_checkpoint(config, fit, inputs, tmp_path, monkeypatch):
    original = make_model

    def poisoned_model():
        model = original()
        next(model.fusion.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        return model

    monkeypatch.setattr(__import__(__name__, fromlist=["make_model"]), "make_model", poisoned_model)
    store = JointCheckpointStore(tmp_path)
    with pytest.raises(ValueError, match="Nonfinite gradient"):
        run(config, fit, inputs, store)
    assert not (tmp_path / "runs/run/checkpoint_manifest.json").exists()


def prepare_finalizer(monkeypatch):
    from multimodal import attention_fusion_joint_inputs as joint_inputs

    def reconstruct(provenance, device):
        assert provenance == {"fixture": "tiny_online_v1"}
        return make_model().to(device)

    monkeypatch.setattr(joint_inputs, "make_model_from_manifest", reconstruct)
    monkeypatch.setattr(joint_inputs, "build_loaders", loader_factory)


def test_finalization_reconstructs_and_only_evaluates_original_weights(
        config, fit, inputs, tmp_path, monkeypatch):
    store = JointCheckpointStore(tmp_path)
    run(config, fit, inputs, store)
    directory = tmp_path / "runs/run"
    manifest_before = store.read_manifest(directory / "checkpoint_manifest.json")
    preserved = {name: (directory / name).read_bytes() for name in (
        "run_spec.json", "config.json", "history.json", "geo_preprocessing.json",
        "freeze_checks.json", "final_training_predictions.npz", "best_training_predictions.npz")}
    status = json.loads((directory / "run.json").read_text())
    status.update(status="failed_or_interrupted", error="original finalization failure")
    (directory / "run.json").write_text(json.dumps(status))
    prepare_finalizer(monkeypatch)
    original_epoch = training._epoch
    calls = []

    def only_evaluate(model, loader, device, optimizer=None):
        assert optimizer is None
        calls.append("evaluation")
        return original_epoch(model, loader, device, optimizer)

    def forbid_training_state(*args, **kwargs):
        pytest.fail("Evaluation-only recovery must not construct/load training state")

    monkeypatch.setattr(training, "_epoch", only_evaluate)
    monkeypatch.setattr(torch.optim, "Adam", forbid_training_state)
    monkeypatch.setattr(store, "load_state", forbid_training_state)
    guards = []
    result = training.finalize_fit(config, fit, inputs, directory, store, device="cpu",
                                  make_plots=False, guard=lambda: guards.append("checked"))
    assert result["status"] == "complete" and result["completed_epoch"] == 2
    assert "error" not in result and calls == ["evaluation", "evaluation"]
    assert len(guards) == 4
    assert all((directory / name).read_bytes() == content for name, content in preserved.items())
    manifest_after = store.read_manifest(directory / "checkpoint_manifest.json")
    assert manifest_after["provenance"] == manifest_before["provenance"]
    assert manifest_after["run_identity"] == manifest_before["run_identity"]
    for policy in ("final", "best"):
        for component in ("visual", "fusion"):
            assert manifest_after[policy][component] == manifest_before[policy][component]
    attempts = sorted((directory / "finalization_attempts").glob("*.json"))
    attempt = json.loads(attempts[-1].read_text())
    assert attempt["training_epochs_performed"] == 0
    assert attempt["previous_run"]["error"] == "original finalization failure"
    assert attempt["completed_policies"] == ["final", "best"]


def test_best_checksum_failure_keeps_final_evaluation_then_recovers_without_training(
        config, fit, inputs, tmp_path, monkeypatch):
    store = JointCheckpointStore(tmp_path)
    original_load = training.load_weight_refs
    calls = 0

    def fail_best(store, model, refs, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            ref = refs["visual"]
            raise JointCheckpointIntegrityError(ref["path"], ref["sha256"], "b" * 64, 3)
        return original_load(store, model, refs, **kwargs)

    monkeypatch.setattr(training, "load_weight_refs", fail_best)
    with pytest.raises(JointCheckpointIntegrityError):
        run(config, fit, inputs, store)
    directory = tmp_path / "runs/run"
    history_before = (directory / "history.json").read_bytes()
    manifest = store.read_manifest(directory / "checkpoint_manifest.json")
    status = json.loads((directory / "run.json").read_text())
    assert status["status"] == "checkpoint_unresolved"
    assert status["completed_epoch"] == 2
    assert status["checkpoint_unresolved"]["policy"] == "best"
    assert set(manifest["checkpoint_evaluation"]) == {"final"}
    assert (directory / "final_predictions.npz").is_file()
    assert not (directory / "best_predictions.npz").exists()
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["metrics_source"] == "partial_reloaded_checkpoint_evaluation"
    assert metrics["final"]["metrics_source"] == "reloaded_checkpoint_evaluation"
    assert metrics["best"]["metrics_source"] == "epoch_validation"
    failed_attempt = next((directory / "finalization_attempts").glob("*.json"))
    assert json.loads(failed_attempt.read_text())["completed_policies"] == ["final"]
    evidence = failed_attempt.read_bytes()

    monkeypatch.setattr(training, "load_weight_refs", original_load)
    prepare_finalizer(monkeypatch)
    original_epoch = training._epoch

    def reject_training(model, loader, device, optimizer=None):
        assert optimizer is None
        return original_epoch(model, loader, device, optimizer)

    monkeypatch.setattr(training, "_epoch", reject_training)
    recovered = training.finalize_fit(config, fit, inputs, directory, store,
                                      device="cpu", make_plots=False)
    assert recovered["status"] == "complete"
    assert "checkpoint_unresolved" not in recovered
    assert (directory / "best_predictions.npz").exists()
    assert (directory / "history.json").read_bytes() == history_before
    assert failed_attempt.read_bytes() == evidence
    assert len(list((directory / "finalization_attempts").glob("*.json"))) == 2


@pytest.mark.parametrize("changed", ["configuration", "fit", "inputs", "preprocessing", "ontology"])
def test_finalization_rejects_identity_changes_before_mutation(
        config, fit, inputs, tmp_path, monkeypatch, changed):
    store = JointCheckpointStore(tmp_path)
    run(config, fit, inputs, store)
    directory = tmp_path / "runs/run"
    before = {path.relative_to(directory): path.read_bytes()
              for path in directory.rglob("*") if path.is_file()}
    config, fit, inputs = deepcopy(config), deepcopy(fit), deepcopy(inputs)
    if changed == "configuration":
        config["training"]["learning_rate"] *= 2
    elif changed == "fit":
        fit["seed"] = 2
    elif changed == "inputs":
        inputs["provenance"]["fixture"] = "other-data"
    elif changed == "preprocessing":
        inputs["scaler"]["mode"] = "native"
    else:
        inputs["source_config"]["class_ontology"]["classes"][0]["label_name"] = "other-class"
    with pytest.raises(ValueError, match="differs from original"):
        training.finalize_fit(config, fit, inputs, directory, store,
                              device="cpu", make_plots=False)
    after = {path.relative_to(directory): path.read_bytes()
             for path in directory.rglob("*") if path.is_file()}
    assert before == after


def test_finalization_rejects_unfinished_training(config, fit, inputs, tmp_path):
    store = JointCheckpointStore(tmp_path)
    run(config, fit, inputs, store, stop_after_epoch=1)
    directory = tmp_path / "runs/run"
    with pytest.raises(ValueError, match="complete declared training history"):
        training.finalize_fit(config, fit, inputs, directory, store,
                              device="cpu", make_plots=False)
    assert not (directory / "finalization_attempts").exists()


@pytest.mark.parametrize("changed", ["selection", "freeze_check", "run_identity"])
def test_finalization_requires_original_selection_and_successful_freeze_check(
        config, fit, inputs, tmp_path, changed):
    store = JointCheckpointStore(tmp_path)
    run(config, fit, inputs, store)
    directory = tmp_path / "runs/run"
    manifest = store.read_manifest(directory / "checkpoint_manifest.json")
    if changed == "selection":
        manifest["best"]["selection_metrics"]["acc1"] += 0.01
        expected = "selection differs"
    elif changed == "freeze_check":
        manifest["freeze_checks"]["end"]["values"]["frozen_parameters_unchanged"] = False
        (directory / "freeze_checks.json").write_text(json.dumps(manifest["freeze_checks"]))
        expected = "successful end-of-fit freezing"
    else:
        manifest["run_identity"] = "b" * 64
        expected = "run identity"
    store.write_manifest(directory / "checkpoint_manifest.json", manifest)
    with pytest.raises(ValueError, match=expected):
        training.finalize_fit(config, fit, inputs, directory, store,
                              device="cpu", make_plots=False)
