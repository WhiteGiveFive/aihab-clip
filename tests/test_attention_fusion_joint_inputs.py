"""M3J input isolation, shared views, train-only scaling and visual gradients."""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

from multimodal import attention_fusion_joint_inputs as joint
from multimodal.geo_helpfulness_protocol import sha256_file

REPO = Path(__file__).resolve().parents[1]


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Module()
        self.trunk.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 2), nn.GELU(), nn.Dropout(0.1)) for _ in range(27)
        ])
        self.trunk.norm = nn.LayerNorm(2)
        self.trunk.attn_pool = nn.Linear(2, 1152)

    def forward(self, value):
        for block in self.trunk.blocks:
            value = value + block(value)
        return self.trunk.attn_pool(self.trunk.norm(value))


@pytest.fixture
def config():
    return {
        "study_id": "attention_fusion_joint_v1", "initialization_seed_namespace": "attention_fusion_v1",
        "base_config": "configs/multimodal_attention_fusion.yaml",
        "bypass_config": "configs/multimodal_attention_fusion_cls_bypass.yaml",
        "source_config": "configs/multimodal_geo_helpfulness.yaml",
        "training_seeds": [1, 2, 3, 4], "authorized_scope": {"seeds": [1]},
        "training": {"batch_size": 2, "num_workers": 0,
                     "dropout_seed_namespace": "attention_fusion_joint_v1"},
        "paths": {"artifact_root": "artifacts"},
    }


def test_augmentation_stream_is_repeatable_and_does_not_consume_global_rngs():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

    def draw():
        return random.random(), np.random.random(), torch.rand(3)

    seed = joint.augmentation_seed("study", 1, 2, "image-a")
    with joint.isolated_augmentation_rng(seed):
        first = draw()
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    draw()
    with joint.isolated_augmentation_rng(seed):
        replay = draw()
    assert first[:2] == replay[:2]
    assert torch.equal(first[2], replay[2])
    assert seed != joint.augmentation_seed("study", 1, 3, "image-a")
    assert seed != joint.augmentation_seed("study", 2, 2, "image-a")


def test_augmentation_restores_rng_even_if_transform_fails():
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with joint.isolated_augmentation_rng(17):
            torch.rand(5)
            raise RuntimeError("transform failed")
    assert torch.equal(before, torch.get_rng_state())


def test_partial_visual_modes_and_gradient_reach_match_exact_declared_subset(config):
    model = joint.make_model(config, "image_only", "pretrained", 1, "joint_partial", "cpu",
                             visual=TinyVisual(), strict_scope=False)
    assert model.visual.training is False
    assert all(not model.visual.trunk.blocks[i].training for i in range(17))
    assert all(model.visual.trunk.blocks[i].training for i in range(17, 27))
    logits = model(torch.randn(3, 2), torch.randn(3, 64))
    nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    for name, parameter in model.visual.named_parameters():
        if joint._partial_name(name):
            assert parameter.requires_grad and parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        else:
            assert not parameter.requires_grad and parameter.grad is None
    assert model.fusion.classifier.net[0].weight.grad is not None
    model.eval()
    assert all(not module.training for module in model.modules())
    model.train()
    assert not model.visual.trunk.blocks[0].training
    assert model.visual.trunk.blocks[26].training


def test_frozen_encoder_stays_eval_and_has_no_gradients(config):
    model = joint.make_model(config, "raw_concat", "cs_adapted", 1, "frozen_online", "cpu",
                             visual=TinyVisual(), strict_scope=False)
    initial_visual = {name: value.detach().clone() for name, value in model.visual.state_dict().items()}
    initial_head = {name: value.detach().clone() for name, value in model.fusion.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    model.train()
    assert all(not module.training for module in model.visual.modules())
    image = torch.randn(3, 2, requires_grad=True)
    assert not model.encode_image(image).requires_grad
    model(image, torch.randn(3, 64)).sum().backward()
    assert all(not p.requires_grad and p.grad is None for p in model.visual.parameters())
    assert image.grad is None
    assert any(p.grad is not None for p in model.fusion.parameters())
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized.isdisjoint(id(p) for p in model.visual.parameters())
    optimizer.step()
    assert all(torch.equal(value, initial_visual[name]) for name, value in model.visual.state_dict().items())
    assert any(not torch.equal(value, initial_head[name]) for name, value in model.fusion.state_dict().items())
    assert model.trainable_manifest["trainable_visual_parameters"] == 0
    model.eval()
    model.train()
    assert all(not module.training for module in model.visual.modules())


@pytest.mark.parametrize("candidate", ["image_only", "raw_concat", "projected_concat", "tokenwise_capacity",
                                      "attention_two_token", "attention_cls", "attention_cls_skip_adapted",
                                      "attention_cls_skip_original"])
def test_all_eight_online_models_preserve_fusion_definitions_and_global_rng(config, candidate):
    first_visual, second_visual = TinyVisual(), TinyVisual()
    before = torch.get_rng_state().clone()
    first = joint.make_model(config, candidate, "pretrained", 1, "joint_partial", "cpu",
                             visual=first_visual, strict_scope=False)
    second = joint.make_model(config, candidate, "cs_adapted", 1, "joint_partial", "cpu",
                              visual=second_visual, strict_scope=False)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(tensor, second.fusion.state_dict()[name])
               for name, tensor in first.fusion.state_dict().items())
    first.eval()
    with torch.no_grad():
        assert first(torch.randn(2, 2), torch.randn(2, 64)).shape == (2, 18)


def test_geo_only_never_constructs_or_calls_visual_encoder(config):
    model = joint.make_model(config, "geo_only", "not_applicable", 1, "geo_only", "cpu")
    assert model.visual is None
    assert model(torch.zeros(3, 0), torch.randn(3, 64)).shape == (3, 18)
    assert model.provenance["initialization"] == "not_applicable"


def test_geo_only_rejects_an_image_initialization_arm(config):
    with pytest.raises(joint.JointInputError, match="initialization-independent"):
        joint.make_model(config, "geo_only", "pretrained", 1, "geo_only", "cpu")


def test_geo_loaders_use_declared_shuffle_without_loading_images(config, monkeypatch):
    from multimodal import geo_helpfulness_oof
    monkeypatch.setattr(geo_helpfulness_oof, "_LegacyImageDataset",
                        lambda *a, **k: pytest.fail("Geo-only loaded an image"))
    rows = pd.DataFrame({"row_uid": [str(i) for i in range(30)], "label_id_dense": [0] * 30})
    role = {"rows": rows, "geo": np.ones((30, 64)), "label": np.zeros(30, dtype=int)}
    inputs = {"roles": {"train": role, "validation": role}, "source_config": {}}
    config["training"]["shuffle_seed_namespace"] = "declared_shared_namespace"

    def collect():
        loader, _ = joint.build_loaders(inputs, config, 1, 1, train_transform=None,
                                        prediction_transform=None, include_images=False)
        batches = list(loader)
        assert all(b["image"].shape[1] == 0 for b in batches)
        return [uid for b in batches for uid in b["row_uid"]]

    before = collect()
    config["study_id"] = "another_package_identity"
    assert collect() == before
    config["training"]["shuffle_seed_namespace"] = "different_namespace"
    assert collect() != before


def test_manifest_reconstruction_needs_no_original_config_or_checkpoint_paths(config, monkeypatch):
    import open_clip.model

    original = joint.make_model(config, "attention_cls_skip_original", "cs_adapted", 1,
                                "joint_partial", "cpu", visual=TinyVisual(), strict_scope=False)
    source = yaml.safe_load((REPO / config["source_config"]).read_text())
    provenance = deepcopy(original.provenance)
    provenance.update(
        encoder_architecture={"embed_dim": 1152, "vision_cfg": {"timm_model_pretrained": False}},
        encoder_builder={"library": "open_clip_torch", "version": joint.version("open_clip_torch"),
                         "callable": "open_clip.model._build_vision_tower"},
        historical_preprocessing_recipe=source["experts"]["image_encoder"][
            "fold_contained_adaptation"]["adaptation_recipe"]["preprocessing"],
    )
    monkeypatch.setattr(joint, "_yaml", lambda *args: pytest.fail("Original configuration opened"))
    monkeypatch.setattr(open_clip.model, "_build_vision_tower", lambda **kwargs: TinyVisual())
    restored = joint.make_model_from_manifest(provenance, "cpu", strict_scope=False)
    restored.visual.load_state_dict(original.visual.state_dict(), strict=True)
    restored.fusion.load_state_dict(original.fusion.state_dict(), strict=True)
    original.eval()
    restored.eval()
    image, geo = torch.randn(2, 2), torch.randn(2, 64)
    with torch.no_grad():
        assert torch.equal(original(image, geo), restored(image, geo))
    assert restored.prediction_transform is not None
    assert tuple(restored.train_transform.transforms[0].ratio) == (0.75, 1.3333)


def test_strict_scope_refuses_wrong_model_before_training(config):
    with pytest.raises(joint.JointInputError, match="parameter count"):
        joint.make_model(config, "image_only", "pretrained", 1, "joint_partial", "cpu",
                         visual=TinyVisual())


@pytest.mark.parametrize("initialization", ["pretrained", "cs_adapted"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_direct_visual_loading_never_materializes_text_or_aliases_file(
        config, tmp_path, monkeypatch, initialization, seed):
    import json
    import safetensors
    from safetensors.torch import save_file
    from multimodal import geo_helpfulness_oof

    base = yaml.safe_load((REPO / config["base_config"]).read_text())
    source = yaml.safe_load((REPO / config["source_config"]).read_text())
    for relative, value in ((config["base_config"], base), (config["source_config"], source),
                            (base["paths"]["source_protocol_root"] + "/protocol/resolved_protocol.yaml", source)):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(value))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot.joinpath("open_clip_config.json").write_text(json.dumps({
        "model_cfg": {"embed_dim": 1152, "vision_cfg": {"timm_model_pretrained": False}},
    }))
    pretrained, adapted = TinyVisual(), TinyVisual()
    pinned = snapshot / "open_clip_model.safetensors"
    save_file({**{"visual." + key: value for key, value in pretrained.state_dict().items()},
               "text.not_used": torch.ones(3)}, str(pinned))
    adapted_path = tmp_path / base["paths"]["encoder_template"].format(seed=seed)
    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(adapted.state_dict(), str(adapted_path))
    hashes_before = [sha256_file(path) for path in (pinned, adapted_path)]
    records = [{"seed": seed, "checkpoint": {"sha256": hashes_before[1]}}]
    if seed != 1:
        # A valid but different seed-1 encoder must not be selected by position.
        seed1_path = tmp_path / base["paths"]["encoder_template"].format(seed=1)
        seed1_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(pretrained.state_dict(), str(seed1_path))
        records.insert(0, {"seed": 1, "checkpoint": {"sha256": sha256_file(seed1_path)}})
    monkeypatch.setattr(joint.audited_inputs, "_bind_m1", lambda *args: {
        "audit": {"encoders": records},
    })
    monkeypatch.setattr(geo_helpfulness_oof, "_pinned_snapshot", lambda *args: (
        snapshot, {"open_clip_model.safetensors": hashes_before[0]}))
    monkeypatch.setattr(geo_helpfulness_oof, "_load_fresh_encoder", lambda *args:
                        pytest.fail("Unused text tower was constructed"))
    monkeypatch.setattr(joint, "_construct_visual_and_transforms", lambda *args: (TinyVisual(), None, None))
    original_safe_open = safetensors.safe_open
    keys_read = []

    class CheckedOpen:
        def __init__(self, *args, **kwargs):
            self.handle = original_safe_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_tensor(self, key):
            assert not key.startswith("text.")
            keys_read.append(key)
            return self.handle.get_tensor(key)

    monkeypatch.setattr(safetensors, "safe_open", CheckedOpen)
    model = joint.make_model(config, "image_only", initialization, seed, "joint_partial", "cpu",
                             root=tmp_path, strict_scope=False)
    reference = pretrained if initialization == "pretrained" else adapted
    assert keys_read
    assert all(torch.equal(value, reference.state_dict()[key]) for key, value in model.visual.state_dict().items())
    with torch.no_grad():
        next(model.visual.parameters()).add_(1)
    assert [sha256_file(path) for path in (pinned, adapted_path)] == hashes_before
    assert model.provenance["visual_loading"]["materialized_text_tensors"] is False
    assert model.provenance["visual_loading"]["post_transfer_exact_source_values"] is True
    assert model.provenance["visual_loading"]["post_transfer_source_sha256_verified"] is True
    if initialization == "cs_adapted":
        assert model.provenance["adapted_checkpoint"]["path"] == str(adapted_path.relative_to(tmp_path))
        assert model.provenance["adapted_checkpoint"]["sha256"] == hashes_before[1]

    original_to = joint.OnlineFusionModel.to

    def corrupt_transfer(instance, *args, **kwargs):
        result = original_to(instance, *args, **kwargs)
        with torch.no_grad():
            weight = instance.visual.trunk.blocks[5][0].weight
            weight.view(torch.int32).reshape(-1)[0] ^= 0x40
        return result

    monkeypatch.setattr(joint.OnlineFusionModel, "to", corrupt_transfer)
    with pytest.raises(joint.JointInputError, match="after device transfer: trunk.blocks.5.0.weight"):
        joint.make_model(config, "image_only", initialization, seed, "joint_partial", "cpu",
                         root=tmp_path, strict_scope=False)
    assert [sha256_file(path) for path in (pinned, adapted_path)] == hashes_before


@pytest.fixture
def prepared(tmp_path, monkeypatch, config):
    (tmp_path / "configs").mkdir()
    base = yaml.safe_load((REPO / config["base_config"]).read_text())
    base["data"].update(train_rows=2, validation_rows=1, geo_columns=["a", "b"])
    (tmp_path / config["base_config"]).write_text(yaml.safe_dump(base))
    (tmp_path / config["source_config"]).write_text("development_universe: {}\n")
    rows = pd.DataFrame({"row_uid": ["a", "b", "c"], "plot_idx": ["p1", "p2", "p3"],
                         "label_id_dense": [0, 1, 2]})
    context = {
        "source_config": {"development_universe": {}},
        "roles": {"train": {"rows": rows.iloc[:2], "geo": np.array([[1., 3.], [3., 5.]])},
                  "validation": {"rows": rows.iloc[2:], "geo": np.array([[100., 200.]])}},
        "audit": {"protocol_files": {"development_assignments.parquet": {"sha256": "a" * 64}},
                  "development_geo_source_files": [], "encoders": [{"seed": 1}]},
        "m1_binding": {"manifest_file_sha256": "b" * 64},
    }
    monkeypatch.setattr(joint.audited_inputs, "audit_source_inputs", lambda *args, **kwargs: context)
    monkeypatch.setattr(joint.audited_inputs, "_image_inventory", lambda *args: [{"rows": 3}])
    monkeypatch.setattr(joint.audited_inputs, "_validate_historical_image_fingerprint", lambda *args: {"matches": True})
    return config, tmp_path, context


def test_preparation_fits_only_train_and_reuses_shared_files_across_seeds(prepared):
    config, root, context = prepared
    inputs = joint.prepare_inputs(config, root)
    assert inputs["scaler"]["mean"] == [2., 4.]
    np.testing.assert_array_equal(inputs["roles"]["train"]["geo"], [[-1., -1.], [1., 1.]])
    np.testing.assert_array_equal(inputs["roles"]["validation"]["geo"], [[98., 196.]])
    saved = Path(inputs["manifest_path"])
    original = saved.read_bytes()
    later_seed = deepcopy(config)
    later_seed["authorized_scope"]["seeds"] = [2]
    joint.prepare_inputs(later_seed, root)
    assert saved.read_bytes() == original
    context["roles"]["train"]["geo"][0, 0] = 2
    with pytest.raises(joint.JointInputError, match="provenance changed"):
        joint.prepare_inputs(config, root)


def test_geo_preparation_authenticates_geo_without_unused_image_content(prepared, monkeypatch):
    config, root, context = prepared
    config["authorized_scope"]["regime"] = "geo_only"
    calls = []

    def audit(*args, **kwargs):
        calls.append(kwargs)
        return context

    monkeypatch.setattr(joint.audited_inputs, "audit_source_inputs", audit)
    monkeypatch.setattr(joint.audited_inputs, "_image_inventory",
                        lambda *a: pytest.fail("Geo-only decoded/hashes unused images"))
    monkeypatch.setattr(joint.audited_inputs, "_validate_historical_image_fingerprint",
                        lambda *a: pytest.fail("Geo-only fingerprints unused images"))
    result = joint.prepare_inputs(config, root)
    assert calls[0]["authenticate_image_producers"] is False
    assert "image_fingerprint" not in result["provenance"]
    assert "image_inventory_sha256" not in result["provenance"]
    assert result["provenance"]["test_features_or_labels_loaded"] is False
    assert not (root / "artifacts/inputs/image_inventory.json").exists()
    assert result["roles"]["train"]["geo"].dtype == np.float32
    assert np.array_equal(result["roles"]["train"]["geo"], [[-1., -1.], [1., 1.]])


def test_corrupted_prepared_file_rejected(prepared):
    config, root, _ = prepared
    inputs = joint.prepare_inputs(config, root)
    target = Path(inputs["manifest_path"]).parent / "train.npz"
    target.write_bytes(target.read_bytes() + b"corruption")
    with pytest.raises(joint.JointInputError, match="hash mismatch"):
        joint.prepare_inputs(config, root)


def test_overlapping_plots_refused_before_output(prepared):
    config, root, context = prepared
    context["roles"]["validation"]["rows"] = context["roles"]["validation"]["rows"].copy()
    context["roles"]["validation"]["rows"].loc[:, "plot_idx"] = "p1"
    with pytest.raises(joint.JointInputError, match="plots overlap"):
        joint.prepare_inputs(config, root)
    assert not (root / "artifacts/inputs").exists()


def test_loaders_pair_shuffling_and_views_independently_of_dropout(config, monkeypatch):
    from multimodal import geo_helpfulness_oof

    class ImageFixture:
        def __init__(self, rows, **kwargs):
            self.labels = rows.label_id_dense.tolist()

        def __getitem__(self, index):
            return torch.rand(3), self.labels[index]

    monkeypatch.setattr(geo_helpfulness_oof, "_LegacyImageDataset", ImageFixture)
    rows = pd.DataFrame({"row_uid": ["a", "b", "c", "d", "e"], "label_id_dense": list(range(5))})
    role = {"rows": rows, "geo": np.arange(10).reshape(5, 2), "label": np.arange(5)}
    inputs = {"roles": {"train": role, "validation": role}, "source_config": {}}

    def collect(epoch):
        loader, _ = joint.build_loaders(inputs, config, 1, epoch, train_transform=None, prediction_transform=None)
        result = []
        for batch in loader:
            result.extend(zip(batch["row_uid"], batch["image"], batch["label"].tolist()))
            torch.rand(7)  # Simulate a model-specific dropout stream.
        return result

    first, replay, later = collect(1), collect(1), collect(2)
    assert [row[0] for row in first] == [row[0] for row in replay]
    assert len(first) == 5  # The smaller final batch is retained.
    assert all(torch.equal(a[1], b[1]) and a[2] == b[2] for a, b in zip(first, replay))
    first_map = {uid: tensor for uid, tensor, _ in first}
    assert any(not torch.equal(first_map[uid], tensor) for uid, tensor, _ in later)


@pytest.fixture
def collect_image_loader(monkeypatch):
    """Exercise batching and transformed images while avoiding image-file IO."""
    from multimodal import geo_helpfulness_oof

    class ImageFixture:
        def __init__(self, rows, *, transform, **kwargs):
            self.labels = rows.label_id_dense.tolist()
            self.transform = transform

        def __getitem__(self, index):
            return self.transform(index), self.labels[index]

    def transform(index):
        return torch.tensor([index, random.random(), np.random.random(), torch.rand(()).item()])

    monkeypatch.setattr(geo_helpfulness_oof, "_LegacyImageDataset", ImageFixture)
    rows = pd.DataFrame({"row_uid": [f"row-{i:02d}" for i in range(31)],
                         "label_id_dense": np.arange(31) % 18})
    role = {"rows": rows, "geo": np.arange(62).reshape(31, 2),
            "label": rows.label_id_dense.to_numpy()}
    inputs = {"roles": {"train": role, "validation": role}, "source_config": {}}

    def collect(config, seed=2, epoch=1):
        loader, _ = joint.build_loaders(inputs, config, seed, epoch,
                                        train_transform=transform, prediction_transform=transform)
        result = []
        for batch in loader:
            result.extend(zip(batch["row_uid"], batch["image"], batch["label"].tolist()))
            # Different model/dropout work must not change later transformed rows.
            random.random()
            np.random.random(3)
            torch.rand(7)
        return ([uid for uid, _, _ in result],
                {uid: (tensor, label) for uid, tensor, label in result})

    return collect


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_declared_streams_pair_image_orders_and_transforms_across_packages(
        config, collect_image_loader, seed):
    config["training"].update(
        loader_rng_policy="declared_namespaces_v1",
        augmentation_seed_namespace="attention_fusion_joint_v1",
        shuffle_seed_namespace="attention_fusion_joint_v1",
    )
    first_order, first_rows = collect_image_loader(config, seed=seed)
    for regime, initialization, candidate in (
        ("joint_partial", "pretrained", "image_only"),
        ("joint_partial", "cs_adapted", "attention_two_token"),
        ("frozen_online", "pretrained", "attention_cls_skip_original"),
        ("frozen_online", "cs_adapted", "raw_concat"),
    ):
        package = deepcopy(config)
        package["study_id"] = f"{regime}_seed{seed}"
        package["authorized_scope"].update(regime=regime, initializations=[initialization],
                                            candidates=[candidate])
        order, rows = collect_image_loader(package, seed=seed)
        assert order == first_order
        assert len(order) == 31  # Include the smaller final batch.
        assert all(torch.equal(rows[uid][0], first_rows[uid][0])
                   and rows[uid][1] == first_rows[uid][1] for uid in order)

    for other_seed, other_epoch in ((1, 1), (seed, 2)):
        order, rows = collect_image_loader(config, seed=other_seed, epoch=other_epoch)
        assert order != first_order
        assert all(not torch.equal(rows[uid][0], first_rows[uid][0]) for uid in order)


def test_declared_shuffle_and_augmentation_namespaces_are_independent(config, collect_image_loader):
    config["training"].update(
        loader_rng_policy="declared_namespaces_v1",
        augmentation_seed_namespace="attention_fusion_joint_v1",
        shuffle_seed_namespace="attention_fusion_joint_v1",
    )
    first_order, first_rows = collect_image_loader(config)
    shuffle_changed = deepcopy(config)
    shuffle_changed["training"]["shuffle_seed_namespace"] = "different_shuffle"
    order, rows = collect_image_loader(shuffle_changed)
    assert order != first_order
    assert all(torch.equal(rows[uid][0], first_rows[uid][0]) for uid in order)
    augmentation_changed = deepcopy(config)
    augmentation_changed["training"]["augmentation_seed_namespace"] = "different_augmentation"
    order, rows = collect_image_loader(augmentation_changed)
    assert order == first_order
    assert all(not torch.equal(rows[uid][0], first_rows[uid][0]) for uid in order)


def test_legacy_image_streams_preserve_seed1_study_identity(config, collect_image_loader):
    first_order, first_rows = collect_image_loader(config, seed=1)
    explicit = deepcopy(config)
    explicit["training"].update(
        loader_rng_policy="legacy_study_id_v1",
        augmentation_seed_namespace="ignored_for_legacy_images",
        shuffle_seed_namespace="also_ignored_for_legacy_images",
    )
    order, rows = collect_image_loader(explicit, seed=1)
    assert order == first_order
    assert all(torch.equal(rows[uid][0], first_rows[uid][0]) for uid in order)
    explicit["study_id"] = "different_seed1_package"
    order, rows = collect_image_loader(explicit, seed=1)
    assert order != first_order
    assert all(not torch.equal(rows[uid][0], first_rows[uid][0]) for uid in order)


def test_unknown_loader_rng_policy_is_rejected(config, collect_image_loader):
    config["training"]["loader_rng_policy"] = "undeclared_policy"
    with pytest.raises(ValueError, match="Unknown M3J loader RNG policy"):
        collect_image_loader(config)
