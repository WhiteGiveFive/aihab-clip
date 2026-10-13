from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from multimodal.attention_fusion import (
    AttentionFusionError, GeoPreprocessor, build_model, count_parameters,
)
from multimodal.models import MLPHead


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "image_only", "geo_only", "raw_concat", "projected_concat",
    "tokenwise_capacity", "attention_two_token", "attention_cls",
)
EXPECTED_COUNTS = (299794, 21266, 316178, 227090, 359442, 359570, 459410)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "configs/multimodal_attention_fusion.yaml").read_text())


def inputs(image_dim=1152, geo_dim=64, batch=4):
    generator = torch.Generator().manual_seed(938)
    return (torch.randn(batch, image_dim, generator=generator),
            torch.randn(batch, geo_dim, generator=generator))


@pytest.mark.parametrize("candidate,expected", zip(CANDIDATES, EXPECTED_COUNTS))
def test_counts_relu_head_and_finite_modal_gradients(config, candidate, expected):
    model = build_model(config, candidate, seed=1).eval()
    assert count_parameters(model) == expected
    assert isinstance(model.classifier, MLPHead)
    assert isinstance(model.classifier.net[1], nn.ReLU)
    image, geo = (value.requires_grad_() for value in inputs())
    logits = model(image, geo)
    assert logits.shape == (4, 18)
    assert torch.isfinite(logits).all()
    nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for tensor, used in ((image, candidate != "geo_only"), (geo, candidate != "image_only")):
        if used:
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
            assert torch.count_nonzero(tensor.grad) > 0
        else:
            assert tensor.grad is None


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_evaluation_is_batch_independent_and_checkpoint_replays(config, candidate):
    model = build_model(config, candidate, seed=2).eval()
    image, geo = inputs()
    with torch.no_grad():
        together = model(image, geo)
        separately = torch.cat([model(image[i:i+1], geo[i:i+1]) for i in range(len(image))])
        torch.testing.assert_close(together, separately, atol=1e-6, rtol=1e-5)
        perm = torch.tensor([3, 1, 0, 2])
        torch.testing.assert_close(model(image[perm], geo[perm]), together[perm], atol=1e-6, rtol=1e-5)
        restored = build_model(config, candidate, seed=3).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(image, geo), together, atol=0, rtol=0)


def test_token_axis_and_capacity_control_do_not_mix_samples_or_tokens(config):
    image, geo = inputs()
    capacity = build_model(config, "tokenwise_capacity", 1).eval()
    attention = build_model(config, "attention_two_token", 1).eval()
    replacement = torch.roll(geo, 1, dims=0)
    with torch.no_grad():
        initial = capacity.adapted_tokens(image, geo)
        updated = capacity.updated_tokens(image, geo)
        changed = capacity.updated_tokens(image, replacement)
        assert initial.shape == updated.shape == (4, 2, 128)
        torch.testing.assert_close(updated[:, 0], changed[:, 0], atol=0, rtol=0)
        assert not torch.equal(updated[:, 1], changed[:, 1])
        attended = attention.updated_tokens(image, geo)
        attended_changed = attention.updated_tokens(image, replacement)
        assert not torch.allclose(attended[:, 0], attended_changed[:, 0])


def test_shared_components_separate_cls_layers_and_rng_isolation(config):
    before = torch.random.get_rng_state().clone()
    models = [build_model(config, candidate, 1) for candidate in CANDIDATES[3:]]
    assert torch.equal(before, torch.random.get_rng_state())
    for attribute in ("image_adapter", "geo_adapter"):
        expected = getattr(models[0], attribute).state_dict()
        for model in models[1:]:
            for key, value in getattr(model, attribute).state_dict().items():
                assert torch.equal(value, expected[key])
    for attribute in ("image_type", "geo_type"):
        assert all(torch.equal(getattr(models[0], attribute), getattr(m, attribute)) for m in models[1:])
    for model in models[1:3]:
        for key, value in model.classifier.state_dict().items():
            assert torch.equal(value, models[0].classifier.state_dict()[key])
    cls = models[-1]
    assert len(cls.fusion_layers) == 2
    first, second = (layer.attention.in_proj_weight for layer in cls.fusion_layers)
    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first, second)
    assert cls.updated_tokens(*inputs()).shape == (4, 3, 128)
    changed = deepcopy(config)
    changed["training"]["learning_rates"] = [0.02]
    other = build_model(changed, "attention_two_token", 1)
    for key, value in other.state_dict().items():
        assert torch.equal(value, models[2].state_dict()[key])
    changed_seed = build_model(config, "attention_two_token", 2)
    assert not torch.equal(changed_seed.image_adapter[0].weight, models[2].image_adapter[0].weight)


def test_initialization_contract(config):
    model = build_model(config, "attention_two_token", 1)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, nn.GELU):
            assert module.approximate == "none"
    attention = model.fusion_layers[0].attention
    assert attention.batch_first
    assert torch.count_nonzero(attention.in_proj_bias) == 0
    assert torch.count_nonzero(attention.out_proj.bias) == 0


def test_widths_and_depth_are_configuration_driven(config):
    config["study_id"] = "attention_fusion_small_new_study"
    config["data"].update(image_dim=12, geo_dim=5, num_classes=3)
    config["model"].update(token_dim=8, feedforward_dim=13, classifier_hidden_dim=11, heads=2)
    for candidate in config["candidates"]:
        candidate.pop("expected_trainable_parameters")
        if "input_dim" in candidate:
            candidate["input_dim"] = {"image_only": 12, "geo_only": 5, "raw_concat": 17}[candidate["id"]]
        if candidate["id"] == "attention_two_token":
            candidate["layers"] = 3
    for name in CANDIDATES:
        model = build_model(config, name, 7)
        assert model(*inputs(12, 5)).shape == (4, 3)
    assert len(build_model(config, "attention_two_token", 7).fusion_layers) == 3


def test_rejects_count_and_architecture_mismatches(config):
    config["candidates"][0]["expected_trainable_parameters"] += 1
    with pytest.raises(AttentionFusionError, match="parameters"):
        build_model(config, "image_only", 1)
    config["model"]["heads"] = 3
    with pytest.raises(AttentionFusionError, match="divisible"):
        build_model(config, "attention_two_token", 1)
    config["model"]["classifier"] = "linear_gelu_dropout_linear"
    with pytest.raises(AttentionFusionError, match="classifier"):
        build_model(config, "projected_concat", 1)


def test_standardization_is_training_local_and_replayable():
    train = np.array([[1, 5, 3], [3, 5, 7]], dtype=np.float32)
    transform = GeoPreprocessor("standardize", ["a", "b", "c"]).fit(train, ["r1", "r2"])
    state = transform.to_dict()
    assert state["mean"] == [2, 5, 5]
    assert state["std"] == [1, 0, 2]
    assert state["scale"] == [1, 1, 2]
    assert state["constant_mask"] == [False, True, False]
    validation = np.array([[5, 8, 9]], dtype=np.float64)
    expected = np.array([[3, 3, 2]], dtype=np.float32)
    np.testing.assert_array_equal(transform.transform(validation), expected)
    assert transform.to_dict() == state
    restored = GeoPreprocessor.from_dict(state)
    np.testing.assert_array_equal(restored.transform(validation), expected)
    assert restored.transform(validation).dtype == np.float32
    reordered = GeoPreprocessor("standardize", ["a", "b", "c"]).fit(train[::-1], ["r2", "r1"])
    assert reordered.to_dict()["fit_row_sha256"] != state["fit_row_sha256"]
    state["mean"][0] = -999
    assert transform.to_dict()["mean"][0] == 2


def test_native_l2_and_zero_vectors_remain_distinct():
    values = np.array([[3, 4], [0, 0], [1e-14, 0]])
    native = GeoPreprocessor("native", ["a", "b"]).fit(values, ["a", "b", "c"])
    l2 = GeoPreprocessor("l2", ["a", "b"]).fit(values, ["a", "b", "c"])
    np.testing.assert_array_equal(native.transform(values), values.astype(np.float32))
    np.testing.assert_allclose(l2.transform(values), [[0.6, 0.8], [0, 0], [0.01, 0]], atol=1e-8)
    assert l2.last_transform_zero_vector_count == l2.to_dict()["fit_zero_vector_count"] == 1
    np.testing.assert_array_equal(GeoPreprocessor.from_dict(l2.to_dict()).transform(values), l2.transform(values))


def test_preprocessing_rejects_bad_inputs_and_corrupted_state():
    transform = GeoPreprocessor("standardize", ["a", "b"])
    with pytest.raises(AttentionFusionError, match="fit or restore"):
        transform.transform([[1, 2]])
    with pytest.raises(AttentionFusionError, match="finite"):
        transform.fit([[np.nan, 2]], ["r1"])
    with pytest.raises(AttentionFusionError, match="unique"):
        transform.fit([[1, 2], [3, 4]], ["r1", "r1"])
    transform.fit([[1, 2], [3, 4]], ["r1", "r2"])
    with pytest.raises(AttentionFusionError, match="width"):
        transform.transform([[1]])
    state = transform.to_dict()
    state["scale"][0] = 0
    with pytest.raises(AttentionFusionError, match="scale"):
        GeoPreprocessor.from_dict(state)
