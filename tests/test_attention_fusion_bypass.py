"""Bypass placement, differentiability and matched CLS initialization."""

from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from multimodal.attention_fusion import AttentionFusionError, build_model as build_original_model, count_parameters
from multimodal.attention_fusion_bypass import READOUTS, build_model


ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"attention_cls_skip_adapted": 524946, "attention_cls_skip_original": 770706}


def extension_config():
    config = yaml.safe_load((ROOT / "configs/multimodal_attention_fusion.yaml").read_text())
    config["study_id"] = "attention_fusion_cls_bypass_v1"
    config["initialization_seed_namespace"] = "attention_fusion_v1"
    config["candidates"] = [
        {"id": candidate, "geo": True, "architecture": "self_attention", "layers": 2,
         "readout": readout, "cls_type_embedding": False,
         "expected_trainable_parameters": COUNTS[candidate]}
        for candidate, readout in READOUTS.items()
    ]
    return config


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs():
    generator = torch.Generator().manual_seed(9301)
    return (torch.randn(4, 1152, generator=generator), torch.randn(4, 64, generator=generator))


@pytest.mark.parametrize("candidate", READOUTS)
def test_exact_readout_preserves_pre_attention_features(candidate):
    config = extension_config()
    model = build_model(config, candidate, 1).eval()
    image, geo = inputs()
    with torch.no_grad():
        original_tokens = model.backbone.adapted_tokens(image, geo)
        updated = model.backbone.updated_tokens(image, geo)
        features = model.readout_features(image, geo)
        torch.testing.assert_close(features[:, :128], updated[:, 0], atol=0, rtol=0)
        expected = (original_tokens.flatten(start_dim=1) if candidate.endswith("adapted")
                    else torch.cat((image, geo), dim=-1))
        torch.testing.assert_close(features[:, 128:], expected, atol=0, rtol=0)
        assert not torch.equal(original_tokens, updated[:, 1:])
        if candidate.endswith("adapted"):
            untyped = torch.cat((model.backbone.image_adapter(image), model.backbone.geo_adapter(geo)), dim=-1)
            assert not torch.equal(features[:, 128:], untyped)
        assert model(image, geo).shape == (4, 18)
    assert model.classifier_input_dim == (384 if candidate.endswith("adapted") else 1344)
    assert model.classifier.net[0].in_features == model.classifier_input_dim
    assert isinstance(model.classifier.net[1], nn.ReLU)
    assert count_parameters(model) == COUNTS[candidate]


@pytest.mark.parametrize("candidate", READOUTS)
def test_complete_model_has_finite_gradients_in_both_modalities_and_cls_backbone(candidate):
    model = build_model(extension_config(), candidate, 1).eval()
    image, geo = (value.requires_grad_() for value in inputs())
    nn.functional.cross_entropy(model(image, geo), torch.tensor([0, 1, 2, 3])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name
    for value in (image, geo):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad)


def test_adapted_bypass_gradient_reaches_adapters_without_traversing_attention():
    model = build_model(extension_config(), "attention_cls_skip_adapted", 1).eval()
    image, geo = inputs()
    # A nonuniform linear probe avoids LayerNorm's constant sum/squared-norm.
    tail = model.readout_features(image, geo)[:, 128:]
    (tail * torch.arange(1, 257)).sum().backward()
    for component in (model.backbone.image_adapter, model.backbone.geo_adapter):
        assert all(p.grad is not None and torch.count_nonzero(p.grad) for p in component.parameters())
    assert torch.count_nonzero(model.backbone.image_type.grad)
    assert torch.count_nonzero(model.backbone.geo_type.grad)
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in model.backbone.fusion_layers.parameters())


def test_original_bypass_is_direct_and_does_not_reproject_or_renormalize_inputs():
    model = build_model(extension_config(), "attention_cls_skip_original", 1).eval()
    image, geo = (value.requires_grad_() for value in inputs())
    features = model.readout_features(image, geo)
    features[:, 128:].sum().backward()
    torch.testing.assert_close(image.grad, torch.ones_like(image), atol=0, rtol=0)
    torch.testing.assert_close(geo.grad, torch.ones_like(geo), atol=0, rtol=0)
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in model.backbone.image_adapter.parameters())


def test_original_shared_initialization_and_global_rng_are_preserved():
    original_config = yaml.safe_load((ROOT / "configs/multimodal_attention_fusion.yaml").read_text())
    original = build_original_model(original_config, "attention_cls", 2)
    before = torch.random.get_rng_state().clone()
    for candidate in READOUTS:
        config = extension_config()
        model = build_model(config, candidate, 2)
        assert torch.equal(before, torch.random.get_rng_state())
        for name, value in original.state_dict().items():
            if not name.startswith("classifier."):
                assert torch.equal(value, model.backbone.state_dict()[name]), name
        # Artifact naming and LR selection cannot change shared initialization.
        config["study_id"] = "a_different_artifact_namespace"
        config["training"]["learning_rates"] = [0.02]
        other = build_model(config, candidate, 2)
        assert all(torch.equal(value, other.state_dict()[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("candidate", READOUTS)
def test_evaluation_is_sample_isolated_and_state_dict_replays(candidate):
    model = build_model(extension_config(), candidate, 2).eval()
    image, geo = inputs()
    with torch.no_grad():
        together = model(image, geo)
        separate = torch.cat([model(image[i:i+1], geo[i:i+1]) for i in range(4)])
        torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
        permutation = torch.tensor([3, 1, 0, 2])
        torch.testing.assert_close(model(image[permutation], geo[permutation]), together[permutation], atol=1e-6, rtol=1e-5)
        restored = build_model(extension_config(), candidate, 3).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(image, geo), together, atol=0, rtol=0)


@pytest.mark.parametrize("mutation", ["namespace", "readout", "depth", "types", "count", "activation"])
def test_contract_mismatches_fail_closed(mutation):
    config = extension_config()
    candidate = config["candidates"][0]
    if mutation == "namespace":
        config["initialization_seed_namespace"] = "new_study"
    elif mutation == "readout":
        candidate["readout"] = "updated_cls"
    elif mutation == "depth":
        candidate["layers"] = 1
    elif mutation == "types":
        candidate["cls_type_embedding"] = True
    elif mutation == "count":
        candidate["expected_trainable_parameters"] += 1
    else:
        config["model"]["activation"] = "relu"
    with pytest.raises(AttentionFusionError):
        build_model(config, candidate["id"], 1)
