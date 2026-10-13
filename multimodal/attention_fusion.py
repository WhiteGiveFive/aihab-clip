"""Frozen-vector attention fusion models and fitted geo preprocessing.

This module has no data-loading or encoder-training side effects. Model
construction uses isolated component seeds so it cannot alter data shuffling.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from multimodal.models import MLPHead


class AttentionFusionError(ValueError):
    """A model or preprocessing contract is inconsistent."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AttentionFusionError(f"{name} must be a positive integer")
    return value


def component_seed(study_id: str, seed: int, name: str) -> int:
    payload = f"{study_id}|{seed}|{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@contextmanager
def _component_rng(seed: int):
    # All parameters are constructed on CPU. Do not seed or disturb CUDA RNGs.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        nn.init.zeros_(module.in_proj_bias)
        nn.init.kaiming_uniform_(module.out_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(module.out_proj.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            bound = 1 / math.sqrt(module.in_features)
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    else:
        for child in module.children():
            _initialize(child)


def _component(factory: Callable[[], nn.Module], seed: int) -> nn.Module:
    with _component_rng(seed):
        module = factory()
        # Constructor implementation details must not affect the declared init.
        torch.random.default_generator.manual_seed(seed)
        _initialize(module)
    return module


def _learned_token(width: int, seed: int) -> nn.Parameter:
    with _component_rng(seed):
        value = torch.empty(1, 1, width)
        nn.init.normal_(value, std=0.02)
    return nn.Parameter(value)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class TokenwiseFeedForward(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(approximate="none"),
            nn.Dropout(dropout), nn.Linear(hidden, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class AttentionBlock(nn.Module):
    """One pre-LN attention/FFN block; the sequence axis is within each sample."""

    def __init__(self, width: int, heads: int, hidden: int, dropout: float, epsilon: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=epsilon)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True,
        )
        self.residual1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width, eps=epsilon)
        self.feedforward = TokenwiseFeedForward(width, hidden, dropout)
        self.residual2 = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        # Explicit weights computation uses the ordinary multi-key softmax path.
        updated, _ = self.attention(normalized, normalized, normalized, need_weights=True)
        tokens = tokens + self.residual1(updated)
        return tokens + self.residual2(self.feedforward(self.norm2(tokens)))


class TokenwiseCapacityBlock(nn.Module):
    """Two shared token-wise residual FFNs with no interaction between tokens."""

    def __init__(self, width: int, hidden: int, dropout: float, epsilon: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=epsilon)
        self.feedforward1 = TokenwiseFeedForward(width, hidden, dropout)
        self.residual1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width, eps=epsilon)
        self.feedforward2 = TokenwiseFeedForward(width, hidden, dropout)
        self.residual2 = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.residual1(self.feedforward1(self.norm1(tokens)))
        return tokens + self.residual2(self.feedforward2(self.norm2(tokens)))


class FusionClassifier(nn.Module):
    def __init__(self, config: Mapping[str, Any], candidate: Mapping[str, Any], seed: int):
        super().__init__()
        data, settings = config["data"], config["model"]
        self.candidate_id = str(candidate["id"])
        self.image_dim = _positive_int(data["image_dim"], "image_dim")
        self.geo_dim = _positive_int(data["geo_dim"], "geo_dim")
        classes = _positive_int(data["num_classes"], "num_classes")
        width = _positive_int(settings["token_dim"], "token_dim")
        hidden = _positive_int(settings["feedforward_dim"], "feedforward_dim")
        classifier_hidden = _positive_int(settings["classifier_hidden_dim"], "classifier_hidden_dim")
        dropout = float(settings["dropout"])
        epsilon = float(settings["layernorm_epsilon"])
        if not 0 <= dropout < 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise AttentionFusionError("invalid dropout or LayerNorm epsilon")
        seed_for = lambda name: component_seed(str(config["study_id"]), seed, name)
        self.fusion_layers = nn.ModuleList()
        self.image_adapter = self.geo_adapter = None
        self.register_parameter("image_type", None)
        self.register_parameter("geo_type", None)
        self.register_parameter("cls_token", None)

        if self.candidate_id in {"image_only", "geo_only", "raw_concat"}:
            input_width = {"image_only": self.image_dim, "geo_only": self.geo_dim,
                           "raw_concat": self.image_dim + self.geo_dim}[self.candidate_id]
            if candidate.get("architecture") != "mlp" or candidate.get("input_dim", input_width) != input_width:
                raise AttentionFusionError("baseline architecture or input_dim mismatch")
        else:
            adapter = lambda size: nn.Sequential(
                nn.Linear(size, width), nn.GELU(approximate="none"),
                nn.LayerNorm(width, eps=epsilon),
            )
            self.image_adapter = _component(lambda: adapter(self.image_dim), seed_for("image_adapter"))
            self.geo_adapter = _component(lambda: adapter(self.geo_dim), seed_for("geo_adapter"))
            self.image_type = _learned_token(width, seed_for("image_type"))
            self.geo_type = _learned_token(width, seed_for("geo_type"))
            input_width = 2 * width
            if self.candidate_id == "projected_concat":
                if candidate.get("architecture") != "adapters_then_concat":
                    raise AttentionFusionError("projected_concat architecture mismatch")
            elif self.candidate_id == "tokenwise_capacity":
                if candidate.get("architecture") != "two_preln_tokenwise_residual_ffns":
                    raise AttentionFusionError("capacity architecture mismatch")
                self.fusion_layers.append(_component(
                    lambda: TokenwiseCapacityBlock(width, hidden, dropout, epsilon),
                    seed_for("fusion_layer_0"),
                ))
            elif self.candidate_id in {"attention_two_token", "attention_cls"}:
                if candidate.get("architecture") != "self_attention":
                    raise AttentionFusionError("attention architecture mismatch")
                heads = _positive_int(settings["heads"], "heads")
                if width % heads:
                    raise AttentionFusionError("token_dim must be divisible by heads")
                layers = _positive_int(candidate["layers"], "layers")
                for index in range(layers):
                    self.fusion_layers.append(_component(
                        lambda: AttentionBlock(width, heads, hidden, dropout, epsilon),
                        seed_for(f"fusion_layer_{index}"),
                    ))
                if self.candidate_id == "attention_cls":
                    if candidate.get("cls_type_embedding", False):
                        raise AttentionFusionError("CLS has no additional type embedding")
                    self.cls_token = _learned_token(width, seed_for("cls_token"))
                    input_width = width
            else:
                raise AttentionFusionError(f"unsupported candidate: {self.candidate_id}")
        expected_readout = "updated_cls" if self.candidate_id == "attention_cls" else "updated_modality_concat"
        if "readout" in candidate and candidate["readout"] != expected_readout:
            raise AttentionFusionError("candidate readout does not match its architecture")
        self.classifier = _component(
            lambda: MLPHead(input_width, classifier_hidden, classes, dropout),
            seed_for("classifier"),
        )

    def _check_inputs(self, image: torch.Tensor, geo: torch.Tensor) -> None:
        if image.ndim != 2 or geo.ndim != 2 or image.shape[0] != geo.shape[0]:
            raise AttentionFusionError("inputs must have matching batch × feature shapes")
        if image.shape[1] != self.image_dim or geo.shape[1] != self.geo_dim:
            raise AttentionFusionError("input feature width does not match the model")

    def adapted_tokens(self, image: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        self._check_inputs(image, geo)
        if self.image_adapter is None:
            raise AttentionFusionError("this baseline has no modality adapters")
        image_token = self.image_adapter(image).unsqueeze(1) + self.image_type
        geo_token = self.geo_adapter(geo).unsqueeze(1) + self.geo_type
        return torch.cat((image_token, geo_token), dim=1)

    def updated_tokens(self, image: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        tokens = self.adapted_tokens(image, geo)
        if self.cls_token is not None:
            tokens = torch.cat((self.cls_token.expand(len(image), -1, -1), tokens), dim=1)
        for layer in self.fusion_layers:
            tokens = layer(tokens)
        return tokens

    def forward(self, image: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        self._check_inputs(image, geo)
        if self.candidate_id == "image_only":
            features = image
        elif self.candidate_id == "geo_only":
            features = geo
        elif self.candidate_id == "raw_concat":
            features = torch.cat((image, geo), dim=-1)
        else:
            tokens = self.updated_tokens(image, geo)
            features = tokens[:, 0] if self.cls_token is not None else tokens.flatten(start_dim=1)
        return self.classifier(features)


def build_model(config: dict, candidate_id: str, seed: int) -> nn.Module:
    """Construct one explicitly declared candidate without changing global RNGs."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AttentionFusionError("seed must be a nonnegative integer")
    settings = config["model"]
    contracts = {
        "adapter": "linear_gelu_layernorm", "adapter_dropout": 0.0,
        "modality_type_embeddings": True, "token_order": ["image", "geo"],
        "activation": "gelu_exact", "norm_first": True, "linear_bias": True,
        "final_encoder_norm": False, "classifier": "linear_relu_dropout_linear",
    }
    for key, expected in contracts.items():
        if settings.get(key) != expected:
            raise AttentionFusionError(f"unsupported model setting {key}: {settings.get(key)!r}")
    initialization = settings.get("initialization", {})
    initialization_contract = {
        "linear": "kaiming_uniform_a_sqrt5_bias_uniform_inverse_sqrt_fan_in",
        "qkv": "xavier_uniform_combined_qkv_bias_zero",
        "attention_output": "kaiming_uniform_a_sqrt5_bias_zero",
        "layernorm": "weight_one_bias_zero",
        "learned_tokens_and_types": "normal_std_0.02",
        "transformer_layers": "independently_initialized",
        "component_seed_template": "{study_id}|{seed}|{component_name}",
        "component_seed_hash": "sha256_first_4_bytes_unsigned_big_endian",
        "shared_shape_matched_components_same_seed": True,
    }
    for key, expected in initialization_contract.items():
        if initialization.get(key) != expected:
            raise AttentionFusionError(f"unsupported initialization setting {key}")
    candidates = [item for item in config["candidates"] if item["id"] == candidate_id]
    if len(candidates) != 1:
        raise AttentionFusionError(f"candidate must occur exactly once: {candidate_id}")
    model = FusionClassifier(config, candidates[0], seed)
    expected = candidates[0].get("expected_trainable_parameters")
    if expected is not None and count_parameters(model) != expected:
        raise AttentionFusionError(
            f"{candidate_id} has {count_parameters(model)} parameters, declared {expected}"
        )
    return model


class GeoPreprocessor:
    """Three geo treatments with explicit training-partition fitting identity."""

    def __init__(self, mode: str, feature_names: Sequence[str], l2_epsilon: float = 1e-12,
                 constant_epsilon: float = 1e-12):
        if mode not in {"native", "l2", "standardize"}:
            raise AttentionFusionError(f"unsupported geo mode: {mode}")
        self.mode = mode
        self.feature_names = list(feature_names)
        if (not self.feature_names or any(not isinstance(x, str) or not x for x in self.feature_names)
                or len(set(self.feature_names)) != len(self.feature_names)):
            raise AttentionFusionError("feature names must be unique nonempty strings")
        self.l2_epsilon, self.constant_epsilon = float(l2_epsilon), float(constant_epsilon)
        if any(not math.isfinite(x) or x <= 0 for x in (self.l2_epsilon, self.constant_epsilon)):
            raise AttentionFusionError("preprocessing epsilons must be finite and positive")
        self._state = None
        self.last_transform_zero_vector_count = 0

    def _matrix(self, values: Any) -> np.ndarray:
        try:
            matrix = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AttentionFusionError("geo values must be numeric") from exc
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise AttentionFusionError("geo matrix width does not match feature order")
        if not np.isfinite(matrix).all():
            raise AttentionFusionError("geo values must be finite")
        return matrix

    def fit(self, values: Any, row_uids: Sequence[str]) -> "GeoPreprocessor":
        matrix = self._matrix(values)
        rows = list(row_uids)
        if (not len(matrix) or len(rows) != len(matrix)
                or any(not isinstance(x, str) or not x for x in rows)
                or len(set(rows)) != len(rows)):
            raise AttentionFusionError("fitting identities must be unique, nonempty and row-aligned")
        state = {
            "schema_version": 1, "mode": self.mode, "feature_names": self.feature_names.copy(),
            "l2_epsilon": self.l2_epsilon, "constant_epsilon": self.constant_epsilon,
            "statistics_dtype": "float64", "output_dtype": "float32", "ddof": 0,
            "fit_row_count": len(rows),
            "fit_row_sha256": hashlib.sha256(json.dumps(
                rows, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
            "fit_zero_vector_count": int(np.count_nonzero(np.all(matrix == 0, axis=1))),
            "mean": None, "std": None, "scale": None, "constant_mask": None,
        }
        if self.mode == "standardize":
            mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=0)
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise AttentionFusionError("nonfinite fitted standardization statistics")
            constant = std <= self.constant_epsilon
            state.update(mean=mean.tolist(), std=std.tolist(),
                         scale=np.where(constant, 1.0, std).tolist(),
                         constant_mask=constant.tolist())
        self._state = state
        return self

    def transform(self, values: Any) -> np.ndarray:
        if self._state is None:
            raise AttentionFusionError("fit or restore preprocessing before transform")
        matrix = self._matrix(values)
        self.last_transform_zero_vector_count = int(np.count_nonzero(np.all(matrix == 0, axis=1)))
        if self.mode == "l2":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.isfinite(norms).all():
                raise AttentionFusionError("nonfinite geo vector norm")
            matrix = matrix / np.maximum(norms, self.l2_epsilon)
        elif self.mode == "standardize":
            matrix = (matrix - np.asarray(self._state["mean"])) / np.asarray(self._state["scale"])
        with np.errstate(over="ignore", invalid="ignore"):
            result = matrix.astype(np.float32)
        if not np.isfinite(result).all():
            raise AttentionFusionError("transformed geo values must be finite float32")
        return result

    def to_dict(self) -> dict[str, Any]:
        if self._state is None:
            raise AttentionFusionError("preprocessor has not been fitted")
        return json.loads(json.dumps(self._state, allow_nan=False))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "GeoPreprocessor":
        expected_fields = {
            "schema_version", "mode", "feature_names", "l2_epsilon", "constant_epsilon",
            "statistics_dtype", "output_dtype", "ddof", "fit_row_count", "fit_row_sha256",
            "fit_zero_vector_count", "mean", "std", "scale", "constant_mask",
        }
        if set(state) != expected_fields:
            raise AttentionFusionError("preprocessing state fields do not match the schema")
        if state.get("schema_version") != 1:
            raise AttentionFusionError("unsupported preprocessing state version")
        instance = cls(state["mode"], state["feature_names"], state["l2_epsilon"], state["constant_epsilon"])
        if (state.get("statistics_dtype") != "float64" or state.get("output_dtype") != "float32"
                or state.get("ddof") != 0):
            raise AttentionFusionError("preprocessing state dtype/variance contract mismatch")
        count = _positive_int(state.get("fit_row_count"), "fit_row_count")
        zero_count = state.get("fit_zero_vector_count")
        if isinstance(zero_count, bool) or not isinstance(zero_count, int) or not 0 <= zero_count <= count:
            raise AttentionFusionError("invalid fitted zero-vector count")
        digest = state.get("fit_row_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
            raise AttentionFusionError("invalid fitting-row hash")
        if instance.mode == "standardize":
            arrays = {key: np.asarray(state[key], dtype=np.float64) for key in ("mean", "std", "scale")}
            width = len(instance.feature_names)
            if any(a.shape != (width,) or not np.isfinite(a).all() for a in arrays.values()):
                raise AttentionFusionError("invalid standardization state shape or values")
            std = arrays["std"]
            mask = state.get("constant_mask")
            if not isinstance(mask, list) or len(mask) != width or any(type(x) is not bool for x in mask):
                raise AttentionFusionError("invalid constant-coordinate mask")
            expected_mask = std <= instance.constant_epsilon
            if ((std < 0).any() or not np.array_equal(mask, expected_mask)
                    or not np.array_equal(arrays["scale"], np.where(expected_mask, 1.0, std))):
                raise AttentionFusionError("inconsistent standardization scale or mask")
        elif any(state.get(key) is not None for key in ("mean", "std", "scale", "constant_mask")):
            raise AttentionFusionError("native/L2 preprocessing must not contain fitted coordinate statistics")
        instance._state = json.loads(json.dumps(dict(state), allow_nan=False))
        return instance
