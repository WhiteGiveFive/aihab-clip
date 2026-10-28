"""Two-layer CLS bypass models without changing the completed M3 model code."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
from torch import nn

from .attention_fusion import (
    AttentionFusionError, _component, build_model as build_original_model,
    component_seed, count_parameters,
)
from .models import MLPHead


READOUTS = {
    "attention_cls_skip_adapted": "updated_cls_plus_original_typed_tokens",
    "attention_cls_skip_original": "updated_cls_plus_original_embeddings",
}
INITIALIZATION_SEED_NAMESPACE = "attention_fusion_v1"


class BypassFusionClassifier(nn.Module):
    """Retain pre-attention inputs alongside the updated two-layer CLS token.

    ``geo`` is already transformed by the fitted, arm-specific preprocessor.
    The original-width bypass therefore receives exactly the same geo values
    as the adapter. Adapted bypass tokens include learned modality identities.
    """

    def __init__(self, config: Mapping[str, Any], candidate: Mapping[str, Any], seed: int):
        super().__init__()
        self.candidate_id = str(candidate["id"])
        namespace = config.get("initialization_seed_namespace")
        if namespace != INITIALIZATION_SEED_NAMESPACE:
            raise AttentionFusionError("bypass initialization_seed_namespace must preserve attention_fusion_v1")
        if (candidate.get("architecture") != "self_attention"
                or candidate.get("layers") != 2
                or candidate.get("cls_type_embedding", False)
                or candidate.get("readout") != READOUTS.get(self.candidate_id)):
            raise AttentionFusionError("bypass candidate must declare its two-layer CLS architecture and exact readout")
        if self.candidate_id not in READOUTS:
            raise AttentionFusionError(f"unsupported bypass candidate: {self.candidate_id}")

        # Validate and construct through the unchanged original model factory.
        # Artifact identity remains the new study; only component seeds inherit
        # the original namespace, allowing authenticated same-recipe controls.
        backbone_config = deepcopy(dict(config))
        backbone_config["study_id"] = namespace
        backbone_config["candidates"] = [{
            "id": "attention_cls", "geo": True, "architecture": "self_attention",
            "layers": 2, "readout": "updated_cls", "cls_type_embedding": False,
        }]
        self.backbone = build_original_model(backbone_config, "attention_cls", seed)
        self.image_dim, self.geo_dim = self.backbone.image_dim, self.backbone.geo_dim
        self.token_dim = int(config["model"]["token_dim"])
        input_width = (3 * self.token_dim if self.candidate_id == "attention_cls_skip_adapted"
                       else self.token_dim + self.image_dim + self.geo_dim)
        self.classifier_input_dim = input_width
        self.backbone.classifier = _component(
            lambda: MLPHead(input_width, config["model"]["classifier_hidden_dim"],
                            config["data"]["num_classes"], config["model"]["dropout"]),
            component_seed(namespace, seed, "classifier"),
        )

    @property
    def classifier(self) -> nn.Module:
        return self.backbone.classifier

    def readout_features(self, image: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        original_tokens = self.backbone.adapted_tokens(image, geo)
        tokens = torch.cat((self.backbone.cls_token.expand(len(image), -1, -1), original_tokens), dim=1)
        for layer in self.backbone.fusion_layers:
            tokens = layer(tokens)
        bypass = (original_tokens.flatten(start_dim=1)
                  if self.candidate_id == "attention_cls_skip_adapted"
                  else torch.cat((image, geo), dim=-1))
        return torch.cat((tokens[:, 0], bypass), dim=-1)

    def forward(self, image: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.readout_features(image, geo))


def build_model(config: dict[str, Any], candidate_id: str, seed: int) -> BypassFusionClassifier:
    """Build a declared bypass candidate with the original component seed scope."""
    if candidate_id not in READOUTS:
        raise AttentionFusionError(f"unsupported bypass candidate: {candidate_id}")
    candidates = [item for item in config["candidates"] if item["id"] == candidate_id]
    if len(candidates) != 1:
        raise AttentionFusionError(f"candidate must occur exactly once: {candidate_id}")
    model = BypassFusionClassifier(config, candidates[0], seed)
    expected = candidates[0].get("expected_trainable_parameters")
    if expected is not None and count_parameters(model) != expected:
        raise AttentionFusionError(
            f"{candidate_id} has {count_parameters(model)} parameters, declared {expected}"
        )
    return model
