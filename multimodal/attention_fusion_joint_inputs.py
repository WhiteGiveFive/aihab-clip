"""Audited online inputs and visual wrappers for the M3J development study.

The historical cached-feature models and their preparation code are unchanged.
Only development rows are exposed here; per-row augmentation has a separate RNG
stream so candidate order, dropout and encoder initialization cannot alter views.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from .attention_fusion import GeoPreprocessor, build_model as build_fusion_model, component_seed
from .attention_fusion_bypass import READOUTS, build_model as build_bypass_model
from .attention_fusion_joint_protocol import realized_loader_namespaces
from . import attention_fusion_inputs as audited_inputs
from .geo_helpfulness_protocol import canonical_sha256, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTIAL_BLOCKS = tuple(range(17, 27))
EXPECTED_VISUAL_PARAMETERS = 428256704
EXPECTED_PARTIAL_PARAMETERS = 167635696
EXPECTED_PARTIAL_TENSORS = 135


class JointInputError(ValueError):
    """An online input or visual training contract was violated."""


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise JointInputError(message)


def _path(root: Path, value: str | Path) -> Path:
    value = Path(value)
    _require(not value.is_absolute() and ".." not in value.parts,
             "Study paths must be repository relative without traversal")
    result = (root / value).resolve()
    _require(result.is_relative_to(root), "Study path escapes repository")
    return result


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    _require(isinstance(value, dict), f"Expected a YAML mapping: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_inputs(config: Mapping[str, Any], root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Reauthenticate development inputs and persist one shared train-fitted scaler.

    Image packages authenticate their raw pixels; geo-only skips unused pixel
    and encoder content. No cached image embeddings or test features/labels are
    loaded. Historical test-identity-only overlap checks remain in the audit.
    """
    root = Path(root).resolve()
    base = _yaml(_path(root, config["base_config"]))
    seeds = list(config["authorized_scope"]["seeds"])
    geo_only = config["authorized_scope"].get("regime") == "geo_only"
    if geo_only:
        context = audited_inputs.audit_source_inputs(
            base, seeds, repo_root=root, authenticate_image_producers=False)
    else:
        context = audited_inputs.audit_source_inputs(base, seeds, repo_root=root)
    inventory = [] if geo_only else audited_inputs._image_inventory(context)
    fingerprint = (None if geo_only else
                   audited_inputs._validate_historical_image_fingerprint(context, inventory))
    source_config = deepcopy(context["source_config"])
    declared_source = _yaml(_path(root, config["source_config"]))
    _require(canonical_sha256(declared_source) == canonical_sha256(source_config),
             "Historical source configuration changed since protocol freeze")
    train = context["roles"]["train"]
    scaler = GeoPreprocessor(
        "standardize", base["data"]["geo_columns"],
        constant_epsilon=base["geo"]["standardize_constant_epsilon"],
    ).fit(train["geo"], train["rows"].row_uid.astype(str).tolist())
    roles = {}
    for role in ("train", "validation"):
        rows = context["roles"][role]["rows"].copy().reset_index(drop=True)
        labels = rows.label_id_dense.to_numpy(dtype=np.int64, copy=True)
        _require(len(rows) == base["data"][f"{role}_rows"], "Development row count changed")
        _require(rows.row_uid.is_unique and rows.row_uid.tolist() == sorted(rows.row_uid.tolist()),
                 "Development rows must have unique, sorted identities")
        _require(((labels >= 0) & (labels < base["data"]["num_classes"])).all(),
                 "Development label outside fixed ontology")
        roles[role] = {
            "rows": rows, "geo": scaler.transform(context["roles"][role]["geo"]),
            "label": labels, "row_uid": rows.row_uid.astype(str).to_numpy(),
            "plot_idx": rows.plot_idx.astype(str).to_numpy(),
        }
    _require(not (set(roles["train"]["row_uid"]) & set(roles["validation"]["row_uid"])),
             "Training and validation rows overlap")
    _require(not (set(roles["train"]["plot_idx"]) & set(roles["validation"]["plot_idx"])),
             "Training and validation plots overlap")
    provenance = {
        "schema": "attention_fusion_joint.inputs.v1", "study_id": config["study_id"],
        "source_role": "train", "validation_role": "validation",
        "test_features_or_labels_loaded": False, "geo_mode": "standardize",
        "base_config_sha256": canonical_sha256(base),
        "source_config_sha256": canonical_sha256(source_config),
        "m1_manifest_sha256": context["m1_binding"]["manifest_file_sha256"],
        "assignments_sha256": context["audit"]["protocol_files"]["development_assignments.parquet"]["sha256"],
        "geo_source_files": context["audit"]["development_geo_source_files"],
        "scaler_sha256": canonical_sha256(scaler.to_dict()),
        "role_rows": {role: len(item["rows"]) for role, item in roles.items()},
        "role_row_sha256": {role: canonical_sha256(item["row_uid"].tolist())
                            for role, item in roles.items()},
    }
    if not geo_only:
        provenance.update(image_fingerprint=fingerprint,
                          image_inventory_sha256=canonical_sha256(inventory))
    destination = _path(root, config["paths"]["artifact_root"]) / "inputs"
    manifest_path = destination / "manifest.json"
    if destination.exists():
        _require(manifest_path.is_file(), "Incomplete existing joint input directory")
        manifest = json.loads(manifest_path.read_text())
        _require(manifest["provenance"] == provenance, "Saved joint input provenance changed")
        for name, digest in manifest["file_sha256"].items():
            _require(Path(name).name == name, "Invalid prepared artifact filename")
            _require(sha256_file(destination / name) == digest, f"Saved input hash mismatch: {name}")
        _require(json.loads((destination / "geo_standardization.json").read_text()) == scaler.to_dict(),
                 "Saved training-only scaler changed")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".inputs-", dir=destination.parent))
        try:
            _atomic_json(staging / "geo_standardization.json", scaler.to_dict())
            if not geo_only:
                _atomic_json(staging / "image_inventory.json", {"rows": inventory})
            for role, item in roles.items():
                item["rows"].to_parquet(staging / f"{role}_rows.parquet", index=False)
                np.savez_compressed(staging / f"{role}.npz", **{
                    key: item[key] for key in ("geo", "label", "row_uid", "plot_idx")
                })
            manifest = {"provenance": provenance,
                        "file_sha256": {p.name: sha256_file(p) for p in sorted(staging.iterdir())}}
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            _atomic_json(staging / "manifest.json", manifest)
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    _require(canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
             == manifest["manifest_sha256"], "Prepared manifest self-hash mismatch")
    return {
        "roles": roles, "scaler": scaler.to_dict(), "provenance": provenance,
        "manifest": manifest, "manifest_path": str(manifest_path),
        "source_config": source_config, "base_config": base, "repo": root,
        "encoder_producers": context["audit"]["encoders"],
    }


@contextmanager
def isolated_augmentation_rng(seed: int):
    """Use CPU augmentation RNGs without consuming the model/dropout stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def augmentation_seed(study_id: str, seed: int, epoch: int, row_uid: str) -> int:
    return component_seed(study_id, seed, f"augmentation|epoch={epoch}|row={row_uid}")


class JointImageGeoDataset(Dataset):
    """Canonical aligned rows with image-specific, replayable training views."""

    def __init__(self, role_inputs: Mapping[str, Any], *, source_config: Mapping[str, Any],
                 transform: Any, study_id: str, seed: int, epoch: int,
                 training: bool, include_images: bool = True,
                 augmentation_namespace: str | None = None):
        from .geo_helpfulness_oof import _LegacyImageDataset

        self.rows = role_inputs["rows"].reset_index(drop=True)
        self.geo = np.asarray(role_inputs["geo"], dtype=np.float32)
        self.labels = np.asarray(role_inputs["label"], dtype=np.int64)
        self.row_uids = self.rows.row_uid.astype(str).tolist()
        self.plot_ids = (self.rows.plot_idx.astype(str).tolist() if "plot_idx" in self.rows
                         else [""] * len(self.rows))
        self.study_id, self.seed, self.epoch = study_id, seed, epoch
        self.augmentation_namespace = (study_id if augmentation_namespace is None
                                       else augmentation_namespace)
        self.training = training
        _require(len(self.rows) == len(self.geo) == len(self.labels), "Input rows are not aligned")
        self.images = (_LegacyImageDataset(self.rows, config=source_config, transform=transform,
                                          include_labels=True) if include_images else None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.images is None:
            image = torch.zeros(0, dtype=torch.float32)
        elif self.training:
            local_seed = augmentation_seed(
                self.augmentation_namespace, self.seed, self.epoch, self.row_uids[index])
            with isolated_augmentation_rng(local_seed):
                image, label = self.images[index]
            _require(label == self.labels[index], "Image/geo labels diverged")
        else:
            image, label = self.images[index]
            _require(label == self.labels[index], "Image/geo labels diverged")
        return {"image": image, "geo": torch.from_numpy(self.geo[index].copy()),
                "label": int(self.labels[index]), "row_uid": self.row_uids[index],
                "plot_idx": self.plot_ids[index], "index": index}


def build_loaders(inputs: Mapping[str, Any], config: Mapping[str, Any], seed: int, epoch: int,
                  *, train_transform: Any, prediction_transform: Any, include_images: bool = True,
                  num_workers: int | None = None) -> tuple[DataLoader, DataLoader]:
    """Return paired loaders; view/shuffle streams do not depend on model order."""
    _require(seed in config["training_seeds"], "Undeclared seed")
    _require(isinstance(epoch, int) and epoch >= 1, "Epoch is one-based")
    settings = config["training"]
    namespaces = realized_loader_namespaces(config, include_images=include_images)
    workers = settings["num_workers"] if num_workers is None else num_workers
    _require(workers == 0, "The declared historical input recipe uses zero workers")
    loaders = []
    for role, transform in (("train", train_transform), ("validation", prediction_transform)):
        dataset = JointImageGeoDataset(
            inputs["roles"][role], source_config=inputs["source_config"], transform=transform,
            study_id=config["study_id"], seed=seed, epoch=epoch, training=role == "train",
            include_images=include_images,
            augmentation_namespace=namespaces["augmentation_seed_namespace"],
        )
        # Later-seed packages explicitly share the declared streams across
        # regimes; the default resolver preserves completed seed-1 streams.
        generator = torch.Generator(device="cpu").manual_seed(component_seed(
            namespaces["shuffle_seed_namespace"], seed, f"loader|{role}|epoch={epoch}",
        ))
        loaders.append(DataLoader(dataset, batch_size=settings["batch_size"], shuffle=role == "train",
                                  drop_last=False, num_workers=workers, generator=generator))
    return tuple(loaders)


def _partial_name(name: str) -> bool:
    return (name.startswith(("trunk.norm.", "trunk.attn_pool."))
            or any(name.startswith(f"trunk.blocks.{index}.") for index in PARTIAL_BLOCKS))


def configure_visual_scope(visual: nn.Module, regime: str, *, strict: bool = True) -> dict[str, Any]:
    """Select by explicit names, then verify the pinned model's exact scope."""
    _require(regime in {"joint_partial", "frozen_online"}, "Unknown image training regime")
    _require(hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"), "Missing visual trunk")
    _require(len(visual.trunk.blocks) == 27, "Expected the pinned 27 visual blocks")
    _require(hasattr(visual.trunk, "norm") and hasattr(visual.trunk, "attn_pool"),
             "Missing visual final normalization or attention pooling")
    names = []
    for name, parameter in visual.named_parameters():
        trainable = regime == "joint_partial" and _partial_name(name)
        parameter.requires_grad_(trainable)
        if trainable:
            names.append({"name": name, "shape": list(parameter.shape), "elements": parameter.numel()})
    total = sum(parameter.numel() for parameter in visual.parameters())
    count = sum(item["elements"] for item in names)
    if strict:
        _require(total == EXPECTED_VISUAL_PARAMETERS, "Pinned visual parameter count changed")
        if regime == "joint_partial":
            _require(count == EXPECTED_PARTIAL_PARAMETERS and len(names) == EXPECTED_PARTIAL_TENSORS,
                     "Partial visual parameter count/tensor count differs from historical scope")
    _require(regime != "joint_partial" or count > 0, "Partial scope selected no parameters")
    return {"regime": regime, "partial_block_indices": list(PARTIAL_BLOCKS),
            "visual_parameters": total, "trainable_visual_parameters": count,
            "trainable_visual_tensors": len(names), "trainable_parameters": names,
            "trainable_manifest_sha256": canonical_sha256(names)}


class OnlineFusionModel(nn.Module):
    """Differentiable image-to-global-vector path plus unchanged fusion models."""

    def __init__(self, visual: nn.Module | None, fusion: nn.Module, *, regime: str,
                 train_transform: Any = None, prediction_transform: Any = None,
                 strict_scope: bool = True, provenance: Mapping[str, Any] | None = None):
        super().__init__()
        self.visual, self.fusion, self.regime = visual, fusion, regime
        self.candidate_id = fusion.candidate_id
        self.train_transform, self.prediction_transform = train_transform, prediction_transform
        self.provenance = dict(provenance or {})
        self.trainable_manifest = (configure_visual_scope(visual, regime, strict=strict_scope)
                                   if visual is not None else {
                                       "regime": "geo_only", "visual_parameters": 0,
                                       "trainable_visual_parameters": 0, "trainable_visual_tensors": 0,
                                       "trainable_parameters": [],
                                       "trainable_manifest_sha256": canonical_sha256([])})
        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.visual is not None:
            self.visual.eval()
            if mode and self.regime == "joint_partial":
                for index in PARTIAL_BLOCKS:
                    self.visual.trunk.blocks[index].train(True)
                self.visual.trunk.norm.train(True)
                self.visual.trunk.attn_pool.train(True)
        return self

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        _require(self.visual is not None, "Geo-only has no image encoder")
        if self.regime == "frozen_online":
            with torch.no_grad():
                encoded = self.visual(image)
        else:
            encoded = self.visual(image)
        _require(encoded.ndim == 2 and encoded.shape[1] == self.fusion.image_dim,
                 "Image encoder did not return the declared global vector")
        _require(torch.isfinite(encoded).all().item() and (encoded.norm(dim=-1) > 0).all().item(),
                 "Image encoder emitted zero or nonfinite features")
        # Keep the historical adaptation's operation and autocast dispatch;
        # no detach or NumPy/cache conversion may interrupt joint gradients.
        return F.normalize(encoded, dim=-1)

    def forward(self, image: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        if self.visual is None:
            image = geo.new_zeros((len(geo), self.fusion.image_dim))
        else:
            image = self.encode_image(image)
        return self.fusion(image, geo)


def make_model(config: Mapping[str, Any], candidate: str, initialization: str, seed: int,
               regime: str, device: str | torch.device, *, root: str | Path = REPO_ROOT,
               visual: nn.Module | None = None, transforms: tuple[Any, Any] | None = None,
               strict_scope: bool = True) -> OnlineFusionModel:
    """Load only pinned local weights; optionally inject a tiny visual for tests."""
    root, device = Path(root).resolve(), torch.device(device)
    if candidate == "geo_only":
        _require(initialization == "not_applicable" and regime == "geo_only",
                 "Geo-only requires a single initialization-independent geo_only fit")
    else:
        _require(initialization in {"pretrained", "cs_adapted"}, "Unknown image initialization")
    _require(seed in config["training_seeds"], "Undeclared training seed")
    base = _yaml(_path(root, config["base_config"]))
    model_config = (_yaml(_path(root, config["bypass_config"])) if candidate in READOUTS else deepcopy(base))
    model_config["study_id"] = config["initialization_seed_namespace"]
    model_config["initialization_seed_namespace"] = config["initialization_seed_namespace"]
    fusion = (build_bypass_model if candidate in READOUTS else build_fusion_model)(model_config, candidate, seed)
    provenance = {"initialization": initialization, "seed": seed, "candidate": candidate, "regime": regime,
                  "initialization_seed_namespace": config["initialization_seed_namespace"],
                  "initialization_is_injected_test_fixture": visual is not None,
                  "fusion_model_config": {key: deepcopy(model_config[key]) for key in (
                      "study_id", "initialization_seed_namespace", "data", "model", "candidates")}}
    train_transform, prediction_transform = transforms or (None, None)
    visual_source = None
    if candidate == "geo_only":
        _require(visual is None, "Geo-only cannot accept an image encoder")
    elif visual is None:
        from safetensors import safe_open
        from .geo_helpfulness_oof import _pinned_snapshot

        binding = audited_inputs._bind_m1(base, root)
        source_path = _path(root, base["paths"]["source_protocol_root"]) / "protocol/resolved_protocol.yaml"
        source = _yaml(source_path)
        _require(canonical_sha256(source) == canonical_sha256(_yaml(_path(root, config["source_config"]))),
                 "Source adaptation configuration differs from frozen protocol")
        snapshot, snapshot_hashes = _pinned_snapshot(source)
        frozen = source["experts"]["image_encoder"]["externally_pretrained_fixed"]
        architecture = json.loads((snapshot / frozen["open_clip_config_filename"]).read_text())["model_cfg"]
        provenance["pretrained"] = {
            "checkpoint_id": frozen["checkpoint_id"], "hub_revision": frozen["hub_revision"],
            "snapshot_path": str(snapshot), "snapshot_file_sha256": snapshot_hashes,
        }
        provenance["encoder_architecture"] = {
            "embed_dim": architecture["embed_dim"], "vision_cfg": architecture["vision_cfg"],
            "quick_gelu": architecture.get("quick_gelu", False),
        }
        provenance["encoder_builder"] = {"library": "open_clip_torch", "version": version("open_clip_torch"),
                                         "callable": "open_clip.model._build_vision_tower"}
        provenance["historical_preprocessing_recipe"] = deepcopy(source["experts"]["image_encoder"][
            "fold_contained_adaptation"]["adaptation_recipe"]["preprocessing"])
        # Construct only the required visual tower. The unused text tower is
        # never allocated or loaded. load_state_dict's default copy semantics
        # keep parameter storage independent of safetensors' mapped source.
        visual, train_transform, prediction_transform = _construct_visual_and_transforms(provenance)
        if initialization == "cs_adapted":
            record = next(item for item in binding["audit"]["encoders"] if item["seed"] == seed)
            path = _path(root, base["paths"]["encoder_template"].format(seed=seed))
            _require(sha256_file(path) == record["checkpoint"]["sha256"], "CS-adapted encoder hash changed")
            provenance["adapted_checkpoint"] = {"path": str(path.relative_to(root)),
                                                "sha256": record["checkpoint"]["sha256"]}
            prefix = ""
        else:
            path, prefix = snapshot / frozen["checkpoint_filename"], "visual."
        source_sha256 = (record["checkpoint"]["sha256"] if initialization == "cs_adapted"
                         else snapshot_hashes[frozen["checkpoint_filename"]])
        visual_source = (path, prefix, source_sha256)
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            state = {name.removeprefix(prefix): checkpoint.get_tensor(name)
                     for name in checkpoint.keys() if name.startswith(prefix)}
            _require(set(state) == set(visual.state_dict()), "Visual source keys differ from the pinned architecture")
            visual.load_state_dict(state, strict=True)
            _require(all(torch.equal(value, state[name]) for name, value in visual.state_dict().items()),
                     "Loaded visual parameter values differ from checkpoint source tensors")
            _require(all(value.data_ptr() != state[name].data_ptr()
                         for name, value in visual.state_dict().items()),
                     "Visual parameters unexpectedly alias checkpoint storage")
            del state
        provenance["visual_loading"] = {
            "source_prefix": prefix, "materialized_text_tensors": False,
            "source_storage_independent": True, "exact_source_values": True,
            "load_state_dict_assign": False,
        }
    model = OnlineFusionModel(visual, fusion, regime=regime, train_transform=train_transform,
                              prediction_transform=prediction_transform, strict_scope=strict_scope,
                              provenance=provenance)
    model = model.float().to(device)
    if visual_source is not None:
        _verify_initial_visual_transfer(model.visual, *visual_source)
        model.provenance["visual_loading"].update(
            post_transfer_exact_source_values=True,
            post_transfer_source_sha256_verified=True,
            initial_device=str(device),
        )
    # A plain Python attribute, never a parameter/buffer or serialized model state.
    # Exploratory frozen references must originate in the pinned source, not a
    # potentially changed live-model copy made after the source check.
    model._initial_visual_source = visual_source
    return model


def canonical_frozen_values(model: nn.Module, names: list[str]) -> dict[str, torch.Tensor] | None:
    """Read frozen values from the pinned original checkpoint, without repairing the model."""
    source = getattr(model, "_initial_visual_source", None)
    if source is None:
        _require(getattr(model, "provenance", {}).get("initialization_is_injected_test_fixture", False)
                 or getattr(model, "provenance", {}).get("fixture") is not None,
                 "Missing authenticated original source for exploratory frozen baseline")
        return None
    from safetensors import safe_open

    path, prefix, digest = source
    _require(sha256_file(path) == digest, "Canonical frozen checkpoint source hash changed")
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        _require(all(prefix + name in checkpoint.keys() for name in names),
                 "Canonical frozen checkpoint lacks required parameters")
        reference = {name: checkpoint.get_tensor(prefix + name) for name in names}
    _require(sha256_file(path) == digest, "Canonical frozen source hash changed during baseline read")
    return reference


def _verify_initial_visual_transfer(visual: nn.Module, source_path: Path,
                                    prefix: str, expected_sha256: str) -> None:
    """Check every initial device tensor against the authenticated CPU source."""
    from safetensors import safe_open

    _require(sha256_file(source_path) == expected_sha256,
             "Initial visual checkpoint source hash changed after device transfer")
    with safe_open(str(source_path), framework="pt", device="cpu") as checkpoint:
        names = {name.removeprefix(prefix) for name in checkpoint.keys() if name.startswith(prefix)}
        current = visual.state_dict()
        _require(names == set(current), "Initial visual keys changed after device transfer")
        for name, value in current.items():
            reference = checkpoint.get_tensor(prefix + name)
            _require(torch.equal(value.detach().cpu(), reference),
                     f"Initial visual values differ from source after device transfer: {name}")


def _construct_visual_and_transforms(provenance: Mapping[str, Any]) -> tuple[nn.Module, Any, Any]:
    """Build the recorded vision architecture and exact historical transforms."""
    from open_clip.model import _build_vision_tower
    from open_clip.transform import PreprocessCfg, image_transform_v2
    from .geo_helpfulness_oof import _enforce_transform_contract

    builder = provenance["encoder_builder"]
    _require(builder["library"] == "open_clip_torch"
             and builder["callable"] == "open_clip.model._build_vision_tower"
             and builder["version"] == version("open_clip_torch"),
             "Exported visual builder version differs from installed OpenCLIP")
    architecture = deepcopy(provenance["encoder_architecture"])
    _require(architecture["vision_cfg"].get("timm_model_pretrained", False) is False,
             "Standalone reconstruction must not download pretrained weights")
    preprocessing = provenance["historical_preprocessing_recipe"]
    train_cfg, prediction_cfg = preprocessing["train_transform"], preprocessing["prediction_transform"]
    architecture["vision_cfg"]["image_size"] = tuple(prediction_cfg["resize_size"])
    visual = _build_vision_tower(**architecture, cast_dtype=None)
    preprocess = PreprocessCfg(
        size=tuple(prediction_cfg["resize_size"]), mean=tuple(prediction_cfg["normalize_mean"]),
        std=tuple(prediction_cfg["normalize_std"]), interpolation="bicubic", resize_mode="squash",
    )
    train_transform = image_transform_v2(preprocess, is_train=True, aug_cfg={
        "scale": tuple(train_cfg["random_resized_crop_scale"]),
    })
    prediction_transform = image_transform_v2(preprocess, is_train=False)
    _enforce_transform_contract(train_transform, prediction_transform, {"preprocessing": preprocessing})
    return visual, train_transform, prediction_transform


def make_model_from_manifest(provenance: Mapping[str, Any], device: str | torch.device,
                             *, visual: nn.Module | None = None,
                             strict_scope: bool = True) -> OnlineFusionModel:
    """Reconstruct architecture without any original weights, paths or downloads.

    The caller must authenticate its exported manifest and then load the saved
    visual/fusion state dictionaries strictly before performing predictions.
    This function deliberately never opens the original encoder or HF snapshot.
    """
    candidate, seed = provenance["candidate"], int(provenance["seed"])
    config = deepcopy(provenance["fusion_model_config"])
    fusion = (build_bypass_model if candidate in READOUTS else build_fusion_model)(config, candidate, seed)
    train_transform = prediction_transform = None
    if candidate != "geo_only" and visual is None:
        visual, train_transform, prediction_transform = _construct_visual_and_transforms(provenance)
    if candidate == "geo_only":
        _require(visual is None, "Geo-only cannot restore a visual encoder")
    model = OnlineFusionModel(
        visual, fusion, regime=provenance["regime"], train_transform=train_transform,
        prediction_transform=prediction_transform, strict_scope=strict_scope, provenance=provenance,
    )
    return model.float().to(device)
