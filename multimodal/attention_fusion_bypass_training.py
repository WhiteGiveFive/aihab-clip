"""Bind the unchanged M3 training engine to the separately versioned bypasses.

The historical trainer source and imported module remain untouched. A private
module instance receives its model factory and provenance provider exactly once
at construction, so legacy and extension bundles can coexist in one process.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable

from . import attention_fusion_training as original
from .attention_fusion_bypass import build_model


PINNED_ENGINE_SHA256 = "494aec62b6040d89eabd776325e952fbd9afd4bd7d772aa12378ba05df4040a0"
SOURCE_FILES = (
    "attention_fusion_training.py", "attention_fusion.py", "models.py",
    "attention_fusion_bypass.py", "attention_fusion_bypass_training.py",
)
TrainingError = original.TrainingError
configuration_hash = original.configuration_hash
classification_metrics = original.classification_metrics
sha256_file = original.sha256_file


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    hashes = {name: sha256_file(directory / name) for name in SOURCE_FILES}
    if hashes["attention_fusion_training.py"] != PINNED_ENGINE_SHA256:
        raise TrainingError("Historical training engine changed; a new declared engine version is required")
    return hashes


def _instantiate_engine(model_factory: Callable, source_hash_provider: Callable) -> ModuleType:
    """Construct an isolated engine; dependencies are fixed before exposing it.

    The explicit dependency arguments also permit a legacy-factory equivalence
    test without changing any production module's globals.
    """
    path = Path(original.__file__).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINNED_ENGINE_SHA256:
        raise TrainingError("Historical training engine source does not match its pinned version")
    name = "multimodal._attention_fusion_bypass_engine_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise TrainingError("Cannot construct the private training engine")
    engine = importlib.util.module_from_spec(spec)
    # Execute the exact authenticated bytes, avoiding a second mutable file read.
    exec(compile(source, str(path), "exec"), engine.__dict__)
    engine.build_model = model_factory
    engine._source_hashes = source_hash_provider
    engine.TrainingError = TrainingError
    return engine


_ENGINE = _instantiate_engine(build_model, _source_hashes)
train_run = _ENGINE.train_run
InferenceBundle = _ENGINE.InferenceBundle
load_inference_bundle = _ENGINE.load_inference_bundle
export_inference_bundle = _ENGINE.export_inference_bundle
