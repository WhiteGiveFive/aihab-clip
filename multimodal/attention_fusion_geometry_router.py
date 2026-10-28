"""R1/R2 development-only inputs and geometry scores for fixed M3J experts.

No fitting, threshold selection, routing evaluation or test access occurs here.
Completed training packages remain read only. R2 uses query identities and
expert candidates but no query labels when calculating geometry scores.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import yaml

from .geometry_routing import class_plot_distances, score_candidates


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/multimodal_attention_fusion_geometry_router.yaml")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve(path: str | Path) -> Path:
    return (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        name = handle.name
        np.savez_compressed(handle, **arrays)
    os.replace(name, path)


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict, Path]:
    config = yaml.safe_load(resolve(path).read_text())
    scope = config["scope"]
    require((scope["seed"], scope["candidate"], scope["initialization"], scope["regime"])
            == (1, "attention_cls_skip_original", "pretrained", "joint_partial"),
            "This R1/R2 package is seed-1 pretrained-initialized partial-joint CLS-original only")
    require(scope["checkpoint_policies"] == ["final", "best"]
            and scope["geometry_arms"] == ["pretrained", "checkpoint_matched"]
            and not scope["training_allowed"] and not scope["cleaned_test_allowed"],
            "R1/R2 comparison or data-access scope differs")
    require(config["geometry"] == {
        "k": 5, "epsilon": 1e-8, "reference_role": "train", "query_role": "validation",
        "image_normalization": "l2", "geo_input": "native", "geo_normalization": "l2",
        "distance": "cosine_float64", "plot_distance": "minimum_photo_distance",
        "class_distance": "mean_nearest_k_distinct_plots", "support": "strict_both_candidate_classes",
        "score": "minimum_image_geo_log_distance_ratio"}, "Declared geometry recipe differs")
    require(config["data"] == {"train_images": 3378, "train_plots": 1300,
        "validation_images": 822, "validation_plots": 325, "classes": 18,
        "image_dim": 1152, "geo_dim": 64}, "Declared development cohort differs")
    require(config["extraction"]["device"] == "cuda" and config["extraction"]["batch_size"] == 16
            and config["extraction"]["autocast"] is False, "Extraction recipe differs")
    output = resolve(config["artifact_root"])
    base = resolve("multimodal_artifacts/analysis/cs/gse_100m/attention_fusion_router")
    require(output != base and output.is_relative_to(base), "Use a separate router-study output directory")
    return config, output


def validate_prediction_arrays(arrays: dict, rows: pd.DataFrame, classes: int) -> dict:
    """Align one exported checkpoint to the declared query cohort."""
    required = {"row_uid", "plot_idx", "label", "prediction", "probabilities"}
    require(required <= arrays.keys(), "Missing prediction fields")
    require(classes > 0 and len(rows) > 0 and rows.row_uid.is_unique,
            "Invalid query identities or class count")
    truth = rows.label_id_dense.to_numpy()
    require(truth.dtype.kind in "iu" and ((truth >= 0) & (truth < classes)).all(),
            "Invalid query class IDs")
    for key in ("row_uid", "plot_idx", "label", "prediction"):
        values = np.asarray(arrays[key])
        require(values.shape == (len(rows),), f"Invalid prediction {key} shape")
        if key in ("label", "prediction"):
            require(values.dtype.kind in "iu" and ((values >= 0) & (values < classes)).all(),
                    f"Invalid {key} class IDs")
    ids = np.asarray(arrays["row_uid"]).astype(str)
    require(len(np.unique(ids)) == len(ids), "Duplicate prediction row_uid")
    require(set(ids) == set(rows.row_uid), "Prediction cohort differs")
    order = pd.Series(np.arange(len(ids)), index=ids).loc[rows.row_uid].to_numpy()
    aligned = {key: np.asarray(arrays[key])[order] for key in required}
    require(np.array_equal(aligned["plot_idx"].astype(str), rows.plot_idx.astype(str)),
            "Prediction plot identity differs")
    require(np.array_equal(aligned["label"], rows.label_id_dense), "Prediction labels differ")
    probabilities = aligned["probabilities"]
    require(probabilities.shape == (len(rows), classes) and np.isfinite(probabilities).all()
            and (probabilities >= 0).all()
            and np.allclose(probabilities.sum(axis=1), 1, atol=1e-10, rtol=0),
            "Invalid exported probabilities")
    require(np.array_equal(probabilities.argmax(axis=1), aligned["prediction"]),
            "Exported prediction differs from probability argmax")
    return aligned


def prepare(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    config, output = load_config(config_path)
    if (output / "protocol/declaration.json").exists():
        declaration = read_json(output / "protocol/declaration.json")
        require(declaration["config_sha256"] == canonical_hash(config), "Existing protocol/config differs")
        verify_prepared(output)
        return {"status": "reused", "output": str(output)}
    require(not output.exists() or not any(output.iterdir()),
            "Output must be new/empty or contain this study's completed declaration")
    fusion_root, geo_root = resolve(config["fusion_study"]), resolve(config["geo_study"])
    fusion_dir, geo_dir = fusion_root / config["fusion_run"], geo_root / config["geo_run"]
    sources = {}

    def bind(path: Path) -> None:
        sources[str(path)] = sha256(path)

    bind(resolve(config_path))
    manifests = {}
    for family, directory in (("fusion", fusion_dir), ("geo", geo_dir)):
        manifests[family] = read_json(directory / "checkpoint_manifest.json")
        run = read_json(directory / "run.json")
        require(run["status"] == "complete" and run["seed"] == 1, "Source fit incomplete or wrong seed")
        if family == "fusion":
            require(run["candidate_id"] == config["scope"]["candidate"]
                    and run["regime"] == "joint_partial" and run["initialization"] == "pretrained",
                    "Fusion source identity differs")
        else:
            require(run["candidate_id"] == "geo_only", "Wrong geo source")
        for name in ("checkpoint_manifest.json", "run.json", "run_spec.json"):
            bind(directory / name)
        for policy in ("final", "best"):
            require(manifests[family][policy]["epoch"] == config["checkpoints"][family][policy],
                    f"Unexpected {family}/{policy} checkpoint epoch")
    classes = manifests["fusion"]["class_labels"]
    require(classes == manifests["geo"]["class_labels"] and len(classes) == config["data"]["classes"],
            "Expert class order differs")
    source_inputs = read_json(fusion_root / "inputs/manifest.json")
    bind(fusion_root / "inputs/manifest.json")
    rows = {}
    for role in ("train", "validation"):
        path = fusion_root / "inputs" / f"{role}_rows.parquet"
        bind(path)
        require(sha256(path) == source_inputs["file_sha256"][path.name], "Source row table differs")
        frame = pd.read_parquet(path).sort_values("row_uid").reset_index(drop=True)
        require(len(frame) == config["data"][f"{role}_images"]
                and frame.plot_idx.nunique() == config["data"][f"{role}_plots"]
                and frame.row_uid.is_unique and frame.file.is_unique,
                f"Unexpected {role} identities or counts")
        require(frame.development_role.eq(role).all(), "Source row role differs")
        require(frame.label_id_dense.between(0, len(classes) - 1).all(), "Invalid label ID")
        require(all(classes[int(i)] == name for i, name in zip(frame.label_id_dense, frame.label_name)),
                "Source labels differ from expert class order")
        rows[role] = frame
    for key in ("row_uid", "plot_idx", "file"):
        require(not (set(rows["train"][key]) & set(rows["validation"][key])),
                f"Reference/query overlap: {key}")
    columns = [f"A{i:02d}" for i in range(config["data"]["geo_dim"])]
    pieces = []
    for entry in source_inputs["provenance"]["geo_source_files"]:
        path = resolve(entry["path"])
        bind(path)
        require(sources[str(path)] == entry["sha256"], "Native geo source changed")
        pieces.append(pd.read_parquet(path, columns=["file", "plot_idx", "label_id", *columns]))
    native = pd.concat(pieces, ignore_index=True)
    require(native.file.is_unique, "Native geo file identities repeated")
    require(set(native.file) == set(pd.concat(list(rows.values())).file), "Native geo cohort differs")
    native = native.set_index("file")
    require(native.groupby("plot_idx")[columns].nunique().to_numpy().max() == 1,
            "Native geo varies within a plot")
    provenance = read_json(fusion_dir / "run_spec.json")["model_provenance"]
    pretrained = provenance["pretrained"]
    original_path = Path(pretrained["snapshot_path"]) / "open_clip_model.safetensors"
    encoders = {"pretrained": {"path": str(original_path), "sha256": pretrained[
        "snapshot_file_sha256"][original_path.name], "prefix": "visual.", "epoch": 0}}
    for policy in ("final", "best"):
        ref = manifests["fusion"][policy]["visual"]
        path = (fusion_root / ref["path"]).resolve()
        require(path.is_relative_to(fusion_root / "objects") and path.stat().st_size == ref["size"],
                "Visual checkpoint path/size invalid")
        encoders[f"joint_{policy}"] = {"path": str(path), "sha256": ref["sha256"],
                                       "prefix": "", "epoch": config["checkpoints"]["fusion"][policy]}
    source_config = yaml.safe_load(resolve(config["source_config"]).read_text())
    bind(resolve(config["source_config"]))
    # Record current image identities/size/mtime without reinstating a raw-byte replay gate.
    from .geo_helpfulness_oof import _resolved_image_path
    inventory = []
    for row in pd.concat(list(rows.values())).to_dict("records"):
        path = _resolved_image_path(row, source_config)
        stat = path.stat()
        inventory.append({"row_uid": row["row_uid"], "path": str(path),
                          "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    comparisons = [{"comparison_id": f"{policy}__{arm}", "policy": policy,
                    "geometry_encoder": "pretrained" if arm == "pretrained" else f"joint_{policy}",
                    "fusion_epoch": config["checkpoints"]["fusion"][policy],
                    "geo_epoch": config["checkpoints"]["geo"][policy]}
                   for policy in ("final", "best") for arm in config["scope"]["geometry_arms"]]
    output.mkdir(parents=True, exist_ok=True)
    for role, frame in rows.items():
        path = output / "inputs" / f"{role}_rows.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        selected = native.loc[frame.file]
        require(np.array_equal(selected.plot_idx.astype(str), frame.plot_idx.astype(str))
                and np.array_equal(selected.label_id, frame.label_id_dense), "Native geo alignment differs")
        vectors = selected[columns].to_numpy(dtype=np.float64)
        require(np.isfinite(vectors).all() and (np.linalg.norm(vectors, axis=1) > 0).all(),
                "Nonfinite/zero native geo")
        write_npz(output / "inputs" / f"{role}_geo.npz", {
            "row_uid": frame.row_uid.to_numpy(dtype=str), "plot_idx": frame.plot_idx.to_numpy(dtype=str),
            "geo_native": vectors})
    for family, directory in (("fusion", fusion_dir), ("geo", geo_dir)):
        for policy in ("final", "best"):
            path = directory / f"{policy}_predictions.npz"
            bind(path)
            with np.load(path, allow_pickle=False) as saved:
                arrays = validate_prediction_arrays(dict(saved), rows["validation"], len(classes))
            acc = float((arrays["prediction"] == arrays["label"]).mean())
            require(abs(acc - manifests[family][policy]["metrics"]["acc1"]) < 1e-12,
                    "Prediction accuracy differs from saved checkpoint evaluation")
            write_npz(output / "inputs" / f"{family}_{policy}_predictions.npz", arrays)
    write_json(output / "inputs/image_inventory.json", {"basis": "identity_size_mtime_no_raw_byte_replay_gate",
                                                       "images": inventory})
    input_hashes = {str(path.relative_to(output)): sha256(path) for path in (output / "inputs").iterdir()}
    declaration = {"schema_version": 1, "study_id": config["study_id"],
        "config": config, "config_sha256": canonical_hash(config), "encoders": encoders,
        "model_provenance": provenance, "source_config": source_config,
        "extraction": config["extraction"], "class_labels": classes, "comparisons": comparisons,
        "source_sha256": sources, "input_sha256": input_hashes,
        "result_assurance": config["result_assurance"], "test_data_scored": False,
        "thresholds_selected": False, "routing_applied": False,
        "parent_protocol": "dev_plans/attention_fusion/README.md"}
    write_json(output / "protocol/declaration.json", declaration)
    write_json(output / "protocol/resolved_config.json", config)
    verify_prepared(output)
    return {"status": "prepared", "output": str(output), "comparisons": len(comparisons),
            "encoders": list(encoders), "training_images": 3378, "validation_images": 822}


def verify_prepared(output: Path) -> dict:
    declaration = read_json(output / "protocol/declaration.json")
    require(canonical_hash(declaration["config"]) == declaration["config_sha256"], "Config binding changed")
    for relative, digest in declaration["input_sha256"].items():
        require(sha256(output / relative) == digest, f"Prepared input changed: {relative}")
    for source, digest in declaration["source_sha256"].items():
        require(sha256(Path(source)) == digest, f"Recorded source changed: {source}")
    return declaration


def _geometry_arrays(geometry) -> dict:
    return {field.name: np.asarray(getattr(geometry, field.name)) for field in fields(geometry)}


def score(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    _, output = load_config(config_path)
    declaration = verify_prepared(output)
    cfg = declaration["config"]
    rows = {role: pd.read_parquet(output / "inputs" / f"{role}_rows.parquet")
            for role in ("train", "validation")}
    kwargs = {"reference_row_ids": rows["train"].row_uid.to_numpy(),
              "reference_plot_ids": rows["train"].plot_idx.to_numpy(),
              "reference_labels": rows["train"].label_id_dense.to_numpy(),
              "query_row_ids": rows["validation"].row_uid.to_numpy(),
              "query_plot_ids": rows["validation"].plot_idx.to_numpy(),
              "k": cfg["geometry"]["k"], "num_classes": len(declaration["class_labels"])}
    geo = {}
    for role in rows:
        with np.load(output / "inputs" / f"{role}_geo.npz") as saved:
            geo[role] = saved["geo_native"]
    geo_geometry = class_plot_distances(geo["train"], geo["validation"], **kwargs)
    geometries = {}
    for encoder_id in declaration["encoders"]:
        folder = output / "embeddings" / encoder_id
        manifest = read_json(folder / "manifest.json")
        require(set(manifest["file_sha256"]) == {"train.npz", "validation.npz"},
                "Embedding manifest must bind both role files")
        require(manifest["declaration_sha256"] == sha256(output / "protocol/declaration.json"),
                "Embedding declaration differs")
        for relative, digest in manifest["file_sha256"].items():
            require(sha256(folder / relative) == digest, "Embedding artifact changed")
        vectors = {}
        for role, frame in rows.items():
            with np.load(folder / f"{role}.npz", allow_pickle=False) as saved:
                require(np.array_equal(saved["row_uid"], frame.row_uid)
                        and np.array_equal(saved["plot_idx"], frame.plot_idx), "Embedding alignment differs")
                vectors[role] = saved["image_embedding"]
        geometries[encoder_id] = class_plot_distances(vectors["train"], vectors["validation"], **kwargs)
    directory = output / "scores"
    write_npz(directory / "geo_class_distances.npz", _geometry_arrays(geo_geometry))
    for name, geometry in geometries.items():
        write_npz(directory / f"{name}_class_distances.npz", _geometry_arrays(geometry))
    comparisons = []
    for comparison in declaration["comparisons"]:
        predictions = {}
        for family in ("fusion", "geo"):
            with np.load(output / "inputs" / f"{family}_{comparison['policy']}_predictions.npz") as saved:
                predictions[family] = saved["prediction"]
        candidate = score_candidates(geometries[comparison["geometry_encoder"]], geo_geometry,
                                     predictions["fusion"], predictions["geo"],
                                     epsilon=cfg["geometry"]["epsilon"])
        arrays = _geometry_arrays(candidate)
        arrays.update(row_uid=rows["validation"].row_uid.to_numpy(dtype=str),
                      plot_idx=rows["validation"].plot_idx.to_numpy(dtype=str))
        write_npz(directory / f"{comparison['comparison_id']}.npz", arrays)
        comparisons.append({**comparison, "queries": len(rows["validation"]),
                            "disagreements": int(candidate.disagreement.sum()),
                            "geometry_supported": int(candidate.geometry_supported.sum()),
                            "eligible_disagreements": int(candidate.eligible.sum())})
    manifest = {"declaration_sha256": sha256(output / "protocol/declaration.json"),
        "query_labels_used_for_geometry": False, "thresholds_selected": False,
        "routing_applied": False, "test_data_scored": False, "comparisons": comparisons,
        "source_sha256": {str(p): sha256(p) for p in (
            Path(__file__), REPO_ROOT / "multimodal/geometry_routing.py")},
        "embedding_manifest_sha256": {name: sha256(output / "embeddings" / name / "manifest.json")
                                       for name in geometries},
        "file_sha256": {p.name: sha256(p) for p in sorted(directory.glob("*.npz"))}}
    write_json(directory / "manifest.json", manifest)
    return {"status": "scored_without_routing", "output": str(directory), "comparisons": comparisons}


def verify(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    _, output = load_config(config_path)
    declaration = verify_prepared(output)
    inputs = {role: pd.read_parquet(output / "inputs" / f"{role}_rows.parquet")
              for role in ("train", "validation")}
    embedding_summary = {}
    for encoder_id in declaration["encoders"]:
        folder = output / "embeddings" / encoder_id
        manifest = read_json(folder / "manifest.json")
        require(set(manifest["file_sha256"]) == {"train.npz", "validation.npz"},
                "Embedding manifest must bind both role files")
        require(manifest["declaration_sha256"] == sha256(output / "protocol/declaration.json"),
                "Embedding provenance differs")
        for name, digest in manifest["file_sha256"].items():
            require(sha256(folder / name) == digest, "Embedding digest differs")
        counts = {}
        for role, frame in inputs.items():
            with np.load(folder / f"{role}.npz", allow_pickle=False) as saved:
                require(np.array_equal(saved["row_uid"], frame.row_uid)
                        and np.array_equal(saved["plot_idx"], frame.plot_idx), "Embedding cohort differs")
                vectors = saved["image_embedding"]
            require(vectors.shape == (len(frame), 1152) and np.isfinite(vectors).all(),
                    "Invalid image embedding shape/values")
            require(np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=2e-6),
                    "Image embedding is not unit normalized")
            counts[role] = len(vectors)
        embedding_summary[encoder_id] = counts
    manifest = read_json(output / "scores/manifest.json")
    require(manifest["declaration_sha256"] == sha256(output / "protocol/declaration.json"),
            "Score declaration binding differs")
    expected_files = {"geo_class_distances.npz",
                      *(f"{name}_class_distances.npz" for name in declaration["encoders"]),
                      *(f"{row['comparison_id']}.npz" for row in declaration["comparisons"])}
    require(set(manifest["file_sha256"]) == expected_files, "Incomplete geometry score file inventory")
    require(set(manifest["embedding_manifest_sha256"]) == set(declaration["encoders"]),
            "Incomplete score encoder bindings")
    require(len(manifest["comparisons"]) == len(declaration["comparisons"]),
            "Incomplete comparison inventory")
    for saved, declared in zip(manifest["comparisons"], declaration["comparisons"]):
        require(all(saved.get(key) == value for key, value in declared.items()),
                "Score comparison identity differs")
        require(saved["queries"] == len(inputs["validation"]), "Score query count differs")
    for name, digest in manifest["file_sha256"].items():
        require(sha256(output / "scores" / name) == digest, "Score artifact changed")
    for name, digest in manifest["embedding_manifest_sha256"].items():
        require(sha256(output / "embeddings" / name / "manifest.json") == digest,
                "Score embedding binding differs")
    require(all(manifest[key] is False for key in ("routing_applied", "thresholds_selected",
                "query_labels_used_for_geometry", "test_data_scored")),
            "R2 cannot include routing application/selection, query-label geometry or test scoring")
    summary = {"status": "R1_R2_complete", "result_assurance": declaration["result_assurance"],
        "reference_images": len(inputs["train"]), "reference_plots": inputs["train"].plot_idx.nunique(),
        "query_images": len(inputs["validation"]), "query_plots": inputs["validation"].plot_idx.nunique(),
        "embeddings": embedding_summary,
        "image_vectors": sum(sum(counts.values()) for counts in embedding_summary.values()),
        "comparisons": manifest["comparisons"], "routing_applied": False,
        "thresholds_selected": False, "test_data_scored": False,
        "declaration_sha256": sha256(output / "protocol/declaration.json"),
        "scores_manifest_sha256": sha256(output / "scores/manifest.json")}
    write_json(output / "verification/completion.json", summary)
    return summary
