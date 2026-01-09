import argparse
from ast import literal_eval
from types import SimpleNamespace
from pathlib import Path
import sys

# Ensure repo root is on sys.path when launched from elsewhere (e.g., wandb agent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wandb

from main import load_configs, set_seed
from data.templates import CS_CLASSNAMES
from data.dataloader import build_loaders
from aihab_utils.feature_cache import _feature_cache_dir, cache_preprojection_features, _feature_cache_exists
from aihab_utils.model_init import init_clip_and_text_head, inspect
from aihab_utils.evalution import draw_cm
from methods.ProLIP import ProLIP
from methods.PEFT_openclip import FTOpenCLIP


def parse_args():
    """
    Parse known sweep boilerplate flags; leave any parameter overrides
    (e.g., --lr_v 1e-4 --finetune.unlocked_groups 2) in `unknown`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, default='configs/base.yaml')
    parser.add_argument('--dataset_config', type=str, default='configs/cs.yaml')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None)
    parser.add_argument('--inspect_only', action='store_true', help='Run data/transform inspection only')
    args, unknown = parser.parse_known_args()
    return args, unknown


def _pairwise_overrides(unknown):
    """
    Convert unknown args into a dict of key -> string value.
    Supports both pair style ['--lr_v', '1e-4'] and equals style ['--lr_v=1e-4'].
    """
    overrides = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        key, value = None, None

        # Handle --key=value form
        if '=' in token:
            key, value = token.split('=', 1)
        else:
            # Expect a following value token
            if i + 1 >= len(unknown):
                raise ValueError(f"Unpaired override args: {unknown}")
            key, value = token, unknown[i + 1]
            i += 1  # consume the value token too

        if key.startswith('--'):
            key = key[2:]

        overrides[key] = value
        i += 1
    return overrides


def _get_cfg_value(cfg, path_parts):
    node = cfg
    for p in path_parts:
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return None
    return node


def _coerce_value(val_str, ref):
    """
    Coerce a string override into the type of `ref` when possible.
    Fallback: best-effort literal_eval, else raw string.
    """
    if ref is None:
        try:
            return literal_eval(val_str)
        except Exception:
            return val_str
    if isinstance(ref, bool):
        return str(val_str).lower() in ('1', 'true', 't', 'yes', 'y')
    if isinstance(ref, int) and not isinstance(ref, bool):
        return int(val_str)
    if isinstance(ref, float):
        return float(val_str)
    try:
        return type(ref)(val_str)
    except Exception:
        try:
            return literal_eval(val_str)
        except Exception:
            return val_str


def _set_cfg_value(cfg, path_parts, value):
    node = cfg
    for p in path_parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[path_parts[-1]] = value


def load_cfg_with_overrides(args: argparse.Namespace, unknown):
    """
    Reuse existing config loader then patch arbitrary keys (supports dotted paths)
    coming from W&B CLI params without hardcoding each one.
    """
    base_args = SimpleNamespace(
        base_config=args.base_config,
        dataset_config=args.dataset_config,
        opts=args.opts,
        inspect_only=args.inspect_only,
    )
    cfg = load_configs(base_args)

    overrides = _pairwise_overrides(unknown)
    for k, v_str in overrides.items():
        path = k.split('.')
        ref = _get_cfg_value(cfg, path)
        v = _coerce_value(v_str, ref)
        _set_cfg_value(cfg, path, v)
    return cfg


def maybe_init_wandb(cfg):
    wandb_run = None
    use_wandb = cfg.get('finetune', {}).get('enabled', False) and cfg.get('wandb_project', None)
    if use_wandb:
        run_name = (
            f"{cfg.get('dataset', 'ds')}_"
            f"shots{cfg.get('shots', 0)}_"
            f"seed{cfg.get('seed', 1)}_"
            f"{cfg.get('backbone', 'clip')}_"
            f"{cfg.get('train_epoch', 0)}eps_"
            f"ug{cfg.get('finetune', {}).get('unlocked_groups', 1)}"
        )
        project_name = f"{cfg.get('wandb_project')}_sweep"
        wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config=cfg,
        )
    return wandb_run


def run(cfg, dataset_config_path: str, inspect_only: bool = False):
    set_seed(int(cfg.get('seed', 1)))
    backend = str(cfg.get('clip_backend', 'openai')).lower()

    wandb_run = maybe_init_wandb(cfg)

    clip_bundle = init_clip_and_text_head(cfg)

    # Select transforms: prefer model-native preprocess for OpenCLIP when enabled
    train_tf_override, test_tf_override = None, None
    use_model_preprocess = bool(cfg.get('use_model_preprocess', backend == 'openclip'))
    if use_model_preprocess and backend == 'openclip':
        train_tf_override = clip_bundle.get('preprocess_train', None)
        test_tf_override = clip_bundle.get('preprocess_val', None)

    dl_tr, dl_val, dl_te, train_tf, test_tf, info = build_loaders(
        cfg, train_tf_override=train_tf_override, test_tf_override=test_tf_override)

    inspect(cfg, train_tf, test_tf, dl_tr, dl_val, dl_te, info, clip_bundle)

    if inspect_only:
        print("\nInspection-only run; skipping caching and training.")
        if wandb_run is not None:
            wandb_run.finish()
        return

    # Optionally cache features
    if bool(cfg.get('save_features', False)):
        cache_preprojection_features(cfg, clip_bundle, dl_tr, info)

    do_finetune = cfg.get('finetune', {}).get('enabled', False)

    if do_finetune and backend == 'openclip':
        finetuner = FTOpenCLIP(cfg)
        clip_model = clip_bundle['clip_model']
        text_weights = clip_bundle['text_weights']
        loss, top1_acc, top3_acc, f1, mcc, cm = finetuner(
            train_loader=dl_tr,
            val_loader=dl_val,
            test_loader=dl_te,
            text_weights=text_weights,
            model=clip_model,
            classnames=CS_CLASSNAMES,
            shots=int(cfg.get('shots', 0) or 0),
            config_file=Path(dataset_config_path).stem,
            return_valid=False,
            prompt_tokens=clip_bundle['prompt_tokens'],
            num_templates=clip_bundle['num_templates'],
        )
        print("\n==== OpenCLIP Finetune results ====")
        print(f"Loss: {loss}, Top-1 Accuracy: {top1_acc}, Top-3 Accuracy: {top3_acc}, F1 (weighted): {f1}, MCC: {mcc}")
        # log to W&B
        if wandb_run is not None:
            wandb_run.log({
                'top1_acc': float(top1_acc) if hasattr(top1_acc, 'item') else top1_acc,
                'top3_acc': float(top3_acc) if hasattr(top3_acc, 'item') else top3_acc,
                'f1': f1, 
                'mcc': mcc, 
                'loss': float(loss) if hasattr(loss, 'item') else loss,
            })

            if cm is not None:
                cm_rows = [[true_name] + cm[i].tolist() for i, true_name in enumerate(CS_CLASSNAMES)]
                wandb_run.log({
                    "confusion_matrix": wandb.Table(
                        data=cm_rows,
                        columns=["true_label"] + list(CS_CLASSNAMES),
                    )
                })
                draw_cm(cm, label_list=CS_CLASSNAMES)
    elif do_finetune and backend == 'openai':
        cache_dir = _feature_cache_dir(cfg)
        aug_views = int(cfg.get('aug_views', 1) or 1)
        if not _feature_cache_exists(cache_dir, aug_views):
            if cfg.get('finetune', {}).get('require_cached_features', True):
                raise FileNotFoundError(f"Cached features not found in {cache_dir}; run with save_features=True first.")
            else:
                print(f"[warn] Cached features missing in {cache_dir}; generating now.")
                cache_preprojection_features(cfg, clip_bundle, dl_tr, info)

        prolip = ProLIP(cfg)

        state_dict = clip_bundle['state_dict']
        clip_model = clip_bundle['clip_model']
        text_weights = clip_bundle['text_weights']
        text_weights_before = clip_bundle['text_weights_before']

        loss, acc = prolip(
            train_loader=dl_tr,
            val_loader=dl_val,
            test_loader=dl_te,
            test_loader_v2=None,
            test_loader_sketch=None,
            test_loader_a=None,
            test_loader_r=None,
            text_weights=text_weights,
            text_weights_a=None,
            text_weights_r=None,
            text_weights_before=text_weights_before,
            model=clip_model,
            state_dict=state_dict,
            classnames=CS_CLASSNAMES,
            task=int(cfg.get('seed', 1)),
            shots=int(cfg.get('shots', 0) or 0),
            config_file=Path(dataset_config_path).stem,
            test_config_path=str(dataset_config_path)
        )
        print("\n==== ProLIP results ====")
        print(f"Loss: {loss}, Accuracy: {acc}")
        if wandb_run is not None:
            wandb_run.log({
                'val_acc': float(acc) if hasattr(acc, 'item') else acc,
                'val_loss': float(loss) if hasattr(loss, 'item') else loss,
            })
    else:
        print("\nFinetune disabled (finetune.enabled=False).")

    if wandb_run is not None:
        wandb_run.finish()


def main():
    args, unknown = parse_args()
    cfg = load_cfg_with_overrides(args, unknown)
    run(cfg, dataset_config_path=args.dataset_config, inspect_only=args.inspect_only)


if __name__ == '__main__':
    main()
