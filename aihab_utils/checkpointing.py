from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch

from aihab_utils.feature_cache import _canonical_backbone_name


def _resolve_dir(root: Path, path: str) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = root / out
    return out


def build_openclip_checkpoint_path(cfg: dict, epoch: int, timestamp) -> Path:
    ft_cfg = cfg.get('finetune', {})
    root = Path(cfg.get('root_path', './'))
    ckpt_dir = _resolve_dir(root, ft_cfg.get('save_model_dir', 'model_ckpt'))

    model_raw = cfg.get('open_clip_model', cfg.get('backbone', 'openclip'))
    model_name = _canonical_backbone_name(str(model_raw))
    num_epoch = int(epoch)
    ts = timestamp

    fname = f"{model_name}_{num_epoch}_{ts}.pt"

    fname = fname.replace('/', '_')
    return ckpt_dir / fname


def save_openclip_checkpoint(cfg: dict,
                             model: torch.nn.Module,
                             optimizer: Optional[torch.optim.Optimizer],
                             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                             epoch: int) -> Path:
    ft_cfg = cfg.get('finetune', {})
    save_opt = bool(ft_cfg.get('save_optimizer', True))
    save_sched = bool(ft_cfg.get('save_scheduler', True))

    ts = datetime.now().strftime('%Y%m%d_%H')
    ckpt_path = build_openclip_checkpoint_path(cfg, epoch=epoch, timestamp=ts)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        'model_state': model.state_dict(),
        'epoch': int(epoch),
        'timestamp': ts,
        'cfg': cfg,
        'clip_backend': cfg.get('clip_backend', 'openclip'),
        'open_clip_model': cfg.get('open_clip_model', None),
    }
    if save_opt and optimizer is not None:
        payload['optimizer_state'] = optimizer.state_dict()
    if save_sched and scheduler is not None:
        payload['scheduler_state'] = scheduler.state_dict()

    torch.save(payload, ckpt_path)
    return ckpt_path


def load_openclip_checkpoint(model: torch.nn.Module,
                             ckpt_path: Path,
                             device: Optional[torch.device] = None,
                             strict: bool = True) -> Dict[str, Any]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state', ckpt.get('state_dict', None))
    if state is None:
        raise KeyError("Checkpoint missing 'model_state' key.")
    model.load_state_dict(state, strict=strict)
    return ckpt
    
