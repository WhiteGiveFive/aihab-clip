# W&B Hyperparameter Sweeps for OpenCLIP Fine-Tuning

## What’s here
- `sweeps/openclip_ft.yaml`: grid sweep over user defined parameters, optimizing `val_acc`.
- `tools/run_ft_sweep.py`: launcher that loads configs, applies sweep overrides (including dotted keys), and runs the usual training/validation loop. It prepends the repo root to `sys.path` so importing `main` works from any working directory.

## How the command works
The YAML `command` uses W&B placeholders:
- `${program}` → `tools/run_ft_sweep.py`
- `${interpreter}` → your Python executable
- `${args}` → expands to `--key=value` for every sweep parameter in the current trial
Example expansion for a sampled point:
```
python tools/run_ft_sweep.py \
  --base_config configs/base.yaml \
  --dataset_config configs/cs.yaml 
  optional (--lr_v=0.0001 --train_epoch=50 --finetune.unlocked_groups=2)
```

## Running the sweep
From the repo root:
```bash
wandb sweep --project aihab-clip-sweep sweeps/openclip_ft.yaml
# copy the printed SWEEP_ID, then
wandb agent SWEEP_ID
# optional: limit runs per agent
wandb agent SWEEP_ID --count 3
```
Run one agent per GPU for faster grids.

## Adding more hyperparameters
1) Add them under `parameters:` in `sweeps/openclip_ft.yaml`.
2) Do nothing else: `${args}` will pass every key/value to `tools/run_ft_sweep.py`, and the launcher will type-coerce and insert them into the config (nested keys via dots are supported).

## How overrides are applied (`load_cfg_with_overrides`)
- `parse_args()` keeps known flags (base/dataset configs, inspect_only) and puts everything else in `unknown` (this is what `${args}` expands to). `unknonw` is shown to return `--key=value` format. 
- `_pairwise_overrides(unknown)` accepts both `--key value` and `--key=value` forms from W&B and builds a dict `{key: raw_string_value}`.
- `_get_cfg_value(cfg, path_parts)` walks the loaded config using a dotted path (e.g., `['finetune','unlocked_groups']`) to fetch the current value as a type hint; returns `None` if missing.
- `_coerce_value(val_str, ref)` converts the string to the right Python type using the hint (`bool`, `int`, `float`, etc.); falls back to `literal_eval` or the raw string.
- `_set_cfg_value(cfg, path_parts, value)` writes the coerced value back into the config, creating nested dicts if needed.
- `load_cfg_with_overrides(...)` ties it together: load base/dataset configs, then apply every override from `${args}` so you never have to hardcode sweep parameters.
Example: `unknown = ['--lr_v=5e-05', '--train_epoch=10', '--finetune.unlocked_groups=1']` becomes:
`{'lr_v': 5e-05, 'train_epoch': 10, 'finetune.unlocked_groups': 1}` with correct types.

## Direct single-run sanity check
```bash
python tools/run_ft_sweep.py \
  --base_config configs/base.yaml \
  --dataset_config configs/cs.yaml \
  --lr_v=5e-5 --train_epoch=10 --finetune.unlocked_groups=1
```

## Troubleshooting
- `ModuleNotFoundError: No module named 'main'`: resolved by the launcher adding the repo root to `sys.path`.
- `Unpaired override args`: fixed; the parser now accepts both `--key value` and `--key=value` forms emitted by W&B `${args}`.
