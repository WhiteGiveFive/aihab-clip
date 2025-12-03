import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .method import FSCLIPmethod
from .utils import cls_acc
from tqdm import tqdm
from collections import defaultdict


class FTOpenCLIP(FSCLIPmethod):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.cfg = args

    def forward(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                text_weights: torch.tensor,
                model: nn.Module,
                classnames,
                shots: int, 
                config_file: str
                ):

        cfg = self.cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)    # perhaps we can skip this, as we have loaded the model on device in model init

        # Set up the trainable layers
        unlocked_groups = int(cfg.get('unlocked_groups', 1))
        model.lock_image_tower(unlocked_groups=unlocked_groups)
        model.lock_text_tower()

        # Print out the information about the trainable layers
        frozen, trainable = [], []
        for name, p in model.named_parameters():
            (trainable if p.requires_grad else frozen).append(name)

        print(f"Trainable params: {len(trainable)} ({len(trainable)/(len(trainable)+len(frozen)):.1%})")
        print(f"Frozen params   : {len(frozen)}")
        
        groups = defaultdict(list)
        for n in trainable:
            top = n.split('.')[1] if '.' in n else n  # e.g., visual.trunk.attn_pool.*
            groups[top].append(n)
        for g, names in groups.items():
            print(f"  {g}: {len(names)} params")

        print("Trainable (sample):", trainable[:10])

        # Initialize the optimizer and scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=cfg['lr_v'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'])

        # Training loop
        print('\nStart Training procedure')
        for train_idx in range(cfg['train_epoch']):
            correct_samples, all_samples = 0, 0
            running_loss, running_batches = 0.0, 0
            print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

            model.train()

            for i, (images, targets) in enumerate(tqdm(train_loader)):
                images, targets = images.to(device), targets.to(device)
                with torch.autocast(device_type=device):
                    image_features = model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100.0 * image_features @ text_weights # logit_scale is ignored

                    loss = F.cross_entropy(logits, targets)

                acc = cls_acc(logits, targets)
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                running_loss += loss.item()
                running_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = running_loss / max(running_batches, 1)
            lr_curr = optimizer.param_groups[0]['lr']
            print('Acc: {:.4f} ({:}/{:}), Avg Loss: {:.4f}, LR: {:.2e}'.format(correct_samples / all_samples, correct_samples, all_samples, avg_loss, lr_curr))
            scheduler.step()
        torch.cuda.empty_cache()
        
        return avg_loss, correct_samples / all_samples
