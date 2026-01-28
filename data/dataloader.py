from .dataset import convert_to_coarse_label, image_loader, data_partition, HABDATA, HABMETADATA, HABMETADATA_SUBSET
from . import l2_names_to_l3
import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, Subset
import timm
import numpy as np
from sklearn.model_selection import train_test_split

from .data_utils import BottomSquareCrop, SupConTwoViewTransform, TwoViewTransform, canny_preprocessing
import os
from typing import List, Tuple
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image
from data.clip_transforms import build_clip_transforms


def _data_preprocessing(args: dict, is_train: bool):
    """
    Prepares the data transformations
    :param args: dict: configuration parameters
    :return:
    """
    transform_list = []

    # You possibly do not need this resize, since opencv has resize the images while reading them
    # if 'resize' in args:
    #     resize_dims =args['resize']
    #     transform_list.append(transforms.Resize(resize_dims))
    crop_size = args.get('augmentations', {}).get('crop', 384)
    if isinstance(crop_size, int) or crop_size == 'ratio':
        if crop_size == 'ratio':
            crop_size = int(args['resize'] * 0.875)
    else:
        raise ValueError("Invalid value for 'crop_size'. It must be an integer or the string 'ratio'.")

    if is_train:
        if args['augmentations'].get('bottom_crop', False):
            transform_list.append(BottomSquareCrop(crop_size))
        elif args['augmentations'].get('random_crop', False):
            transform_list.append(v2.RandomResizedCrop(crop_size, scale=(0.5, 1.0)))
        else:
            transform_list.append(v2.CenterCrop(crop_size))

        if args.get('augmentations', {}).get('flip', False):
            transform_list.append(v2.RandomHorizontalFlip())
            # transform_list.append(v2.RandomVerticalFlip())
        if args.get('augmentations', {}).get('rotation', False):
            transform_list.append(v2.RandomRotation(degrees=30))
    else:
        transform_list.append(v2.Resize(crop_size))

    transform_list.append(v2.ToTensor())
    # transform_list.append(v2.ToImage())
    # transform_list.append(v2.ToDtype(torch.float32, scale=True))

    if args.get('normalise', False):
        normalize_params = args.get('normalise_params',
                                    {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
        mean = normalize_params['mean']
        std = normalize_params['std']
        transform_list.append(v2.Normalize(mean=mean, std=std))

    return v2.Compose(transform_list)

def timm_transforms(is_train: bool):
    model = timm.create_model(
        'vit_large_patch16_siglip_384',
        pretrained=True,
        num_classes=0,
    )
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=is_train)
    return transforms

def get_dataloaders(args: dict):
    train_transforms = _data_preprocessing(args['preprocessing'], is_train=True)
    val_transforms = _data_preprocessing(args['preprocessing'], is_train=False)

    # index_file_name = os.path.join(args['dataset_path'], args['index_file_name'])
    index_file_name = args['index_file_names']
    if args.get('metadata', False):
        trainset = HABMETADATA(args['dataset_paths'], index_file_name, partition='train', args=args, transform=train_transforms)
        valset = HABMETADATA(args['dataset_paths'], index_file_name, partition='valid', args=args, transform=val_transforms)
    else:
        trainset = HABDATA(args['dataset_paths'], index_file_name, partition='train', args=args, transform=train_transforms)
        valset = HABDATA(args['dataset_paths'], index_file_name, partition='valid', args=args, transform=val_transforms)

    trainloader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=args['shuffle'],
                             num_workers=args['num_workers'])
    valloader = DataLoader(valset, batch_size=args['batch_size'], shuffle=args['shuffle'],
                           num_workers=args['num_workers'])
    print('================Dataloaders are created.=================')
    return trainloader, valloader

def efficiently_get_dataloaders(args: dict):
    train_transforms = _data_preprocessing(args['preprocessing'], is_train=True)
    val_transforms = _data_preprocessing(args['preprocessing'], is_train=False)

    # Load the image data once for all
    image_data, train_idx_list, valid_idx_list = data_partition(args)
    images, labels, plot_word_labels, poly_labels, poly_word_labels, file_names, plot_idx, image_sources = image_data

    # Convert plot word labels to level 2 labels
    l2_labels = np.array([convert_to_coarse_label(word_label) for word_label in plot_word_labels])

    # Create the training dataset
    trainset = HABMETADATA_SUBSET(
        images=images, labels=labels, l2_labels=l2_labels, poly_labels=poly_labels, plot_word_labels=poly_word_labels,
        poly_word_labels=poly_word_labels, file_names=file_names, plot_idx=plot_idx, image_sources=image_sources,
        args=args, selected_idx=train_idx_list[0], transform=train_transforms
    )

    valset = HABMETADATA_SUBSET(
        images=images, labels=labels, l2_labels=l2_labels, poly_labels=poly_labels, plot_word_labels=poly_word_labels,
        poly_word_labels=poly_word_labels, file_names=file_names, plot_idx=plot_idx, image_sources=image_sources,
        args=args, selected_idx=valid_idx_list[0], transform=val_transforms
    )

    trainloader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=args['shuffle'],
                             num_workers=args['num_workers'])
    valloader = DataLoader(valset, batch_size=args['batch_size'], shuffle=args['shuffle'],
                           num_workers=args['num_workers'])
    print('================Dataloaders are created.=================')
    return trainloader, valloader

class CrossValidDataloaders:
    def __init__(self, args: dict):
        """
        This dataloader manager is used for cross validation.
        Initialise the EfficientDataloaderManager with multiple dataloaders based on the splits from Kfold.
        :param args:
        """
        self.args = args

        # Define transformations for training and validation sets.
        self.train_transforms = _data_preprocessing(args['preprocessing'], is_train=True)
        self.val_transforms = _data_preprocessing(args['preprocessing'], is_train=False)

        # If transform data with multiple views
        if args['preprocessing']['multi_views'].get('supcon', False):
            self.train_transforms = SupConTwoViewTransform(self.train_transforms)
            print("==========>Same two views of transformation created<==========")
        if args['preprocessing']['multi_views'].get('edge', False):
            canny_transforms = canny_preprocessing(args['preprocessing'])
            self.train_transforms = TwoViewTransform(self.train_transforms, canny_transforms)
            print("==========>Different two views of transformation created<==========")

        # Load the dataset and partition it into indices for training and validation sets.
        self.image_data, self.train_idx_list, self.valid_idx_list = data_partition(args)
        (self.images, self.labels, self.plot_word_labels, self.poly_labels, self.poly_word_labels,
         self.file_names, self.plot_idx, self.image_sources) = self.image_data

        # Convert plot word labels to level 2 labels
        self.l2_labels = np.array([convert_to_coarse_label(word_label) for word_label in self.plot_word_labels])

        # Initialise the list of Dataloader pairs
        self.trainvalid_dls = self._create_dataloaders()

    def _create_dataloaders(self):
        """
        Create pairs of dataloaders for all splits from Kfold.
        :return: Paired train and valid dataloaders in a list.
        """
        dataloader_pairs = []

        # Loop over all splits in train_idx_list and valid_idx_list
        for train_idx, valid_idx in zip(self.train_idx_list, self.valid_idx_list):
            trainset = HABMETADATA_SUBSET(
                images=self.images, labels=self.labels, l2_labels=self.l2_labels, poly_labels=self.poly_labels,
                plot_word_labels=self.plot_word_labels, poly_word_labels=self.poly_word_labels,
                file_names=self.file_names, plot_idx=self.plot_idx, image_sources=self.image_sources,
                args=self.args, selected_idx=train_idx, transform=self.train_transforms
            )

            valset = HABMETADATA_SUBSET(
                images=self.images, labels=self.labels, l2_labels=self.l2_labels, poly_labels=self.poly_labels,
                plot_word_labels=self.plot_word_labels, poly_word_labels=self.poly_word_labels,
                file_names=self.file_names, plot_idx=self.plot_idx, image_sources=self.image_sources,
                args=self.args, selected_idx=valid_idx, transform=self.val_transforms
            )

            trainloader = DataLoader(trainset, batch_size=self.args['batch_size'], shuffle=self.args['shuffle'],
                                     num_workers=self.args['num_workers'], pin_memory=True)
            valloader = DataLoader(valset, batch_size=self.args['batch_size'], shuffle=self.args['shuffle'],
                                   num_workers=self.args['num_workers'], pin_memory=True)
            dataloader_pairs.append((trainloader, valloader))

        return dataloader_pairs

    def get_dataloaders(self, idx: int = 0):
        """
        Return the train dataloader and validation dataloader pair based on the provided index.
        :param idx: The index of the chosen Dataloader pair.
        :return: (trainloader, valloader) pair for the specified index.
        """
        if idx < 0 or idx >= len(self.trainvalid_dls):
            raise IndexError("Index out of range. Please provide a valid index.")

        return self.trainvalid_dls[idx]

class TrainTestDataLoaders:
    """
    This class is used for generate train and test dataloaders. Different from CrossValidation dataloaders,
    TrainTestDataLoaders generates a training set and a test set for report the performance.
    """

    def __init__(self, args: dict):
        self.args = args

        # Define transformations for training and validation sets.
        self.train_transforms = _data_preprocessing(args['preprocessing'], is_train=True)
        self.test_transforms = _data_preprocessing(args['preprocessing'], is_train=False)
        
        # If transform data with multiple views for the supervised contrastive learning method
        if args['preprocessing']['multi_views'].get('supcon', False):
            self.train_transforms = SupConTwoViewTransform(self.train_transforms)
            print("==========>Same two views of transformation created<==========")
        if args['preprocessing']['multi_views'].get('edge', False):
            canny_transforms = canny_preprocessing(args['preprocessing'])
            self.train_transforms = TwoViewTransform(self.train_transforms, canny_transforms)
            print("==========>Different two views of transformation created<==========")

        # Set up the path to train set and test set
        self.train_image_folders = self.args['dataset_paths']   # a list of folder paths
        self.train_index_file_names = self.args['index_file_names'] # a list of index file paths

        self.test_image_folders = [image_folder.replace('_train', '_test') for image_folder in self.train_image_folders]
        # self.test_image_folders = [self.test_image_folders[0]]    # If trained on both CS datasets but test on one
        self.test_index_file_names = self.train_index_file_names

    def _create_dataloader(self, image_folder, index_file_name, data_transforms, train_for_valid : bool = False):
        """
        Create a train or a test dataloader from a given folder and index file.
        :param image_folder: the designation of the image folder
        :param index_file_name: label file
        :param data_transforms: transforms
        :param train_for_valid: generate the training set with test transforms if set to True
        :return: a dataloader
        """
        resize_dim = self.args['preprocessing'].get('resize', 256)
        image_data = image_loader(image_folder, index_file_name, resize_dim, verbose=True)
        images, labels, plot_word_labels, poly_labels, poly_word_labels, file_names, plot_idx, image_sources = image_data

        # Convert plot word labels to level 2 labels
        l2_labels = np.array([convert_to_coarse_label(word_label) for word_label in plot_word_labels])

        # Set selection to all instances
        selected_idx = np.arange(images.shape[0])

        # Create the dataset
        ds = HABMETADATA_SUBSET(
            images=images, labels=labels, l2_labels=l2_labels, poly_labels=poly_labels,
            plot_word_labels=plot_word_labels,
            poly_word_labels=poly_word_labels, file_names=file_names, plot_idx=plot_idx, image_sources=image_sources,
            args=self.args, selected_idx=selected_idx, transform=data_transforms
        )
        dl = DataLoader(ds, batch_size=self.args['batch_size'], shuffle=self.args['shuffle'],
                        num_workers=self.args['num_workers'], pin_memory=True)

        # Training data with test transforms, for the SupCon pretraining
        if train_for_valid:
            train_valid_ds = HABMETADATA_SUBSET(
                images=images, labels=labels, l2_labels=l2_labels, poly_labels=poly_labels,
                plot_word_labels=plot_word_labels,
                poly_word_labels=poly_word_labels, file_names=file_names, plot_idx=plot_idx,
                image_sources=image_sources,
                args=self.args, selected_idx=selected_idx, transform=self.test_transforms
            )
            train_valid_dl = DataLoader(train_valid_ds, batch_size=self.args['batch_size'], shuffle=self.args['shuffle'],
                        num_workers=self.args['num_workers'], pin_memory=True)
            return dl, train_valid_dl
        else:
            return dl

    def get_dataloaders(self, train_for_valid: bool = False):
        test_dl = self._create_dataloader(self.test_image_folders, self.test_index_file_names, self.test_transforms)

        if not train_for_valid:
            train_dl = self._create_dataloader(
                self.train_image_folders, self.train_index_file_names, self.train_transforms)
            return train_dl, test_dl
        else:
            # Create an additional Training set with test transforms, used by SupCon Pretraining feature visualisation.
            train_dl, train_for_valid_dl = self._create_dataloader(
                self.train_image_folders, self.train_index_file_names, self.train_transforms, train_for_valid)
            return train_dl, test_dl, train_for_valid_dl

def efficiently_get_dataloaders_obliterate(args: dict):
    train_transforms = _data_preprocessing(args['preprocessing'], is_train=True)
    val_transforms = _data_preprocessing(args['preprocessing'], is_train=False)

    # Load the image data once for all
    image_dir, index_file_name = args['dataset_paths'], args['index_file_names']
    full_dataset = HABMETADATA(image_dir, index_file_name, dataidxs=None, transform=train_transforms)

    # Get all indices and perform the split into train and validation
    full_indices = np.arange(len(full_dataset))
    train_idx, valid_idx = train_test_split(
        full_indices,
        test_size=args['data_split']['valid_split'],
        stratify=full_dataset.labels,
        random_state=args['data_split']['split_seed']
    )

    # Create Subsets for train and validation
    trainset = Subset(full_dataset, train_idx)
    valset = Subset(full_dataset, valid_idx)

    # Since the original dataset was initialized with train_transforms, update the transform for validation set
    # This method could be problematic in terms of assigning transforms for train, valid.
    full_dataset.transform = val_transforms

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=args['shuffle'],
                             num_workers=args['num_workers'])
    valloader = DataLoader(valset, batch_size=args['batch_size'], shuffle=args['shuffle'],
                           num_workers=args['num_workers'])

    print('================Dataloaders are created.=================')
    return trainloader, valloader

def few_shot_indices(labels: np.ndarray, shots: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample N=shots examples per class, with replacement if class has < shots. Helper function for the aihab-clip project."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    sel = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        if len(idx_c) >= shots:
            sel.extend(rng.choice(idx_c, size=shots, replace=False).tolist())
        else:
            sel.extend(rng.choice(idx_c, size=shots, replace=True).tolist())
    return np.array(sel, dtype=np.int64)

def derive_test_paths(train_paths: List[str]) -> List[str]:
    """Helper function for the aihab-clip project."""
    return [p.replace('_train', '_test') for p in train_paths]

def _stratified_group_split_indices(labels: np.ndarray,
                                   groups: np.ndarray,
                                   val_ratio: float,
                                   seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function for the aihab-clip project.
    StratifiedGroup split: preserve class balance while keeping grouped samples together
    (here, group = plot_idx). Uses StratifiedGroupKFold to approximate the requested split.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if val_ratio <= 0:
        return np.arange(len(labels), dtype=np.int64), np.array([], dtype=np.int64)

    n_splits = max(2, int(round(1.0 / val_ratio)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(labels, labels, groups=groups))
    return train_idx.astype(np.int64), val_idx.astype(np.int64)

class CSArrayDataset(Dataset):
    """
    Helper function for the aihab-clip project.
    Simple Dataset wrapper over preloaded arrays from image_loader.
    Returns (image, label) pairs suitable for CLIP feature extraction.
    Optionally returns metadata for validation/testing.
    """
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 file_names: List[str],
                 selected_idx: np.ndarray,
                 transform,
                 plot_word_labels=None,
                 poly_labels=None,
                 poly_word_labels=None,
                 plot_idx=None,
                 image_sources=None,
                 l2_labels=None,
                 return_metadata: bool = False):
        self.images = images[selected_idx]
        self.labels = labels[selected_idx]
        self.file_names = [file_names[i] for i in selected_idx]
        self.transform = transform
        self.return_metadata = return_metadata

        def _sel(arr):
            if arr is None:
                return None
            if isinstance(arr, list):
                return [arr[i] for i in selected_idx]
            return np.asarray(arr)[selected_idx]

        self.plot_word_labels = _sel(plot_word_labels)
        self.poly_labels = _sel(poly_labels)
        self.poly_word_labels = _sel(poly_word_labels)
        self.plot_idx = _sel(plot_idx)
        self.image_sources = _sel(image_sources)

        if l2_labels is not None:
            self.l2_labels = _sel(l2_labels)
        elif self.plot_word_labels is not None:
            self.l2_labels = np.array(
                [convert_to_coarse_label(w) for w in self.plot_word_labels]
            )
        else:
            self.l2_labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        lbl = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        if not self.return_metadata:
            return img, lbl

        poly_label = -1
        if self.poly_labels is not None:
            poly_label = self.poly_labels[idx] if self.poly_labels[idx] is not None else -1

        metadata = {
            "l2_label": int(self.l2_labels[idx]) if self.l2_labels is not None else -1,
            "poly_label": int(poly_label) if poly_label is not None else -1,
            "plot_word_label": self.plot_word_labels[idx] if self.plot_word_labels is not None else "",
            "poly_word_label": self.poly_word_labels[idx] if self.poly_word_labels is not None else "",
            "file_name": self.file_names[idx],
            "plot_idx": self.plot_idx[idx] if self.plot_idx is not None else "",
            "image_source": self.image_sources[idx] if self.image_sources is not None else "",
        }
        return img, lbl, metadata

def build_loaders(cfg,
                  train_tf_override=None,
                  test_tf_override=None) -> Tuple[DataLoader, DataLoader, DataLoader, object, object, dict]:
    """Helper function for the aihab-clip project."""
    # Build transforms: prefer overrides (e.g., OpenCLIP native preprocess) when provided
    if train_tf_override is not None and test_tf_override is not None:
        train_tf = train_tf_override
        test_tf = test_tf_override
    else:
        resolution = cfg['data']['preprocessing']['resolution']
        train_tf = build_clip_transforms(cfg['data']['preprocessing'], is_train=True, resolution=resolution)
        test_tf = build_clip_transforms(cfg['data']['preprocessing'], is_train=False, resolution=resolution)

    # Resolve subset config (optional)
    subset_l2_names = cfg.get('subset_l2_names', []) or []
    if isinstance(subset_l2_names, str):
        subset_l2_names = [subset_l2_names]
    subset_l3_names, subset_l3_ids = l2_names_to_l3(subset_l2_names)
    use_subset = len(subset_l3_ids) > 0

    # Bulk load train split
    images_tr, labels_tr, plot_word_labels_tr, poly_labels_tr, poly_word_labels_tr, file_names_tr, plot_idx_tr, src_tr = \
        image_loader(cfg['data']['dataset_paths'], cfg['data']['index_file_names'], cfg['data']['preprocessing'].get('resize', 256), verbose=True)

    # Bulk load test split (derive _test folder unless explicit test paths provided)
    test_paths = cfg['data'].get('test_dataset_paths', None)
    if test_paths:
        if isinstance(test_paths, str):
            test_paths = [test_paths]
    else:
        test_paths = derive_test_paths(cfg['data']['dataset_paths'])

    test_index_names = cfg['data'].get('test_index_file_names', None)
    if test_index_names:
        if isinstance(test_index_names, str):
            test_index_names = [test_index_names]
    else:
        test_index_names = cfg['data']['index_file_names']

    if len(test_paths) != len(test_index_names):
        raise ValueError(f"Mismatch: test_dataset_paths has {len(test_paths)} entries but "
                         f"test_index_file_names has {len(test_index_names)}.")

    images_te, labels_te, plot_word_labels_te, poly_labels_te, poly_word_labels_te, file_names_te, plot_idx_te, src_te = \
        image_loader(test_paths, test_index_names, cfg['data']['preprocessing'].get('resize', 256), verbose=True)

    # Optional subset filtering by L2 -> L3 mapping (keep original L3 label ids)
    if use_subset:
        mask_tr = np.isin(labels_tr, subset_l3_ids)
        images_tr = images_tr[mask_tr]
        labels_tr = labels_tr[mask_tr]
        plot_word_labels_tr = [x for x, m in zip(plot_word_labels_tr, mask_tr) if m]
        poly_labels_tr = poly_labels_tr[mask_tr]
        poly_word_labels_tr = [x for x, m in zip(poly_word_labels_tr, mask_tr) if m]
        file_names_tr = [x for x, m in zip(file_names_tr, mask_tr) if m]
        plot_idx_tr = np.asarray(plot_idx_tr)[mask_tr]
        src_tr = [x for x, m in zip(src_tr, mask_tr) if m]

        mask_te = np.isin(labels_te, subset_l3_ids)
        images_te = images_te[mask_te]
        labels_te = labels_te[mask_te]
        plot_word_labels_te = [x for x, m in zip(plot_word_labels_te, mask_te) if m]
        poly_labels_te = poly_labels_te[mask_te]
        poly_word_labels_te = [x for x, m in zip(poly_word_labels_te, mask_te) if m]
        file_names_te = [x for x, m in zip(file_names_te, mask_te) if m]
        plot_idx_te = np.asarray(plot_idx_te)[mask_te]
        src_te = [x for x, m in zip(src_te, mask_te) if m]

    # Select indices
    seed = int(cfg.get('seed', 1))
    rng = np.random.RandomState(seed)
    val_cfg = cfg.get('data', {}).get('data_split', {})
    val_ratio = float(val_cfg.get('valid_split', 0.1))
    val_seed = int(val_cfg.get('split_seed', seed))

    train_pool_idx, val_idx = _stratified_group_split_indices(labels_tr, plot_idx_tr, val_ratio, val_seed)

    shots_val = int(cfg.get('shots', 0)) if cfg.get('shots', 0) is not None else 0
    if shots_val > 0:
        # Few-shot selection within the training pool (validation drawn from full data before this step)
        rel_sel = few_shot_indices(labels_tr[train_pool_idx], shots_val, rng)
        sel_tr = train_pool_idx[rel_sel]
    else:
        # Full-data train pool (minus validation)
        sel_tr = train_pool_idx

    sel_te = np.arange(images_te.shape[0])

    # Build datasets and loaders

    ds_tr = CSArrayDataset(
        images_tr, labels_tr, file_names_tr, sel_tr, transform=train_tf,
        plot_word_labels=plot_word_labels_tr, poly_labels=poly_labels_tr,
        poly_word_labels=poly_word_labels_tr, plot_idx=plot_idx_tr,
        image_sources=src_tr, return_metadata=False,
    )
    ds_val = CSArrayDataset(
        images_tr, labels_tr, file_names_tr, val_idx, transform=test_tf,
        plot_word_labels=plot_word_labels_tr, poly_labels=poly_labels_tr,
        poly_word_labels=poly_word_labels_tr, plot_idx=plot_idx_tr,
        image_sources=src_tr, return_metadata=True,
    )
    ds_te = CSArrayDataset(
        images_te, labels_te, file_names_te, sel_te, transform=test_tf,
        plot_word_labels=plot_word_labels_te, poly_labels=poly_labels_te,
        poly_word_labels=poly_word_labels_te, plot_idx=plot_idx_te,
        image_sources=src_te, return_metadata=True,
    )

    dl_tr = DataLoader(ds_tr,
                       batch_size=cfg['data']['batch_size'],
                       shuffle=cfg['data']['shuffle'],
                       num_workers=cfg['data']['num_workers'],
                       pin_memory=True)
    dl_val = DataLoader(ds_val,
                        batch_size=cfg['data']['batch_size'],
                        shuffle=False,
                        num_workers=cfg['data']['num_workers'],
                        pin_memory=True)
    dl_te = DataLoader(ds_te,
                       batch_size=cfg['data']['batch_size'],
                       shuffle=False,
                       num_workers=cfg['data']['num_workers'],
                       pin_memory=True)

    # Few-shot selection map for inspection
    selection_by_class = None
    if shots_val > 0:
        selection_by_class = {}
        classes = np.unique(labels_tr)
        for c in classes:
            idx_c = sel_tr[labels_tr[sel_tr] == c]
            selection_by_class[int(c)] = idx_c.tolist()

    info = {
        'is_few_shot': shots_val > 0,
        'shots': shots_val,
        'train_size': int(len(sel_tr)),
        'train_batches': int(len(dl_tr)),
        'val_size': int(len(val_idx)),
        'val_batches': int(len(dl_val)),
        'val_split': val_ratio,
        'selection_by_class': selection_by_class,
        'subset_enabled': use_subset,
        'subset_l2_names': subset_l2_names,
        'subset_l3_ids': subset_l3_ids,
        'subset_l3_names': subset_l3_names,
    }
    if use_subset:
        print(f'dataloader subset: l2={subset_l2_names} l3_ids={subset_l3_ids} l3_names={subset_l3_names}')

    return dl_tr, dl_val, dl_te, train_tf, test_tf, info
