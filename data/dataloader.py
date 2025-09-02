from .dataset import convert_to_coarse_label, image_loader, data_partition, HABDATA, HABMETADATA, HABMETADATA_SUBSET
import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset
import timm
import numpy as np
from sklearn.model_selection import train_test_split

from .data_utils import BottomSquareCrop, SupConTwoViewTransform, TwoViewTransform, canny_preprocessing
import os


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
