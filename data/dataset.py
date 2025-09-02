from cProfile import label

import cv2
import pandas as pd
import numpy as np
import os
from PIL import Image
from numpy.f2py.crackfortran import verbose
from tqdm import tqdm
import logging

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from . import REASSIGN_NAME_LABEL_L3L2, CORRUPT_IMAGES


def get_image_label(image_name: str, file_list: pd.DataFrame, verbose: bool = False):
    """
    retrieve the image label from the index file, for a single file
    :param image_name: the name of the image
    :param file_list: the index file
    :param verbose: whether to show the missing files in the index list
    :return: the retrieved image label: bh_plot shall only be None or an int equal or greater than 0,
    bh_plot_desc will only be empty str or actual word labels found in REASSIGN_NAME_LABEL_L3L2. If bh_plot_desc is an
    empty str, bh_plot can only be None. The same to bh_poly and bh_poly_desc.
    """
    # Convert the image name and file list names to lowercase to ensure case-insensitive
    image_name = image_name.lower()
    file_list['file'] = file_list['file'].str.lower()

    sample_row = file_list[file_list['file'] == image_name]
    # Filter out invalid rows that without valid 'BH_PLOT_DESC' value.
    if verbose:
        if len(sample_row) > 1:
            print(f"Warning: Multiple entries found for {image_name}")
    sample_row = sample_row[sample_row['BH_PLOT_DESC'].notna() & (sample_row['BH_PLOT_DESC'] != '')]

    # Skip those files cannot be found if additional condition is needed, consider adding "and pd.notna(sample_row['BH_PLOT'].values[0])"
    if not sample_row.empty:
        # Read the plot label and plot id for images
        bh_plot_desc = sample_row['BH_PLOT_DESC'].values[0]
        # Handle NaN or empty string cases for bh_plot_desc. Actually it is not needed because invalid rows have filtered out
        if pd.isna(bh_plot_desc) or bh_plot_desc == '':
            bh_plot_desc = ''
        # Retrieve bh_plot based on bh_plot_desc. If bh_plot_desc is not found in the dict, return None for bh_plot
        bh_plot = None
        if bh_plot_desc in REASSIGN_NAME_LABEL_L3L2:
            bh_plot = REASSIGN_NAME_LABEL_L3L2[bh_plot_desc][0]
        else:
            if verbose:
                print(f'Unrecognized label {bh_plot_desc} found in the dataset')

        plot_id = sample_row['ID'].values[0]

        # Initialise bh_poly and bh_poly_desc as None and empty string in case they don't exist
        bh_poly, bh_poly_desc = None, ''

        # Check if the "BH_POLYDESC" is present in the file_list. If true, do the same as plot labels.
        if "BH_POLYDESC" in sample_row.columns:
            bh_poly_desc = sample_row['BH_POLYDESC'].values[0]
            if pd.isna(bh_poly_desc) or bh_poly_desc == '':
                bh_poly_desc = ''

            if bh_poly_desc in REASSIGN_NAME_LABEL_L3L2:
                bh_poly = REASSIGN_NAME_LABEL_L3L2[bh_poly_desc][0]

        # Deal with the "Boundary and Linear Features" labels. Use valid poly label if possible.
        if bh_plot_desc == 'Boundary and Linear Features':
            bh_plot = bh_poly
            bh_plot_desc = bh_poly_desc
        return bh_plot, bh_plot_desc, bh_poly, bh_poly_desc, plot_id
    else:
        if verbose:
            print(f"Image {image_name} not found or has no BH PLOT")
        return None, "", None, "", ""

def convert_to_coarse_label(word_label: str) -> int:
    """
    Maps the level 3 labels to level 2 labels
    :param word_label: the level 3 word labels
    :return: the numerical level 2 level
    """
    return REASSIGN_NAME_LABEL_L3L2.get(word_label, -1)[1] # Return -1 if no mapping found

def load_images_from_folder(folder_path: str, index_file_name: str, resize_dim: int, verbose: bool = False):
    """
    Loads images from a single folder
    :param folder_path: the path of the folder
    :param index_file_name: the index file name
    :param resize_dim: The resize dimension of the original image.
    :param verbose: whether to show the missing files in the index list.
    :return:
    """
    images = []
    plot_labels = []
    plot_word_labels = []
    poly_labels = []
    poly_word_labels = []
    file_names = []
    plot_idx = []
    image_sources = []

    index_path = os.path.join(folder_path, index_file_name)
    file_list = pd.read_csv(index_path)

    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) & (f not in CORRUPT_IMAGES)]

    # Counter for successfully loaded images
    success_count = 0

    for image_file in tqdm(image_files, desc=f"Loading images from {folder_path}", unit='file'):
        bh_plot, bh_plot_desc, bh_poly, bh_poly_desc, plot_id = get_image_label(image_file, file_list)
        if bh_plot is not None and bh_plot != '':
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            if image is not None:
                # Resize the images to reduce memory consumption
                image_resized = cv2.resize(image, (resize_dim, resize_dim))
                images.append(image_resized)
                plot_labels.append(bh_plot)

                # Append metadata
                plot_word_labels.append(bh_plot_desc)
                poly_labels.append(bh_poly)
                poly_word_labels.append(bh_poly_desc)
                file_names.append(image_file)
                plot_idx.append(plot_id)
                image_sources.append(folder_path)

                success_count += 1
            else:
                print(f"Warning: Could not read image {image_file}")

    images = np.array(images)
    plot_labels = np.array(plot_labels)
    poly_labels = np.array(poly_labels)

    logging.info(f"Successfully loaded {success_count} images from {folder_path}.")

    if verbose:
        save_data_path = os.path.join(folder_path, 'loaded_' + index_file_name)
        csv_data = pd.DataFrame({
            'file_names': file_names,
            'plot_labels': plot_labels,
            'plot_word_labels': plot_word_labels,
            'poly_labels': poly_labels,
            'poly_word_labels': poly_word_labels,
            'plot_idx': plot_idx,
            'image_sources': image_sources
        })
        csv_data.to_csv(save_data_path, index=False)
        logging.info(f"Loaded file names and labels saved to {save_data_path}")

    return images, plot_labels, plot_word_labels, poly_labels, poly_word_labels, file_names, plot_idx, image_sources

def image_loader(folder_paths: list, index_file_names: list, resize_dim: int, verbose: bool = False):
    """
    Load images from multiple folders and merge them
    :param folder_paths: list consisting of the paths of the image folder
    :param index_file_names: list consisting of the index file name, corresponding to the image folder
    :param resize_dim: the resize dimension of the images
    :param verbose: if output word labels
    :return: two lists, a dataset of images and their labels
    """
    all_images = []
    all_plot_labels = []
    all_plot_word_labels = []
    all_poly_labels = []
    all_poly_word_labels = []
    all_file_names = []
    all_plot_idx = []
    all_image_sources = []

    for idx, folder_path in enumerate(folder_paths):
        images, plot_labels, plot_word_labels, poly_labels, poly_word_labels, file_names, plot_idx, image_sources \
            = load_images_from_folder(folder_path, index_file_names[idx], resize_dim, verbose)

        # Concatenate the np.array data
        if len(all_images) == 0:
            all_images = images
            all_plot_labels = plot_labels
            all_poly_labels = poly_labels
        else:
            all_images = np.concatenate((all_images, images), axis=0)
            all_plot_labels = np.concatenate((all_plot_labels, plot_labels), axis=0)
            all_poly_labels = np.concatenate((all_poly_labels, poly_labels), axis=0)

        # Concatenate the list data
        all_plot_word_labels.extend(plot_word_labels)
        all_poly_word_labels.extend(poly_word_labels)
        all_file_names.extend(file_names)
        all_plot_idx.extend(plot_idx)
        all_image_sources.extend(image_sources)

    logging.info(f"Total images loaded: {all_images.shape[0]}")

    return (all_images, all_plot_labels, all_plot_word_labels, all_poly_labels, all_poly_word_labels, all_file_names,
            all_plot_idx, all_image_sources)

def data_partition(args: dict):
    image_folder, index_file_name = args['dataset_paths'], args['index_file_names']
    resize_dim = args['preprocessing'].get('resize', 256)
    image_data = image_loader(image_folder, index_file_name, resize_dim, verbose=True)

    # Partition the dataset
    images = image_data[0]
    labels = image_data[1]
    plot_idx = image_data[-2]

    train_idx_list, valid_idx_list = [], []
    if args['data_split'].get("if_grouped", False):
        n_splits = args['data_split']['num_fold'] if args['data_split']['num_fold'] >= 2 else 2
        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args['data_split']['split_seed'])
        for train_idx, valid_idx in sgk.split(images, labels, groups=plot_idx):
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)

            # If just do cross validation once
            if args['data_split'].get("first_cv_only", True):
                break
    else:
        train_idx, valid_idx = train_test_split(
            np.arange(len(images)),
            test_size=args['data_split']['valid_split'],
            stratify=labels,
            random_state=args['data_split']['split_seed']
        )
        train_idx_list.append(train_idx)
        valid_idx_list.append(valid_idx)
    return image_data, train_idx_list, valid_idx_list

class HABDATA(Dataset):
    """
    This class create the habitat dataset. It can select train, valid, test set. It can also select samples with dataidxs.
    """

    def __init__(self, image_folder, index_file_name, partition, args, dataidxs=None, transform=None, target_transform=None):
        images, labels, _, _, _, _, _, _ = image_loader(image_folder, index_file_name)
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform

        X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=args['data_split']['valid_split'],
                                                              stratify=labels, random_state=args['data_split']['split_seed'])

        if partition == 'train':
            self.images = X_train
            self.labels = y_train
        elif partition == 'valid':
            self.images = X_valid
            self.labels = y_valid
        else:
            exit('wrong partition for creating dataset')

        if self.dataidxs is not None:
            self.images, self.labels = self.images[self.dataidxs], self.labels[self.dataidxs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

class HABMETADATA(Dataset):
    """
    This class create the habitat dataset with meta information, including poly labels and file names.
    """

    def __init__(self, image_folder, index_file_name, partition, args, dataidxs=None, transform=None, target_transform=None):
        """
        load images in folders with metadata
        :param image_folder: a list consisting of the paths of image folders
        :param index_file_name: a list consisting of the index file names, corresponding to the image folders
        :param partition: train, valid, test
        :param args: configuration parameters
        :param dataidxs:
        :param transform:
        :param target_transform:
        """
        images, labels, plot_word_labels, poly_labels, poly_word_labels, file_names, plot_idx, image_sources \
            = image_loader(image_folder, index_file_name, verbose=True)
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform

        # Level 2 label conversion
        l2_labels = np.array([convert_to_coarse_label(word_label) for word_label in plot_word_labels])

        # Perform the train/test split and obtain the indices
        train_idx, valid_idx = train_test_split(
            np.arange(len(images)),
            test_size=args['data_split']['valid_split'],
            stratify=labels,
            random_state=args['data_split']['split_seed']
        )

        if partition == 'train':
            selected_idx = train_idx
        elif partition == 'valid':
            selected_idx = valid_idx
        else:
            raise ValueError("Invalid partition for creating dataset: must be 'train' or 'valid'")

        # Use the selected indices to split all relevant arrays
        self.images = images[selected_idx]
        self.labels = labels[selected_idx]
        self.l2_labels = l2_labels[selected_idx]
        self.poly_labels = poly_labels[selected_idx]
        self.plot_word_labels = [plot_word_labels[idx] for idx in selected_idx] # list indicing
        self.poly_word_labels = [poly_word_labels[idx] for idx in selected_idx] # list indicing
        self.file_names = [file_names[idx] for idx in selected_idx] # list indicing
        self.plot_idx = [plot_idx[idx] for idx in selected_idx] # list indicing
        self.image_sources = [image_sources[idx] for idx in selected_idx]

        if self.dataidxs is not None:
            self.images, self.labels = self.images[self.dataidxs], self.labels[self.dataidxs]
            self.l2_labels = self.l2_labels[self.dataidxs]
            self.poly_labels = self.poly_labels[self.dataidxs]
            self.plot_word_labels = [self.plot_word_labels[idx] for idx in self.dataidxs]   # list indicing
            self.poly_word_labels = [self.poly_word_labels[idx] for idx in self.dataidxs]   # list indicing
            self.file_names = [self.file_names[idx] for idx in self.dataidxs]   # list indicing
            self.plot_idx = [self.plot_idx[idx] for idx in self.dataidxs]
            self.image_sources = [self.image_sources[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert None type in ploy_label to -1, otherwise error occurs when iterating dataloader
        ploy_label = self.poly_labels[idx] if self.poly_labels[idx] is not None else -1

        # Create a metadata dictionary to store additional information
        metadata = {
            'l2_label': self.l2_labels[idx],
            'poly_label': ploy_label,
            'plot_word_label': self.plot_word_labels[idx],
            'poly_word_label': self.poly_word_labels[idx],
            'file_name': self.file_names[idx],
            'plot_idx': self.plot_idx[idx],
            'image_source': self.image_sources[idx]
        }

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, metadata

class HABMETADATA_SUBSET(Dataset):
    """
    A subset of the image data based on preloaded images and metadata. This class is built to make the imaging loading
    efficient.
    """
    def __init__(self, images, labels, l2_labels, poly_labels, plot_word_labels, poly_word_labels, file_names, plot_idx,
                 image_sources, selected_idx, args, transform=None, target_transform=None):
        """
        Initialise the dataset with preloaded data and selected indices for dataset partitioning.
        :param images: preloaded images, numpy array
        :param labels: bh_plot labels, numpy array
        :param l2_labels: l2 labels, numpy array
        :param poly_labels: bh_poly labels, numpy array
        :param plot_word_labels: plot descriptions, list of strings
        :param poly_word_labels: poly descriptions, list of strings
        :param file_names: corresponding file names, list of strings
        :param plot_idx: corresponding plot ids, list of strings
        :param image_sources: corresponding image folder, list of strings
        :param selected_idx: partitioning index, train, valid, or test, numpy array
        :param transform: data transformations
        :param target_transform: target transformations
        """
        self.images = images[selected_idx]
        self.labels = labels[selected_idx]
        self.l2_labels = l2_labels[selected_idx]
        self.poly_labels = poly_labels[selected_idx]
        self.plot_word_labels = [plot_word_labels[idx] for idx in selected_idx] # list indicing
        self.poly_word_labels = [poly_word_labels[idx] for idx in selected_idx] # list indicing
        self.file_names = [file_names[idx] for idx in selected_idx] # list indicing
        self.plot_idx = [plot_idx[idx] for idx in selected_idx] # list indicing
        self.image_sources = [image_sources[idx] for idx in selected_idx] # list indicing
        self.transform = transform
        self.target_transform = target_transform

        self.args = args

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.args.get('use_l2_label', False):
            label = self.l2_labels[idx]
        else:
            label = self.labels[idx]

        # Convert None type in ploy_label to -1, otherwise error occurs when iterating dataloader
        ploy_label = self.poly_labels[idx] if self.poly_labels[idx] is not None else -1

        # Create a metadata dictionary to store additional information
        metadata = {
            'l2_label': self.l2_labels[idx],
            'poly_label': ploy_label,
            'plot_word_label': self.plot_word_labels[idx],
            'poly_word_label': self.poly_word_labels[idx],
            'file_name': self.file_names[idx],
            'plot_idx': self.plot_idx[idx],
            'image_source': self.image_sources[idx]
        }

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, metadata

class HABMETADATA_obliterate(Dataset):
    """
    This class create the habitat dataset with meta information, including poly labels and file names.
    """

    def __init__(self, image_folder, index_file_name, dataidxs=None, transform=None, target_transform=None):
        """
        load images in folders with metadata
        :param image_folder: a list consisting of the paths of image folders
        :param index_file_name: a list consisting of the index file names, corresponding to the image folders
        :param partition: train, valid, test
        :param args: configuration parameters
        :param dataidxs:
        :param transform:
        :param target_transform:
        """
        images, labels, plot_word_labels, poly_labels, poly_word_labels, file_names, plot_idx, image_sources \
            = image_loader(image_folder, index_file_name, verbose=True)
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform

        # Level 2 label conversion
        l2_labels = np.array([convert_to_coarse_label(word_label) for word_label in plot_word_labels])

        # Use the selected indices to split all relevant arrays
        self.images = images
        self.labels = labels
        self.l2_labels = l2_labels
        self.poly_labels = poly_labels
        self.plot_word_labels = plot_word_labels
        self.poly_word_labels = poly_word_labels
        self.file_names = file_names
        self.plot_idx = plot_idx
        self.image_sources = image_sources

        if self.dataidxs is not None:
            self.images, self.labels = self.images[self.dataidxs], self.labels[self.dataidxs]
            self.l2_labels = self.l2_labels[self.dataidxs]
            self.poly_labels = self.poly_labels[self.dataidxs]
            self.plot_word_labels = [self.plot_word_labels[idx] for idx in self.dataidxs]   # list indicing
            self.poly_word_labels = [self.poly_word_labels[idx] for idx in self.dataidxs]   # list indicing
            self.file_names = [self.file_names[idx] for idx in self.dataidxs]   # list indicing
            self.plot_idx = [self.plot_idx[idx] for idx in self.dataidxs]
            self.image_sources = [self.image_sources[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert None type in ploy_label to -1, otherwise error occurs when iterating dataloader
        ploy_label = self.poly_labels[idx] if self.poly_labels[idx] is not None else -1

        # Create a metadata dictionary to store additional information
        metadata = {
            'l2_label': self.l2_labels[idx],
            'poly_label': ploy_label,
            'plot_word_label': self.plot_word_labels[idx],
            'poly_word_label': self.poly_word_labels[idx],
            'file_name': self.file_names[idx],
            'plot_idx': self.plot_idx[idx],
            'image_source': self.image_sources[idx]
        }

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, metadata
