import torch
from torchvision.transforms import functional as F
import torch.nn as nn
from torchvision.transforms import v2
import cv2
import numpy as np
from PIL import Image

class BottomSquareCrop:
    def __init__(self, crop_size):
        """
        Args:
            crop_size (int): The size of the square crop.
        """
        self.crop_size = crop_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image or Tensor): Input image.
        Returns:
            PIL.Image or Tensor: Cropped image.
        """
        img_width, img_height = image.size  # Get size info from image processed with opencv as numpy array
        if self.crop_size > img_width or self.crop_size > img_height:
            raise ValueError("Crop size must be smaller than the image dimensions.")

        # Calculate the starting position for the crop
        top = img_height - self.crop_size  # Align the bottom
        left = (img_width - self.crop_size) // 2  # Center horizontally
        return F.crop(image, top, left, self.crop_size, self.crop_size)

class SupConTwoViewTransform:
    """
    Create two views of the same image. Copied from https://github.com/HobbitLong/SupContrast/blob/master/util.py.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class CannyTransformRGB(nn.Module):
    """
    Enhance the original image with Edge maps from canny edge detection on RGB channels. Referring to [How to write your own v2 transforms]
    (https://pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html#just-create-a-nn-module-and-override-the-forward-method)
    and [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).
    """
    def __init__(self, threshold1=10, threshold2=50, alpha=0.5, beta=0.5, gamma=0):
        """
        Parameters:
            threshold1: lower threshold for the hysteresis procedure.
            threshold2: upper threshold for the hysteresis procedure.
            alpha: weight of the original image.
            beta: weight of the edge map.
            gamma: scalar added to each sum.
        """
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, img: Image.Image) -> Image.Image:
        # Convert PIL image to NumPy array
        img_np = np.array(img)
        if img_np.ndim == 3:
            # Split the image into R, G, and B channels
            R, G, B = cv2.split(img_np)
            # Apply Canny on each channel
            edges_R = cv2.Canny(R, self.threshold1, self.threshold2)
            edges_G = cv2.Canny(G, self.threshold1, self.threshold2)
            edges_B = cv2.Canny(B, self.threshold1, self.threshold2)
            edges = cv2.merge([edges_R, edges_G, edges_B])
        else:
            # If already single channel, simply apply Canny
            edges = cv2.Canny(img_np, self.threshold1, self.threshold2)

        # edged_img = cv2.addWeighted(img_np, self.alpha, edges, self.beta, self.gamma)

        # Convert back to a PIL image
        return Image.fromarray(edges)

class EdgeAug(nn.Module):
    """
    Enhance the original image with Edge maps from canny edge detection on RGB channels. Referring to [How to write your own v2 transforms]
    (https://pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html#just-create-a-nn-module-and-override-the-forward-method)
    and [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).
    """
    def __init__(self, threshold1=10, threshold2=50, alpha=0.5, beta=0.5, gamma=0):
        """
        Parameters:
            threshold1: lower threshold for the hysteresis procedure.
            threshold2: upper threshold for the hysteresis procedure.
            alpha: weight of the original image.
            beta: weight of the edge map.
            gamma: scalar added to each sum.
        """
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, img: torch.tensor):
        """
        This augmentation follows normalisation in the preprocessing. So it first converts the tensor back to numpy for
        Edge detection, then converts the numpy back to tensor. It returns a list consisting of both the original img
        and its corresponding edge map. This transform can be plugin at the end of data_preprocessing (v2.transforms).
        Must note that the actual batch size will be doubled during training.
        :param img:
        :return:
        """
        # Convert PIL image to NumPy array
        img_np = (img.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        if img_np.ndim == 3:
            # Split the image into R, G, and B channels
            R, G, B = cv2.split(img_np)
            # Apply Canny on each channel
            edges_R = cv2.Canny(R, self.threshold1, self.threshold2)
            edges_G = cv2.Canny(G, self.threshold1, self.threshold2)
            edges_B = cv2.Canny(B, self.threshold1, self.threshold2)
            edges = cv2.merge([edges_R, edges_G, edges_B])
        else:
            # If already single channel, simply apply Canny
            edges = cv2.Canny(img_np, self.threshold1, self.threshold2)

        # edged_img = cv2.addWeighted(img_np, self.alpha, edges, self.beta, self.gamma)

        edged_img = torch.from_numpy(edges).permute(2, 0, 1).float() / 255.

        # Convert back to a PIL image
        return [img, edged_img]

class CannyTransformGS(nn.Module):
    """
    Edge Map transformation with canny edge detection. Referring to [How to write your own v2 transforms]
    (https://pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html#just-create-a-nn-module-and-override-the-forward-method)
    and [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).
    """
    def __init__(self, threshold1=10, threshold2=50):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def forward(self, img: Image.Image) -> Image.Image:
        # Convert PIL image to NumPy array
        img_np = np.array(img)
        # Convert to grayscale if the image is in color
        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Apply the Canny edge detector
        edges = cv2.Canny(img_np, self.threshold1, self.threshold2)
        # Convert back to a PIL image
        return Image.fromarray(edges)

def canny_preprocessing(args: dict):
    """
    Prepares the data transformations with edge detection
    :param args: dict: configuration parameters
    :return:
    """
    transform_list = []

    crop_size = args.get('augmentations', {}).get('crop', 384)
    if isinstance(crop_size, int) or crop_size == 'ratio':
        if crop_size == 'ratio':
            crop_size = int(args['resize'] * 0.875)
    else:
        raise ValueError("Invalid value for 'crop_size'. It must be an integer or the string 'ratio'.")

    # transform_list.append(v2.Resize(crop_size))
    if args['augmentations'].get('bottom_crop', False):
        transform_list.append(BottomSquareCrop(crop_size))
    elif args['augmentations'].get('random_crop', False):
        transform_list.append(v2.RandomResizedCrop(crop_size, scale=(0.5, 1.0)))
    else:
        transform_list.append(v2.CenterCrop(crop_size))

    if args.get('augmentations', {}).get('flip', False):
        transform_list.append(v2.RandomHorizontalFlip())
    if args.get('augmentations', {}).get('rotation', False):
        transform_list.append(v2.RandomRotation(degrees=30))

    transform_list.append(CannyTransformRGB(20, 50))
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

class TwoViewTransform:
    """
    Create two views of the same image. Copied from https://github.com/HobbitLong/SupContrast/blob/master/util.py.
    """
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]
