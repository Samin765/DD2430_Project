import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def show_images(X):
    """Given images as tensor shows them"""
    np_images = X.numpy()
    plt.figure(figsize=(10, 10))
    grid_img = make_grid(torch.tensor(np_images))
    plt.imshow(np.transpose(grid_img, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def return_normal(tensor_image, processor, ant, plot):
    """Reverse normalization and then show image, looks weird otherwise"""
    mean = torch.tensor(processor.feature_extractor.image_mean).view(1, 3, 1, 1)# only colour channels
    std = torch.tensor(processor.feature_extractor.image_std).view(1, 3, 1, 1)
    tensor_image = tensor_image.cpu() * std + mean  # Add back the normalization
    if plot:
      show_images(tensor_image[0:ant,:,:,:])# do not show all
    return tensor_image