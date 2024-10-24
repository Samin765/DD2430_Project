import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    mean = torch.tensor(processor.feature_extractor.image_mean).view(
        1, 3, 1, 1)  # only colour channels
    std = torch.tensor(processor.feature_extractor.image_std).view(1, 3, 1, 1)
    tensor_image = tensor_image.cpu() * std + mean  # Add back the normalization
    if plot:
        show_images(tensor_image[0:ant, :, :, :])  # do not show all
    return tensor_image


def accuracy(all_predictions, all_labels):
    """ Given labels and predicted labels returns accuracy in %"""
    correct = all_predictions == all_labels
    return (100*correct.sum()/correct.shape[0]).item()


def confussion_matrix(labels, pred_lab, categories, F1=True):
    """ Given labels and predicted labels shows the confussion matrix"""
    acc = accuracy(pred_lab, labels)
    if F1:
        print(classification_report(pred_lab, labels))  # F1
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(labels, pred_lab)
    annot = np.where(cm != 0, cm, '')
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=categories, yticklabels=categories)
    # Add labels
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.title(f"Confusion Matrix, acc {acc:.2f} %", fontsize=15)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    # Show the plot
    plt.show()


def print_images(dataloader, processor, limit=10):
    """Show a few images of the dataset to see errors"""
    for batch_nr, (image_embeds, labels, images) in enumerate(tqdm(dataloader)):
        if batch_nr < limit:
            return_normal(images, processor, 4, True)
