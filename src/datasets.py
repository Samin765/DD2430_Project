import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from model_functions import get_image_emb
from PIL import Image, ImageOps
import random
from collections import Counter
import os
from torch.utils.data import DataLoader




class HMDataset(Dataset):
    def __init__(self, articles_csv, image_dir, main_class, processor, model, transform=None):
        # Load the CSV files
        self.articles = pd.read_csv(articles_csv)

        # Image directory
        self.image_dir = image_dir  # image folder
        self.processor = processor  # prcessor of clip model

        self.transform = transforms.ToTensor()

        self.model = model

        self.main_class = main_class  # for example prod_name
        self.id_add = 0  # take next if image does not exist
        self.len = self.articles.shape[0]

        self.main_classes = self.articles.columns  # all classes
        # list of all subclasses
        self.sub_classes = list(self.articles[self.main_class].unique())
        # counts in subclasses
        self.count_sub_classes = self.articles[self.main_class].value_counts()

        # <max uniform size
        print('Max uniform size:',
              self.articles[self.main_class].value_counts().min())

        self.class_to_id = {name: i for i, name in enumerate(self.sub_classes)}

        self.max_counts = 4  # max number of data per class
        # number of samples each class
        self.counts = {name: 0 for name in self.sub_classes}

        # make sure image values: False=> [0-1] and True=> [0,255]
        self.processor.feature_extractor.do_rescale = False

        self.pcodes = set()  # to not have doubles

    def __len__(self):
        return self.len

    def get_n_of_each(self, max_counts, allow_duplicates=False):
        """Collects max_counts datapoints from each subclass in large dataset"""
        self.max_counts = max_counts
        all_embeds = []
        all_labels = []
        all_images = []

        for idx in range(self.len):
            id = self.articles['article_id'][idx]
            subclass_name = self.articles[self.main_class][idx]

            # BOOL balanced sets
            not_filled = self.counts[subclass_name] < self.max_counts
            p_code = self.articles['product_code'][idx]
            duplicates = p_code in self.pcodes  # BOOL same cloathing shape
            if not_filled:
                if not duplicates or allow_duplicates:
                    self.pcodes.add(p_code)
                    image_path = f"{self.image_dir}/0{str(id)[0:2]}/0{id}.jpg"

                    try:
                        image = Image.open(image_path)
                        image_tensor = self.transform(image)

                        with torch.no_grad():
                            image_embeds, processed_images = get_image_emb(
                                self.model, self.processor, image_tensor)

                        self.counts[subclass_name] += 1
                        all_embeds.append(image_embeds)
                        all_labels.append(subclass_name)
                        all_images.append(processed_images)
                    except FileNotFoundError:
                        pass
                        #print(f"Image for article {id} not found. Takes next")

        return torch.cat(all_embeds), all_labels, torch.cat(all_images)


class UniformHMDataset(Dataset):
    """Dataset with perfect class balance"""

    def __init__(self, emb, labels, image):
        self.emb = emb
        self.labels = labels
        self.image = image
        self.classes = list(set(labels))
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emb[idx], self.labels[idx], self.image[idx]

    
def create_dataset(n_samples, main_class, subclasses, clip, path, device, allow_duplicates=False, exclude=True, show=True):
    """Create balanced dataset, if exclude it includes only the given subclasses, else excludes"""
    dataset = HMDataset(
        articles_csv = path['hm'] + 'articles.csv',
        image_dir = path['hm']+ 'images',
        main_class = main_class,
        model = clip['m'].to(device),
        processor = clip['p'])
    assert n_samples >=10, 'Must be have more than 10 for val splits'
    assert dataset.articles[dataset.main_class].value_counts().min()>=n_samples, 'Can not make balanced set'
    if exclude:
        for exclude_subclass in subclasses:
            dataset.counts[exclude_subclass]=n_samples
    else:
        for exclude_subclass in dataset.sub_classes:
            dataset.counts[exclude_subclass]=n_samples
        for include_subclass in subclasses:
            dataset.counts[include_subclass]=0
    image_emb, labels, images = dataset.get_n_of_each(n_samples, allow_duplicates)
    data_to_save = {
        'image_embedding': image_emb,
        'class_text': labels,
        'images': images}
    os.makedirs(path['save'], exist_ok=True)
    if show:
        print(Counter(labels))
    torch.save(data_to_save, f"{path['save']}HM_data_{n_samples}_{main_class}_{len(subclasses)}.pth")
    return dataset # to get all classes
        
               
def load_dataset(n_samples, main_class, len_subclasses, path):        
    loaded_data = torch.load( f"{path['save']}HM_data_{n_samples}_{main_class}_{len_subclasses}.pth",
                            weights_only=True)
    return loaded_data

def generate_train_test_val(labels, image_emb, images, batch_size, n_samples, set_sizes, show = True):
    """Generate train_test_val sets that are balanced""" 
    dataset, dataset_train, dataset_test, dataset_val = split(labels, image_emb, images, n_samples, set_sizes, show = show)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return {'train':dataloader_train, 'val':dataloader_val, 'test':dataloader_test}


def split(labels0, image_emb0, images0, n_samples, set_sizes, show = True):
    """Given trainingdata splits it into train/val/test"""
    combined = sorted(zip(labels0, image_emb0, images0), key=lambda x: x[0])
    labels, image_emb, images = zip(*combined)

    train_labels, train_image_emb, train_images = [], [], []
    test_labels, test_image_emb, test_images = [], [], []
    val_labels, val_image_emb, val_images = [], [], []

    for i in range(0, len(combined) - 1, n_samples):
        labels_sub = labels[i: i + n_samples]
        image_emb_sub = image_emb[i: i + n_samples]
        images_sub = images[i: i + n_samples]

        def s(t): return int(float(len(labels_sub)) * set_sizes[t])

        train_labels.extend(labels_sub[:s("train")])
        train_image_emb.extend(image_emb_sub[:s("train")])
        train_images.extend(images_sub[:s("train")])

        test_labels.extend(labels_sub[s("train"):s("train") + s("test")])
        test_image_emb.extend(image_emb_sub[s("train"):s("train") + s("test")])
        test_images.extend(images_sub[s("train"):s("train") + s("test")])

        val_labels.extend(labels_sub[s("train") + s("test"):])
        val_image_emb.extend(image_emb_sub[s("train") + s("test"):])
        val_images.extend(images_sub[s("train") + s("test"):])

    # shuffle the data in each set
    def shuffle_set(labels, image_emb, images):
        combined = list(zip(labels, image_emb, images))
        random.shuffle(combined)
        return zip(*combined)

    train_labels, train_image_emb, train_images = shuffle_set(
        train_labels, train_image_emb, train_images)
    test_labels, test_image_emb, test_images = shuffle_set(
        test_labels, test_image_emb, test_images)
    val_labels, val_image_emb, val_images = shuffle_set(
        val_labels, val_image_emb, val_images)

    # create the datasets
    dataset = UniformHMDataset(labels, image_emb, images)
    dataset_train = UniformHMDataset(
        train_image_emb, train_labels, train_images)
    dataset_test = UniformHMDataset(test_image_emb, test_labels, test_images)
    dataset_val = UniformHMDataset(val_image_emb, val_labels, val_images)

    # checking
    if show:
        print(len(labels), len(dataset_train.labels), len(
            dataset_test.labels), len(dataset_val.labels))
        # check class-balance of splits
        for labels_ in [labels, dataset_train.labels, dataset_val.labels, dataset_test.labels]:
            print(Counter(labels_))
            
    return dataset, dataset_train, dataset_test, dataset_val
