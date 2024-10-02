import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from model_functions import get_image_emb
from PIL import Image, ImageOps

class HMDataset2(Dataset):
    def __init__(self, articles_csv, image_dir, main_class, processor, model, transform=None):
        # Load the CSV files
        self.articles = pd.read_csv(articles_csv)

        # Image directory
        self.image_dir = image_dir # image folder
        self.processor = processor # prcessor of clip model

        self.transform = transforms.ToTensor()

        self.model = model

        self.main_class = main_class #for example prod_name
        self.id_add = 0 # take next if image does not exist
        self.len = self.articles.shape[0]

        self.main_classes = self.articles.columns # all classes
        self.sub_classes = list(self.articles[self.main_class].unique()) # list of all subclasses
        self.count_sub_classes = self.articles[self.main_class].value_counts() # counts in subclasses

        print('Max uniform size:', self.articles[self.main_class].value_counts().min()) # <max uniform size

        self.class_to_id = {name: i for i, name in enumerate(self.sub_classes)}

        self.max_counts = 4 # max number of data per class
        self.counts = {name: 0 for name in self.sub_classes} # number of samples each class

        self.processor.feature_extractor.do_rescale = False # # make sure image values: False=> [0-1] and True=> [0,255]

    def __len__(self):
        return self.len

    def get_n_of_each(self, max_counts):
        """Collects max_counts datapoints from each subclass in large dataset"""
        self.max_counts = max_counts
        all_embeds = []
        all_labels = []
        all_images = []

        for idx in range(self.len):
          id = self.articles['article_id'][idx]
          subclass_name = self.articles[self.main_class][idx]

          if self.counts[subclass_name] < self.max_counts:
              image_path = f"{self.image_dir}/0{str(id)[0:2]}/0{id}.jpg"

              try:
                  image = Image.open(image_path)

                  # get border color
                  # only gets the first pixel, but it's good enough
                  im_matrix = np.array(image)
                  r,g,b = im_matrix[0][0]

                  # rezise and add padding
                  image.thumbnail((200, 200), Image.LANCZOS)
                  padding = (200 - image.size[0], 200 - image.size[1])
                  image = ImageOps.expand(image, (padding[0]//2, padding[1]//2, (padding[0]+1)//2, (padding[1]+1)//2), fill=(r,g,b))

                  image_tensor = self.transform(image)

                  with torch.no_grad():
                      image_embeds, processed_images = get_image_emb(self.model, self.processor, image_tensor)

                  self.counts[subclass_name]+=1
                  all_embeds.append(image_embeds)
                  all_labels.append(subclass_name)
                  all_images.append(processed_images)
              except FileNotFoundError:
                  print(f"Image for article {id} not found. Takes next")

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