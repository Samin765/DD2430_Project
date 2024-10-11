import os
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
import shutil


# ./HM_DATA_PATH/images/..., ./HM_DATA_PATH/articles.csv, etc
HM_DATA_PATH = "./data/"

# path where new dataset will be saved to
HM_DATA_PATH_NEW = "./dataset/"

SRC_IMAGE_DIR = HM_DATA_PATH + "images"
DEST_IMAGE_DIR = HM_DATA_PATH_NEW + "images"

def create_folders():
    if not os.path.exists(DEST_IMAGE_DIR):
        os.makedirs(DEST_IMAGE_DIR)


def copy_csv_json_files(all_files=False):
    # only copy articles.csv
    shutil.copy2(HM_DATA_PATH + "articles.csv", HM_DATA_PATH_NEW + "articles.csv")

    # only copy over .csv and .json
    if all_files:
        for file in os.listdir(HM_DATA_PATH):
            if file.endswith(".csv") or file.endswith(".json"):
                src = HM_DATA_PATH + file
                dest = HM_DATA_PATH_NEW + file

                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)


def images_processing():
    # go through all images
    for root, _, files in os.walk(SRC_IMAGE_DIR):
        for file in tqdm(files):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)

                    # when this works it's good. Otherwise it's bad.

                    # save rgb of all pixels of border of image
                    # np_image = np.array(img)

                    # top_border = np_image[1, 1:-1, :]
                    # bottom_border = np_image[-2, 1:-1, :]
                    # left_border = np_image[1:-1, 1, :]
                    # right_border = np_image[1:-1, -2, :]

                    # border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border], axis=0)

                    # def get_brightest_or_closest_to_white(border_pixels):
                    #     def distance_to_white(pixel):
                    #         r, g, b = pixel
                    #         return np.sqrt((255 - r) ** 2 + (255 - g) ** 2 + (255 - b) ** 2)
                    
                    #     # Find the pixel with the highest brightness (or closest to white)
                    #     brightest_pixel = min(border_pixels, key=distance_to_white)
                    #     return tuple(brightest_pixel)

                    # Choose the brightest or closest to white color for padding
                    # padding_color = get_brightest_or_closest_to_white(border_pixels)


                    # just pick RGB of background manually and assign to every image.
                    padding_color = (236, 235, 233)
                    r,g,b = padding_color

                    # rezise and add padding
                    square_width_height = 224 # transformer take 224x224 image res
                    img.thumbnail((square_width_height, square_width_height), Image.LANCZOS)
                    padding = (square_width_height - img.size[0], square_width_height - img.size[1])
                    img = ImageOps.expand(img, (padding[0]//2, padding[1]//2, (padding[0]+1)//2, (padding[1]+1)//2), fill=(r,g,b))

                    # save new image
                    output_path = os.path.join(DEST_IMAGE_DIR, file)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    img.save(output_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    create_folders()
    copy_csv_json_files()
    images_processing()
