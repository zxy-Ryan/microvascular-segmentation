import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# img_h, img_w = 32, 32
# img_h, img_w = 32, 48  #
means, stdevs = [], []
img_list = []

imgs_path = "../data/hubmap_data/train"

def calculate_rgb_mean_and_std(imgs_path):
    image_files = [os.path.join(imgs_path, img) for img in os.listdir(imgs_path) if img.endswith(".tif")]
    num_images = len(image_files)
    mean = np.zeros(3)
    std = np.zeros(3)
    # mean = np.array([212.95, 186.99, 212.64])
    # std = np.array([22.18, 29.42, 18.97])
    for image_file in tqdm(image_files, desc='images'):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mean += np.mean(image, axis=(0, 1))
        std += np.std(image, axis=(0, 1))


    mean /= num_images
    std /= num_images

    return mean, std

mean, std = calculate_rgb_mean_and_std(imgs_path)
print("RGB mean", mean)
print("RGB std", std)


