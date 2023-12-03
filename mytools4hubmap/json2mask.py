import json, cv2, numpy as np, itertools, random, pandas as pd
# from pycocotools.coco import COCO
# from pycocotools import mask as maskUtils
# from skimage import io
# import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
import os, shutil
# from sklearn import model_selection

# import matplotlib.pyplot as plt
# from skimage import io
# from pycocotools.coco import COCO
# import matplotlib.patches as mpatches
from PIL import Image
from sklearn.model_selection import StratifiedKFold

def coordinates_to_masks(coordinates, shape):
    masks = []
    for coord in coordinates:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(coord)], 1)
        masks.append(mask)
    return masks

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def rle_to_binary_mask(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

## save colored mask using PIL P mode
def save_colored_mask(mask, save_path):
    bin_colormap = np.array([[255,255,255],[255,0,0],[0,0,255],[0,255,0]]).astype(np.uint8)
    mask = mask.astype(np.uint8)
    mask_p = Image.fromarray(mask, mode='P')
    mask_p.putpalette(bin_colormap)
    mask_p.save(save_path)
    # lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    # colormap = imgviz.label_colormap()
    # lbl_pil.putpalette(colormap.flatten())
    # lbl_pil.save(save_path)
## save grey mask using CV2
def save_grey_mask(mask, save_path):
    cv2.imwrite(save_path, mask)

## Loading Datase

df = pd.read_csv("../data/hubmap_data/tile_meta.csv")
# df1 = df.query('dataset == 1')
# df2 = df.query('dataset == 2')
# df3 = df.query('dataset == 3')

df = df.query('dataset != 3')
df.reset_index(inplace=True,drop=True)
# print(df.head())

## Spliting training & Valid


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# from sklearn.model_selection import train_test_split
# train_x,test_x,train_y,test_y = train_test_split(df,df['source_wsi'],test_size=0.2, random_state=42)
# train_x,val_x,train_y,val_y = train_test_split(train_x,train_y,test_size=0.25, random_state=17)

for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['source_wsi']), 1):
    df.loc[val_idx, 'fold'] = fold

df['fold'] = df['fold'].astype(np.uint8)
df.to_csv('../data/hubmap_data/tile_meta_5fold.csv')

selected_fold = 5
train_ids = df.query(f'fold != {selected_fold}')['id'].values.tolist()
valid_ids = df.query(f'fold == {selected_fold}')['id'].values.tolist()
image_path = '../data/hubmap_data/images_all'
train_path = '../data/hubmap_data/train'
valid_path = '../data/hubmap_data/valid'
# generate training dataset and valid dataset
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)

for id in tqdm(train_ids, desc='train files'):
    src_file = os.path.join(image_path, id + '.tif')
    dst_file = os.path.join(train_path, id + '.tif')
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)

for id in tqdm(valid_ids, desc='valid files'):
    src_file = os.path.join(image_path, id + '.tif')
    dst_file = os.path.join(valid_path, id + '.tif')
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)

## Reading polygons.jsonl
jsonl_file_path = "../data/hubmap_data/polygons.jsonl"
data = []
with open(jsonl_file_path, "r") as file:
    for line in file:
        data.append(json.loads(line))

## Cateogories
# categories_list=['blood_vessel','glomerulus'] ## 2 class
# categories_list=['blood_vessel'] ## 1 class
categories_list=['blood_vessel','glomerulus', 'unsure'] ## 3 class
#------------------------------------------------------------------------------
categories_ids = {name:id+1 for id, name in enumerate(categories_list)}
ids_categories = {id+1:name for id, name in enumerate(categories_list)}
categories =[{'id':id,'name':name} for name,id in categories_ids.items()]

print(categories_ids)
print(ids_categories)
print(categories)

## Creating COCO
def coco_structure(images_ids):
    idx = 1
    annotations = []
    images = []
    for item in tqdm(data, total=int(len(images_ids))):
        image_id = item["id"]
        if image_id in images_ids:
            image = {"id": image_id, "file_name": image_id + ".tif", "height": 512, "width": 512}
            images.append(image)
        else:
            continue
        # -----------------------------
        anns = item["annotations"]
        for an in anns:
            category_type = an["type"]
            # if category_type != "unsure":
            if category_type in categories_list:
                category_id = categories_ids[category_type]
                segmentation = an["coordinates"]
                mask_img = coordinates_to_masks(segmentation, (512, 512))[0]
                ys, xs = np.where(mask_img)
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                rle = binary_mask_to_rle(mask_img)

                seg = {
                    "id": idx,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)],
                    "area": int(np.sum(mask_img)),
                    "iscrowd": 0,
                }
                if image_id in images_ids:
                    annotations.append(seg)
                    idx = idx + 1

    return {"info": {}, "licenses": [], "categories": categories, "images": images, "annotations": annotations}

def label_gen(images_ids, labelroot_gray, labelroot_color):
    os.makedirs(labelroot_gray, exist_ok=True)
    os.makedirs(labelroot_color, exist_ok=True)

    images = []
    for item in tqdm(data, total=int(len(images_ids))):
        image_id = item["id"]
        if image_id in images_ids:
            image = {"id": image_id, "file_name": image_id + ".tif", "height": 512, "width": 512}
            images.append(image)
        else:
            continue
        # -----------------------------
        anns = item["annotations"]
        label_img = np.zeros((512,512), dtype = "uint8")
        for an in anns:
            category_type = an["type"]
            # if category_type != "unsure":
            if category_type in categories_list:
                category_id = categories_ids[category_type]
                segmentation = an["coordinates"]
                mask_img = coordinates_to_masks(segmentation, (512, 512))[0]
                mask_img = mask_img*category_id
                label_img = np.maximum(label_img, mask_img)
        label_filename = labelroot_gray + image["file_name"]
        save_grey_mask(label_img, label_filename)
        label_filename = labelroot_color + image["file_name"]
        save_colored_mask(label_img, label_filename)
    return 0

# train_label_root = "../data/hubmap_data/train_label/"
# color_root = "../data/hubmap_data/train_label_color/"
# train_coco_data = label_gen(train_ids, train_label_root, color_root)
val_label_root = "../data/hubmap_data/valid_label_3c/"
color_root = "../data/hubmap_data/valid_label_color_3c/"
valid_coco_data = label_gen(valid_ids, val_label_root, color_root)

