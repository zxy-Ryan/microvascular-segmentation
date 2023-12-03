from itertools import groupby
from pycocotools import mask as mutils
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import cv2,glob
import matplotlib.pyplot as plt
# import wandb
from PIL import Image
import gc
sample = None
import mmcv
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
# from mmdet.apis import init_detector, inference_detector

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import myseg as mmdet
print(mmdet.__version__)

from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())
CUDA_NUMBER = os.getenv("CUDA_VISIBLE_DEVICES")
print(CUDA_NUMBER)
os.environ["CUDA_VISIBLE_DEVICES"]='1'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib

def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str

def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

def get_filtered_masks(pred):
    """
    filter masks using MIN_SCORE for mask and MAX_THRESHOLD for pixels
    """
    use_masks = []
    use_labels = []
    for i, mask in enumerate(pred["masks"]):
        # Filter-out low-scoring results. Not tried yet.
        scr = pred["scores"][i].cpu().item()
        label = pred["labels"][i].cpu().item()
        if scr > min_score_dict[label]:
            mask = mask.cpu().numpy().squeeze()
            # Keep only highly likely pixels
            binary_mask = mask > mask_threshold_dict[label]
            binary_mask = remove_overlapping_pixels(binary_mask, use_masks)
            use_masks.append(binary_mask)
            use_labels.append(label)

    return use_masks,use_labels


import os
import numpy as np
import torch
from PIL import Image

def visualize_segmentation(img_array, pred_img, gt_img, colormap, alpha, outfile):
    original_img = img_array
    segmentation_img = pred_img

    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds = range(len(colormap) + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    ax1 = plt.subplot(1, 3, 2)
    plt.imshow(original_img)
    segm_fig = plt.imshow(gt_img, cmap=cmap, norm=norm, alpha=alpha)
    plt.title('Ground Truth')
    plt.axis('off')

    ax2 = plt.subplot(1, 3, 3)
    plt.imshow(original_img)
    plt.imshow(segmentation_img, cmap=cmap, norm=norm, alpha=alpha) # 使用自定义颜色映射和透明度
    plt.title('Prediction')
    plt.axis('off')

    cbar = plt.colorbar(segm_fig, ax=[ax1, ax2], orientation='vertical', ticks=bounds, fraction=0.01)
    cbar.set_ticklabels(['Class ' + str(i) for i in bounds])

    plt.savefig(outfile)



    # plt.show()

class HuBMAPDataset(torch.utils.data.Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
        self.name_indices = [os.path.splitext(os.path.basename(i))[0] for i in imgs]

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        name = self.name_indices[idx]
        array = tiff.imread(img_path)
        img = Image.fromarray(array)
        return img, name

    def __len__(self):
        return len(self.imgs)


all_imgs = glob('../data/hubmap_data/valid/*.tif')
ann_imgs = glob('../data/hubmap_data/valid_label_3c/*.tif')
dataset_test = HuBMAPDataset(all_imgs)
test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=os.cpu_count(),
                                      pin_memory=True)

from myseg.apis import inference_model, init_model, show_result_pyplot
from mmengine.config import Config
fileroot = '../result/mmseg_result/hubmap/deeplabv3/'
outPath = '../result/mmseg_result/hubmap/deeplabv3/inference/'

config_file = fileroot + 'deeplabv3_r50-d8_4xb4-hubmap-512x512.py'
checkpoint_file = fileroot + 'iter_31000.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config.fromfile(config_file)
model = init_model(config_file, checkpoint_file, device=device)

# cell type specific thresholds
cell_type_dict = {'blood_vessel': 1, 'glomerulus': 2, 'unsure': 3}

mask_threshold_dict = {0: 0.4, 1: 0.80, 2:  0.80}
min_score_dict = {0: 0.6, 1: 0.80, 2: 0.80}

ids = []
heights = []
widths = []
prediction_strings = []
colormap = ['white', 'red', 'blue', 'green', 'yellow']
for img, ann in tqdm(zip(all_imgs, ann_imgs), desc='image files'):
    img_array = mmcv.imread(img,channel_order='rgb')
    ann_array = mmcv.imread(ann, flag='unchanged')
    [h, w, c] = img_array.shape
    pred = inference_model(model,img)
    pred_img = np.squeeze(pred.pred_sem_seg.data.cpu().numpy())
    outfile = outPath + img.split('/')[-1]
    visualize_segmentation(img_array, pred_img, ann_array, colormap, 0.5, outfile)
    # show_result_pyplot(
    #     model,
    #     img,
    #     pred,
    #     title='test result',
    #     opacity=0.5,
    #     draw_gt=False,
    #     show=False,
    #     out_file=outPath+img.split('/')[-1])

    # show_result_pyplot(
    #     model,
    #     img,
    #     ann_array,
    #     title='test label',
    #     opacity=0.5,
    #     draw_gt=False,
    #     show=False,
    #     out_file=outPath + 'label_'+img.split('/')[-1])

#     previous_masks = []
#     masks_use = []
#     labels_use = []
#     pred_string=""
#     for i, mask in enumerate(pred.pred_instances["masks"]):
#         # Filter-out low-scoring results.
#         score = pred.pred_instances["scores"][i].cpu().item()
#         label = pred.pred_instances["labels"][i].cpu().item()
#         if score > min_score_dict[label]:
#             mask = mask.cpu().numpy()
#             # Keep only highly likely pixels
#             binary_mask = mask > mask_threshold_dict[label]
#             binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
#             masks_use.append(binary_mask)
#             labels_use.append(label)
#             previous_masks.append(binary_mask)
#             encoded = encode_binary_mask(binary_mask)
#             #if label != 0: continue
#             if i == 0:
#                 pred_string += f"{int(label)} {score} {encoded.decode('utf-8')}"
#             else:
#                 pred_string += f" {int(label)} {score} {encoded.decode('utf-8')}"
#             #print(pred_classes[i])
#     ids.append(str(img).split('.')[0].split('/')[-1])
#     heights.append(h)
#     widths.append(w)
#     prediction_strings.append(pred_string)
#
# colors = [ 'Set1', 'Set3']
# legend = {0: 'blood_vessel',1: 'glomerulus'}
# # colors = ['Set1']
# # legend = {0: 'blood_vessel'}
# from skimage import io
# import matplotlib.patches as mpatches
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# I = io.imread(str(all_imgs[0]))
# axs[0].imshow(I)
# axs[0].set_title('Image')
# axs[1].imshow(I)
# pred = inference_detector(model,img)
# previous_masks = []
# for i, mask in enumerate(pred.pred_instances["masks"]):
#     # Filter-out low-scoring results.
#     score = pred.pred_instances["scores"][i].cpu().item()
#     label = pred.pred_instances["labels"][i].cpu().item()
#     if score > min_score_dict[label]:
#         mk = mask.cpu().numpy()
#         # Keep only highly likely pixels
#         binary_mask = mk > mask_threshold_dict[label]
#         binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
#         previous_masks.append(binary_mask)
#         color = colors[label]
#         mask = np.ma.masked_where(mk == 0, mk)
#         axs[1].imshow(mask, cmap=color, alpha=0.8)
#         axs[1].set_title('Predicted Masks')
#         # Add score text on each segment
#         y, x = np.where(mk > 0)
#         text_x, text_y = np.min(x), np.min(y)
#         axs[1].text(text_x, text_y, f"{score:.2f}", color='white', fontsize=8)
#         handles = []
#         for cl in legend:
#             color = colors[cl]
#             handles.append(mpatches.Patch(color=plt.colormaps.get_cmap(color)(0)))
#         axs[1].legend(handles, legend.values(), bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.savefig(fileroot+'testresult.tif')
#
# submission = pd.DataFrame()
# submission['id'] = ids
# submission['height'] = heights
# submission['width'] = widths
# submission['prediction_string'] = prediction_strings
# submission = submission.set_index('id')
# submission.to_csv(fileroot+"submission.csv")
# submission.head()