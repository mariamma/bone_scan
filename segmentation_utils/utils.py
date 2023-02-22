import cv2
import math
import numpy as np
import os
from os.path import join, isfile
import pandas as pd
import torch
import torch.nn as nn

from . import base

def convert_to_binary_mask(mask):
    r,h,c = mask.shape
    assert(c == 3)
    none_color = [0,0,0]
    equality = np.equal(mask, none_color)
    class_map = np.all(equality, axis = -1) # height x width boolean array with val at i j = 1 iff mask rgb at i j == none_color
    semantic_map = []
    semantic_map.append((1 - class_map) * 255)
    semantic_map.append((1 - class_map) * 255)
    semantic_map.append((1 - class_map) * 255)
    semantic_map = np.stack(semantic_map, axis=-1) # height x width x 3 array with rgb values = 000 for none class, 255,255,255 for all others
    return semantic_map


def get_unequal_pairs(meta_data_file, image_dir):
    IMG_DIR = '/data/rachit/mtl_train/tuberculosis/tuberculosis_model/dr_annotations/original_images/'
    LBL_DIR = '/data/rachit/mtl_train/tuberculosis/tuberculosis_model/dr_annotations/segmentation_labels/'
    md = pd.read_csv(meta_data_file)
    i = 0
    for index, row in md.iterrows():
        # print(row)
        if i == 0:
            i += 1
            continue
        img_x = cv2.cvtColor(cv2.imread(IMG_DIR + row['image_name']), cv2.COLOR_BGR2RGB)
        img_y = cv2.cvtColor(cv2.imread(LBL_DIR + row['ground_truth_image_name']), cv2.COLOR_BGR2RGB)
        if (img_x.shape != img_y.shape):
            print(row['image_name'], img_x.shape, img_y.shape)
        
        if (img_x.shape[0] != img_x.shape[1]):
            print(row['image_name'], img_x.shape, img_y.shape)


def create_colored_masks(class_color_map, annotation_dir, save_only_mask_dir):
    for f in os.listdir(annotation_dir): #traversed in arbitrary order
        if (isfile(join(annotation_dir, f))):
            img = cv2.cvtColor(cv2.imread(join(annotation_dir, f)), cv2.COLOR_BGR2RGB)
            height = img.shape[0]
            width = img.shape[1]
            only_segmentation_mask = 0 * np.ones(img.shape).astype(np.uint8)
            
            for i in range(height):
                for j in range(width):
                    min_magnitude = 255*3 #>sqrt(3 * 255**2)
                    for color in class_color_map.values():
                        diff = img[i,j,:].astype(int) - color.astype(int)
                        magnitude = math.sqrt(sum(pow(element, 2) for element in diff))
                        if (magnitude < min_magnitude):
                            min_magnitude = magnitude
                            final_color = color
                    if (min_magnitude <= 20):
                        only_segmentation_mask[i,j,:] = final_color
                    
            cv2.imwrite(join(save_only_mask_dir, f), cv2.cvtColor(only_segmentation_mask, cv2.COLOR_RGB2BGR))



def create_metadata_file(meta_data_file, image_dir):
    with open(meta_data_file, 'w') as mf:
        for f in os.listdir(image_dir): #traversed in arbitrary order
            if (isfile(join(image_dir, f))):
                mf.write(f + '\n')


def get_class_frequencies(one_hot_mask_list, class_rgb_values):
    for mask in one_hot_mask_list:
        np.sum(mask, axis = 0)

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None, channel_weights=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    num_channels = gt.shape[1] # m x c x h x w
    combined_channel_score = 0
    print('__', gt.shape, pr.shape)
    if channel_weights is None:
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp

        combined_channel_score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    else:
        for c in range(num_channels):
            tp = torch.sum(gt[:,c,:,:] * pr[:,c,:,:])
            fp = torch.sum(pr[:,c,:,:]) - tp
            fn = torch.sum(gt[:,c,:,:]) - tp

            score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
            combined_channel_score += channel_weights[c] * score
        combined_channel_score /= np.sum(channel_weights)

    return combined_channel_score

class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)



class DiceLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, channel_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.channel_weights = channel_weights

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            channel_weights = self.channel_weights
        )


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def get_channel_weights(img_dir, class_rgb_values, num_images):
    num_channels = len(class_rgb_values)
    class_pixel_counts = {}
    channel_weights = {}
    for c in range(num_channels):
        class_pixel_counts[c] = 0

    count = 0
    total = 0
    for f in os.listdir(img_dir): #traversed in arbitrary order
        if (isfile(join(img_dir, f))):
            if (count >= num_images):
                break
            
            mask = cv2.cvtColor(cv2.imread(join(img_dir, f)), cv2.COLOR_BGR2RGB)
            # print(mask.shape)
            # print(mask[10:20, 10:20, :])
            # one-hot-encode the mask
            mask = one_hot_encode(mask, list(class_rgb_values.values())).astype('float')
            # print(f)
            # print(mask[10, 10, :])
            # print(mask[10:20, 10:20, 8])
            for c in range(num_channels):
                class_pixel_counts[c] += np.sum(mask[:,:,c])
                # if (c == 8):
                    # print(np.sum(mask[:,:,c]))
            
            count += 1
    
    # print(class_pixel_counts)
            
    for c in range(num_channels):
        total += class_pixel_counts[c]
        channel_weights[c] = 0

    for c in range(num_channels):
        channel_weights[c] = total/class_pixel_counts[c]
        print(f'{c}, {channel_weights[c]}')

    

            
            