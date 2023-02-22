import os, cv2, math
from pickletools import uint8
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from os.path import join, isfile
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

import segmentation_models_pytorch as smp
from torchsummary import summary
from torch.autograd import Variable

import segmentation_utils.utils as utils

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# helper function for data visualization
def visualize(figname, **images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()
    plt.savefig(figname)

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
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


class XraySegmentationDataset(torch.utils.data.Dataset):

    """Xray Segmentation Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    def __init__(
            self, 
            df,
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,
            binary = False
    ):
        self.image_paths = df['image_path'].tolist()
        self.mask_paths = df['label_colored_path'].tolist()
        
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.binary = binary
    
    def __getitem__(self, i):
        
        image_name = self.image_paths[i].split('/')[-1]
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        # convert mask to binary for none-vs abnormality segmentation
        if self.binary:
            mask = utils.convert_to_binary_mask(mask)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        # print('___ ', np.sum(mask), mask.shape)
        # assert(np.sum(mask) == mask.shape[0] * mask.shape[1])
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # return image, mask, image_name
        return image, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

def get_training_augmentation():
    train_transform = [
        album.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.5,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor_for_mask(x, **kwargs):
    #set class to 'none' (last channel) if no class is assigned
    x[:,:,x.shape[-1] - 1] += 1. - np.sum(x, axis = -1)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(img_size, preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    _transform.append(album.augmentations.geometric.resize.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST, always_apply=False, p=1) )
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor_for_mask))

    return album.Compose(_transform)

# Center crop padded image / mask to original image dims
def crop_image(image, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[1], width=true_dimensions[2])(image=image)


def process_metadata_file(metadata_df, IMAGE_DIR, SEGMENTATION_MASK_DIR):
    metadata_df.insert(2, 'image_path', metadata_df['image_name'], allow_duplicates=True)
    # metadata_df['image_path'] = metadata_df[['image_name']]
    metadata_df['image_path'] = metadata_df['image_path'].apply(lambda img_pth: os.path.join(IMAGE_DIR, img_pth))
    print(metadata_df.columns)
    metadata_df.insert(3, 'label_colored_path', metadata_df['ground_truth_image_name'], True)
    # metadata_df['label_colored_path'] = metadata_df[['ground_truth_image_name']]
    metadata_df['label_colored_path'] = metadata_df['label_colored_path'].apply(lambda img_pth: os.path.join(SEGMENTATION_MASK_DIR, img_pth))
    print(metadata_df.head())
    return metadata_df


def train(num_epochs, batch_size, img_size, class_weights=None, binary=False, training=False):

    DATA_DIR = '/scratch/mariamma/vinbig/dataset/vinbig_processed/train/'
    IMAGE_DIR = DATA_DIR + 'images' #+ 'original_images_augmented'
    SEGMENTATION_MASK_DIR = DATA_DIR + 'mask' # + 'segmentation_labels_augmented'
    VALID_DATA_DIR = '/scratch/mariamma/vinbig/dataset/vinbig_processed/val/'
    VALID_IMAGE_DIR = VALID_DATA_DIR + 'images' #+ 'original_images_augmented'
    VALID_SEGMENTATION_MASK_DIR = VALID_DATA_DIR + 'mask' # + 'segmentation_labels_augmented'
    
    SEGMENTATION_UTILS_DIR = '/scratch/mariamma/vinbig/dataset/segmentation_utils'
    model_path = f'/scratch/mariamma/vinbig/models/pspnet-resnext50-pytorch_sample/best_model_binary{binary}_weighted_v3_b{batch_size}_d{img_size}.pth'
    BEST_MODEL_PATH = model_path #os.path.join(SEGMENTATION_UTILS_DIR, model_path)
    class_rgb_values_file = 'semantic-class-labels2.csv'
    train_metadata_file = 'train_metadata.csv'
    valid_metadata_file = 'valid_metadata.csv'
    # valid_fraction = 0.15

    if not training:
        DATA_DIR = '/scratch/mariamma/vinbig/dataset/vinbig_processed/test/'
        test_metadata_file = 'test_metadata.csv'
        IMAGE_DIR = DATA_DIR + 'images'
        SEGMENTATION_MASK_DIR = DATA_DIR + 'mask'
        # valid_fraction = 0.5 #used as test set
        test_metadata_df = pd.read_csv(os.path.join(SEGMENTATION_UTILS_DIR, test_metadata_file))
        test_df = process_metadata_file(test_metadata_df, IMAGE_DIR, SEGMENTATION_MASK_DIR)

    if binary:
        class_rgb_values_file = 'binary-semantic-class-labels.csv'

    train_metadata_df = pd.read_csv(os.path.join(SEGMENTATION_UTILS_DIR, train_metadata_file))
    # metadata_df = metadata_df[['image_name']]
    # metadata_df.insert(2, 'image_path', metadata_df[['image_name']], True)
    # # metadata_df['image_path'] = metadata_df[['image_name']]
    # metadata_df['image_path'] = metadata_df['image_path'].apply(lambda img_pth: os.path.join(IMAGE_DIR, img_pth))
    # # print(metadata_df.columns)
    # metadata_df.insert(3, 'label_colored_path', metadata_df[['ground_truth_image_name']], True)
    # # metadata_df['label_colored_path'] = metadata_df[['ground_truth_image_name']]
    # metadata_df['label_colored_path'] = metadata_df['label_colored_path'].apply(lambda img_pth: os.path.join(SEGMENTATION_MASK_DIR, img_pth))
    # Shuffle DataFrame
    train_df = process_metadata_file(train_metadata_df, IMAGE_DIR, SEGMENTATION_MASK_DIR)
    
    valid_metadata_df = pd.read_csv(os.path.join(SEGMENTATION_UTILS_DIR, valid_metadata_file))
    valid_df = process_metadata_file(valid_metadata_df, VALID_IMAGE_DIR, VALID_SEGMENTATION_MASK_DIR)
    # metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    # Perform 90/10 split for train / val
    # valid_df = metadata_df.sample(frac=valid_fraction, random_state=42)
    # train_df = metadata_df.drop(valid_df.index)
    print('Dataframe sizes : ', len(train_df), len(valid_df))

    class_dict = pd.read_csv(os.path.join(SEGMENTATION_UTILS_DIR, class_rgb_values_file))
    # Get class names
    class_names = class_dict['class_name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)
    
    dataset = XraySegmentationDataset(train_df, class_rgb_values=class_rgb_values, binary = binary)
    # random_idx = random.randint(0, len(dataset)-1)
    
    # image, mask = dataset[4]
    # print(image.shape)
    # print(mask.shape)
    # print(image[512,250:300,:])
    # print(mask[512,250:300,:])
    # visualize(
    #     'segmentation_utils/temp1.png',
    #     original_image = image,
    #     ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), class_rgb_values),
    #     one_hot_encoded_mask = reverse_one_hot(mask)
    # )

    # augmented_dataset = XraySegmentationDataset(
    #     train_df, 
    #     # augmentation=get_training_augmentation(),
    #     class_rgb_values=class_rgb_values,
    #     binary=binary
    # )

    # Different augmentations on image/mask pairs
    # for idx in range(3):
    #     image, mask = augmented_dataset[idx]
    #     visualize(
    #         f'segmentation_utils/check_{idx}.png',
    #         original_image = image,
    #         ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), class_rgb_values),
    #         one_hot_encoded_mask = reverse_one_hot(mask)
    #     )
    

    ENCODER = 'resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = class_names
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.PSPNet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Get train and val dataset instances
    train_dataset = XraySegmentationDataset(
        train_df, 
        # augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(img_size, preprocessing_fn),
        class_rgb_values=class_rgb_values,
        binary = binary
    )
    

    valid_dataset = XraySegmentationDataset(
        valid_df, 
        # augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(img_size, preprocessing_fn),
        class_rgb_values=class_rgb_values,
        binary = binary
    )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    TRAINING = training

    # Set num of epochs
    EPOCHS = num_epochs

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function, ignore channel for class 'none'
    # loss = smp.utils.losses.DiceLoss(eps = 1e-7, beta = 1.0, ignore_channels=[len(class_rgb_values) - 1])
    loss = utils.DiceLoss(eps = 1e-7, beta = 1.0, channel_weights = class_weights)
    # loss = smp.losses.dice.DiceLoss(mode = 'multiclass', from_logits = True)

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # load best saved model checkpoint from previous commit (if present)
    if os.path.exists(BEST_MODEL_PATH):
        model = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    model.train()
    if TRAINING:
        # summary(model, (3, 512, 512))
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, BEST_MODEL_PATH)
                print('Model saved!')
    else:
        # load best saved model checkpoint from previous commit (if present)
        if os.path.exists(BEST_MODEL_PATH):
            best_model = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            print('Loaded PSPNet model from a previous commit.')

        best_model.eval()
        # create test dataloader (with preprocessing operation: to_tensor(...))
        test_dataset = XraySegmentationDataset(
            test_df, 
            # augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(img_size, preprocessing_fn),
            class_rgb_values=class_rgb_values,
            binary = binary
        )

        print("Created test_dataset")
        test_dataloader = DataLoader(test_dataset)

        # test dataset for visualization (without preprocessing augmentations & transformations)
        test_dataset_vis = XraySegmentationDataset(
            test_df,
            class_rgb_values=class_rgb_values,
            # preprocessing=get_preprocessing(img_size, preprocessing_fn),
            binary = binary
        )

        print("Created test_dataset_vis")
        # get a random test image/mask index
        # random_idx = random.randint(0, len(test_dataset_vis)-1)
        # image, mask = test_dataset_vis[random_idx]

        # visualize(
        #     'segmentation_utils/test_dataset2.png',
        #     original_image = image,
        #     ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), class_rgb_values),
        #     one_hot_encoded_mask = reverse_one_hot(mask)
        # )

        sample_preds_folder = '/scratch/mariamma/vinbig/dataset/segmentation_utils/sample_predictions/'
        if not os.path.exists(sample_preds_folder):
            os.makedirs(sample_preds_folder)

        iou = smp.utils.metrics.IoU(threshold=0.5)
        total = 0
        count = 0
        for idx in range(len(test_dataset)):
            image, gt_mask, image_name = test_dataset[idx] #(3, size, size) because of preprocessing
            print(image_name)
            image_vis = test_dataset_vis[idx][0].astype('uint8') #(size size, 3)
            true_dimensions = image_vis.shape
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            # Predict test image
            pred_mask = best_model(x_tensor)
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
            # Convert pred_mask from `CHW` format to `HWC` format
            pred_mask = np.transpose(pred_mask,(1,2,0))
            
            # Get prediction channel corresponding to foreground
            # pred_foreground_heatmap = crop_image(pred_mask[:,:,class_names.index('none')], true_dimensions)['image']
            # pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), class_rgb_values), true_dimensions)['image']
            # Convert gt_mask from `CHW` format to `HWC` format
            gt_mask = np.transpose(gt_mask,(1,2,0))
            
            # gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), class_rgb_values), true_dimensions)['image']
            # image_vis = np.transpose(image_vis,(1,2,0))
            # print('Here : ', image_vis.shape)
            # print(gt_mask.shape)
            # print(pred_mask.shape)
            # cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])
            
            # visualize(
            #     f'segmentation_utils/test_prediction_{idx}.png',
            #     original_image = image_vis,
            #     ground_truth_mask = gt_mask,
            #     predicted_mask = pred_mask,
            #     pred_foreground_heatmap = pred_foreground_heatmap
            # )
            # break; check ground truth if boolean; none vs abnormality; normalize input
            visualize(
                f'/scratch/mariamma/vinbig/dataset/segmentation_utils/visualize2/test_binary{binary}-{image_name}',
                original_image = image_vis,
                ground_truth_mask = colour_code_segmentation(reverse_one_hot(gt_mask), class_rgb_values),
                predicted_mask = colour_code_segmentation(reverse_one_hot(pred_mask), class_rgb_values)
            )
            temp = iou(torch.from_numpy(pred_mask[:,:,:-1]), torch.from_numpy(gt_mask[:,:,:-1]))
            total += temp
            count += 1
            # print(temp)
            # print(pred_mask[250,:,:])
            # if idx > 4:
            #     break
        print(total/count)



if __name__ == '__main__':
    class_color_map = { 0:np.array([254,0,0]).astype(np.uint8), 
        1:np.array([0, 114, 255]).astype(np.uint8), 
        2:np.array([0, 180, 55]).astype(np.uint8), 
        3:np.array([173,85,255]).astype(np.uint8), 
        4:np.array([255,252,0]).astype(np.uint8), 
        5:np.array([255,122,3]).astype(np.uint8), 
        6:np.array([110,68,7]).astype(np.uint8), 
        7:np.array([29,106,120]).astype(np.uint8), 
        8:np.array([0,0,0]).astype(np.uint8) }
    # class_weights = [23.84,4233.49,8661.66,82.97,1117.16,1085164.67,1085164.67,1085164.67,1.05] #weighted
    # class_weights = [23.84,4233.49,8661.66,82.97,1117.16,1085164.67,1085164.67,1085164.67,0] #weighted v1
    # class_weights = [23.84,4233.49,8661.66,82.97,1117.16,0,0,0,0] #weighted v2
    # class_weights = [1,1,1,1,1,1,1,1,0] #weighted v3
    # class_weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
    class_weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
    # utils.create_colored_masks(class_color_map, '/data/rachit/mtl_train/tuberculosis/tuberculosis_model/dr_annotations/annotations_test', '/data/rachit/mtl_train/tuberculosis/tuberculosis_model/dr_annotations/segmentation_labels_test')
    # utils.create_metadata_file('segmentation_utils/metadata_test.csv', '/data/rachit/mtl_train/tuberculosis/tuberculosis_model/dr_annotations/annotations_test')
    # train(20, 4, 512, class_weights, binary=False, training=True)
    train(20, 4, 512, class_weights, binary=False, training=True)
    

    # utils.get_channel_weights('/data/rachit/mtl_train/tuberculosis/tuberculosis_model/dr_annotations/segmentation_labels/', class_color_map, 100)
    # a = np.random.randint(low = 0, high = 2, size = (2,2))
    # b = np.random.randint(low = 0, high = 2, size = (2,2))
    # print(a)
    # print(b)
    # loss = utils.DiceLoss(eps = 1e-7, beta = 1.0, channel_weights = [0.8, 0.8])
    # # print(loss(torch.from_numpy(a), torch.from_numpy(b)))
    # c = np.array([[1, 1], [1, 1]])
    # ct = np.array([[0, 1], [0, 1]])
    # d = np.array([[1, 0], [0, 1]])
    # dt = np.array([[1, 1], [1, 0]])
    # # print(loss(torch.from_numpy(c), torch.from_numpy(ct)))
    # # print(loss(torch.from_numpy(d), torch.from_numpy(dt)))
    # e = np.stack([c,d], axis = -1)
    # et = np.stack([ct,dt], axis = -1)

    # # print(e.shape)
    # print(loss(torch.from_numpy(e), torch.from_numpy(et)))
    