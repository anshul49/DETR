import os
import numpy as np 
import pandas as pd 
from datetime import datetime
import time
import random
from tqdm.autonotebook import tqdm
import re
import pydicom
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection

import cv2
import sys
sys.path.append('./detr/')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2

from glob import glob

from tqdm import tqdm
from ensemble_boxes import *

from map_boxes import mean_average_precision_for_boxes

# %%
print(torch.cuda.is_available())

# %%

# thoracic abnormalities (classes)
CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation',
    'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Other lesion', 
    'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No finding'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# %%

def read_images():
    train_df = pd.read_csv('/mnt/home/dhingra/DETR/resized/train.csv')
    train_df.fillna(0, inplace=True)
    return train_df

# %%
def scale_images(train_df):
    train_df.loc[train_df["class_id"] == 14, ['x_max', 'y_max']] = 1.0
    train_df.loc[train_df["class_id"] == 14, ['x_min', 'y_min']] = 0

    IMG_SIZE = 256
    train_df['xmin'] = (train_df['x_min']/train_df['width'])*IMG_SIZE
    train_df['ymin'] = (train_df['y_min']/train_df['height'])*IMG_SIZE
    train_df['xmax'] = (train_df['x_max']/train_df['width'])*IMG_SIZE
    train_df['ymax'] = (train_df['y_max']/train_df['height'])*IMG_SIZE

    # set to the images with no object (class 14), bounding box with coordinates [xmin=0 ymin=0 xmax=1 ymax=1]
    train_df.loc[train_df["class_id"] == 14, ['xmax', 'ymax']] = 1.0
    train_df.loc[train_df["class_id"] == 14, ['xmin', 'ymin']] = 0
    return train_df

# %%
def define_folds(train_df):
    unique_images = train_df["image_id"].unique()
    df_split = pd.DataFrame(unique_images, columns = ['unique_images']) 

    # create one column with the number of fold (for the k-fold cross validation)
    df_split["kfold"] = -1
    df_split = df_split.sample(frac=1).reset_index(drop=True)
    y = df_split.unique_images.values
    kf = model_selection.GroupKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df_split, y=y, groups=df_split.unique_images.values)):
        df_split.loc[v_, "kfold"] = f

    # annotated boxes from same "image id" (image) should be in the same fold [during training each image with its boxes is as one input]
    train_df["kfold"] = -1
    for ind in train_df.index: 
         train_df["kfold"][ind] = df_split.loc[ df_split["unique_images"] ==  train_df["image_id"][ind]]["kfold"]

    train_df.set_index('image_id', inplace=True)
    return train_df

# %%
def boxes_fusion(df):
    # apply weighted boxes fusion for ensemling overlapping annotated boxes
    # Default WBF config 
    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1
    results = []
    image_ids = df.index.unique()
    for image_id in tqdm(image_ids, total=len(image_ids)):
        # All annotations for the current image.
        data = df[df.index == image_id]
        kfold = data['kfold'].unique()[0]
        data = data.reset_index(drop=True)
        
        # WBF expects the coordinates in 0-1 range.
        max_value = data.iloc[:, 4:].values.max()
        data.loc[:, ["xmin", "ymin", "xmax", "ymax"]] = data.iloc[:, 4:] / max_value
        
        if data.class_id.unique()[0] !=14:
            annotations = {}
            weights = []
            # Loop through all of the annotations
            for idx, row in data.iterrows():
                rad_id = row["rad_id"]
                if rad_id not in annotations:
                    annotations[rad_id] = {
                        "boxes_list": [],
                        "scores_list": [],
                        "labels_list": [],
                    }
                    # We consider all of the radiologists as equal.
                    weights.append(1.0)
                annotations[rad_id]["boxes_list"].append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
                annotations[rad_id]["scores_list"].append(1.0)
                annotations[rad_id]["labels_list"].append(row["class_id"])

            boxes_list = []
            scores_list = []
            labels_list = []

            for annotator in annotations.keys():
                boxes_list.append(annotations[annotator]["boxes_list"])
                scores_list.append(annotations[annotator]["scores_list"])
                labels_list.append(annotations[annotator]["labels_list"])

            # Calculate WBF
            boxes, scores, labels = weighted_boxes_fusion(boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr
            )
            for idx, box in enumerate(boxes):
                results.append({
                    "image_id": image_id,
                    "class_id": int(labels[idx]),
                    "rad_id": "wbf",
                    "xmin": box[0]* max_value,
                    "ymin": box[1]* max_value,
                    "xmax": box[2]* max_value,
                    "ymax": box[3]* max_value,
                    "kfold":kfold,
                })
        # if class is nothing then have it once (instead of 3 times in the same image)
        if data.class_id.unique()[0] ==14:
            for idx, box in enumerate([0]):
                results.append({
                    "image_id": image_id,
                    "class_id": data.class_id[0],
                    "rad_id": "wbf",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 1,
                    "ymax": 1,
                    "kfold":kfold,
                })
            
    results = pd.DataFrame(results)
    return results

# %%
def pascal_to_coco(train_df):
    train_df['coco_x'] = train_df['xmin'] + (train_df['xmax'] - train_df['xmin'] )/2
    train_df['coco_y'] = train_df['ymin'] + (train_df['ymax'] - train_df['ymin'] )/2
    train_df['coco_w'] = train_df['xmax'] - train_df['xmin'] 
    train_df['coco_h'] = train_df['ymax'] - train_df['ymin'] 

    train_df.loc[train_df['class_id'] == 14, 'coco_x'] = 1
    train_df.loc[train_df['class_id'] == 14, 'coco_y'] = 1
    train_df.loc[train_df['class_id'] == 14, 'coco_w'] = 0.5
    train_df.loc[train_df['class_id'] == 14, 'coco_h'] = 0.5
    
    return train_df

# %%
def preprocessing():
    train_df = read_images()
    train_df = scale_images(train_df)
    train_df = define_folds(train_df)
    train_df = boxes_fusion(train_df)
    train_df.set_index('image_id', inplace=True)
    train_df = pascal_to_coco(train_df)
    print(train_df.head())
    return train_df

# %%

def get_train_transforms():
    # image augmentations for the training set
    return A.Compose([A.ToGray(p=0.01),
                      A.Cutout(num_holes=10, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
                      ToTensorV2(p=1.0)],
                      p=1.0,
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

def get_valid_transforms():
    # image augmentations for the validation set
    return A.Compose([ToTensorV2(p=1.0)], 
                      p=1.0, 
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

# %%
DIR_TRAIN_PNG = '/mnt/home/dhingra/DETR/resized/train'

class VinDataset(Dataset):
    def __init__(self,image_ids,dataframe,transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self,index):
        image_id = self.image_ids[index]
        records = self.df.loc[image_id]
        labels = records['class_id']
        
        image = cv2.imread(f'{DIR_TRAIN_PNG}/{image_id}.png', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # DETR takes in data in coco format    
        boxes = records[['coco_x', 'coco_y', 'coco_w', 'coco_h']].values
     
        # As pointed out by PRVI It works better if the main class is labelled as zero
        labels =  np.array(labels)
    
        if boxes.ndim == 1 : 
            boxes = np.expand_dims(boxes, axis=0)
            labels = np.expand_dims(labels, axis=0)
        
        # As pointed out by PRVI It works better if the main class is labelled as zero
        labels =  np.array(labels)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }

        sample = self.transforms(**sample)
        image = sample['image']
        boxes = sample['bboxes']
        labels = sample['labels']
        
        # Normalizing BBOXES
        _,h,w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)

        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([index])

        return image/255, target, image_id

# %% [markdown]
# MODEL

# %%

import torch.nn.functional as F
class DETRModel(nn.Module):
    def __init__(self,num_classes,num_queries):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = True


        self.in_features = self.model.class_embed.in_features
        
        self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes+1)
        self.model.num_queries = self.num_queries
        
    def forward(self,images):
        return self.model(images)

# %%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# %%
def train_fn(data_loader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    criterion.train()
    
    summary_loss = AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    check_repeats = []
    for step, (images, targets, image_ids) in enumerate(tk0):
            if image_ids in check_repeats:
                continue
            else:
                check_repeats.append(image_ids)

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                output = model(images)

                loss_dict = criterion(output, targets)
                weight_dict = criterion.weight_dict

                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                optimizer.zero_grad()

                losses.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                summary_loss.update(losses.item(),BATCH_SIZE)
                tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss

# %%
class EarlyStopping():
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

# %%

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device='cuda')
    return b

# %% [markdown]
# DEBUG

# %%
def eval_fn(data_loader, model,criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()
    map_df = pd.DataFrame()
    map_df_target = pd.DataFrame()
    
    with torch.no_grad():
        check_repeats_val = []
        tk0 = tqdm(data_loader, total=len(data_loader))
        
        for step, (images, targets, image_ids) in enumerate(tk0):
            if image_ids in check_repeats_val:
                continue
            else:
                check_repeats_val.append(image_ids)

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)

                # mAP targets
                for count, label in enumerate(targets[0]['labels']):
                    text = f'{CLASSES[label]}' 
                    xmin = targets[0]['boxes'][count][0] - (targets[0]['boxes'][count][2])/2
                    xmax = targets[0]['boxes'][count][0] + (targets[0]['boxes'][count][2])/2  
                    ymin = targets[0]['boxes'][count][1] - (targets[0]['boxes'][count][3])/2
                    ymax = targets[0]['boxes'][count][1] + (targets[0]['boxes'][count][3])/2

                    data = pd.DataFrame({"ImageID": [image_ids[0]],"LabelName": [text], \
                    "XMin": [xmin.item()], "XMax": [xmax.item()], "YMin": [ymin.item()], "YMax": [ymax.item()]})
                    temp_df = pd.DataFrame(data)  # Create a temporary DataFrame 
                    map_df_target = pd.concat([map_df_target, temp_df], ignore_index=True)
                                  

                probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.08
                boxes = rescale_bboxes(outputs['pred_boxes'][0, keep], (256,256))
                prob = probas[keep]

                colors = COLORS * 100
                for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):

                    cl = p.argmax()
                    text = f'{CLASSES[cl]}' 
                    
                    # Dataframe for MAP
                    data = pd.DataFrame({"ImageID": [image_ids[0]],"LabelName": [text], "Conf": [p[cl].item()], "XMin": [xmin/256], "XMax": [xmax/256], "YMin": [ymin/256], "YMax": [ymax/256]})
                    temp_df_1 = pd.DataFrame(data)  # Create a temporary DataFrame 
                    map_df = pd.concat([map_df, temp_df_1], ignore_index=True)          

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict

                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                summary_loss.update(losses.item(),BATCH_SIZE)
                tk0.set_postfix(loss=summary_loss.avg)
        
        ann = map_df_target[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
        det = map_df[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
        mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, iou_threshold=0.4)

        print("mean_ap : {}".format(mean_ap))
        print("average_precisions : {}".format(average_precisions))
        
    return summary_loss, mean_ap

# %%
def collate_fn(batch):
    return tuple(zip(*batch))

# %%
def run(train_df, fold):
            
    df_train = train_df[train_df['kfold'] != fold]
    df_valid = train_df[train_df['kfold'] == fold]

    train_dataset = VinDataset(
                                image_ids=df_train.index.values,
                                dataframe=df_train,
                                transforms=get_train_transforms()
                                )

    valid_dataset = VinDataset(
                                image_ids=df_valid.index.values,
                                dataframe=df_valid,
                                transforms=get_valid_transforms()
                                )
    
    train_data_loader = DataLoader(
                                    train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=collate_fn
                                    )

    valid_data_loader = DataLoader(
                                    valid_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=collate_fn
                                    )
    
    matcher = HungarianMatcher()
    weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
    losses = ['labels', 'boxes', 'cardinality']

    device = torch.device('cuda')
    model = DETRModel(num_classes=num_classes,num_queries=num_queries)
    # from torchvision import models
    # from torchsummary import summary
    # print(summary(model, (3, 256, 256)))
    model = model.to(device)
    print("Model transferred to GPU:", next(model.parameters()).device)
    
    criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef = null_class_coef, losses=losses)
    criterion = criterion.to(device)
    
    # LR = 5e-3
    LR = 3e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    print("Optimizer initialized")
   
    best_loss = 0
    val_loss_track_switch = 0
    all_train_losses = []
    all_valid_losses = []
    all_mean_ap = []
    columns = ['train_losses', 'valid_losses', 'mean_ap']
    df_losses = pd.DataFrame(columns = columns )
    df_losses.to_csv("/mnt/home/dhingra/DETR/all_losses.csv",mode='a', index=False)
    for epoch in range(EPOCHS):
        print("Epoch:", epoch + 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        train_loss = train_fn(train_data_loader, model,criterion, optimizer,device,scheduler=None,epoch=epoch)
        if val_loss_track_switch % 2 == 0: 
            LR = LR/1.12        
        valid_loss, map_validation = eval_fn(valid_data_loader, model,criterion, device)
        val_loss_track_switch = val_loss_track_switch + 1
        
        df_losses = df_losses.append({'train_losses': train_loss.avg,'valid_losses': valid_loss.avg,'mean_ap': map_validation}, ignore_index=True)
        df_losses.to_csv("all_losses.csv",index=False, header=False, mode='a')
        df_losses.drop(df_losses.tail(1).index,inplace=True)
        
        print('| EPOCH {}/{} | TRAIN_LOSS {} | VALID_LOSS {} |'.format(epoch+1, EPOCHS, train_loss.avg, valid_loss.avg))
        
        if map_validation > best_loss:
            best_loss = map_validation
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold, epoch+1))
            torch.save(model.state_dict(), f'/mnt/home/dhingra/DETR/model_checkpoints/detr_model.pth')
        
        # Early stopping
        early_stopping = EarlyStopping()
        early_stopping(best_loss)
        if early_stopping.early_stop:
            break
    return model
     

# %%
import torch
print(torch.cuda.is_available())

# %% [markdown]
# CONFIG

# %%
n_folds = 5
seed = 42
num_classes = 15
num_queries = 2
null_class_coef = 0.2
BATCH_SIZE = 32
EPOCHS = 30

# %% [markdown]
# MODEL TRAINING

# %%
model = DETRModel(num_classes=num_classes,num_queries=num_queries)
from torchvision import models
from torchsummary import summary
def gnn_model_summary(model):
    
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
print(gnn_model_summary(model))
# model.paramlists()
# print(f'Num params: {sum([param.nelement() for param in model.parameters()])}')
# print(summary(model, (3, 256, 256)))

# %%
def model_training():
    train_df = preprocessing()
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    model = run(train_df, fold=0)
    return

# %%
mode = 'train'
if mode == 'train':
    model_training()


