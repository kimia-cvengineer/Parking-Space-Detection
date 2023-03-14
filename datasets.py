import torch
import cv2
import numpy as np
import os
import glob as glob
import random

from xml.etree import ElementTree as ET
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import albumentations
from torchvision import transforms as transforms

'''
Due to Yuchen's unfamiliarity with pytorch, some codes were code adapted from: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
Yuchen had made sure she understood every part of the original code and tried her best to modify them
'''

class custom_dataset(Dataset):
    def __init__(self, images_path,labels_path,width, height, classes,train=False):

        self.images_path = images_path
        self.labels_path = labels_path
        self.height = height
        self.width = width
        self.classes = classes
        self.train = train
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
        self.all_image_paths = []
        
        # get image paths and sort
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = sorted([os.path.basename(image_path) for image_path in self.all_image_paths])

    def load_image_and_labels(self, index):
        
        # pre-process images and load label path
        image_path = os.path.join(self.images_path, self.all_images[index])
        annot_file_path = os.path.join(self.labels_path, self.all_images[index][:-4] + '.xml')
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height)) / 255.0
       
        # extract bndbox info from xml files
        boxes = []
        orig_boxes = []
        labels = []
        tree = ET.parse(annot_file_path)
        root = tree.getroot()
        
        image_height = image.shape[0]
        image_width = image.shape[1] 
        
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            orig_boxes.append([xmin, ymin, xmax, ymax])
            
            
            boxes.append([(xmin/image_width)*self.width, 
                          (ymin/image_height)*self.height, 
                          (xmax/image_width)*self.width, 
                          (ymax/image_height)*self.height])
        
        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # return image, original bounding boxes, labels, area of bndbox, no_crowd instances, and img h&w
        return image, image_resized, orig_boxes, torch.as_tensor(boxes, dtype=torch.float32), \
            torch.as_tensor(labels, dtype=torch.int64),\
            (boxes[:, 3]-boxes[:, 1])*(boxes[:, 2]-boxes[:, 0]), \
            torch.zeros((boxes.shape[0],), dtype=torch.int64), \
            (image_width, image_height)

    # Adapted from: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet Methods 3
    # randomly choose four patches from the original image and three other different images and combine them together
    # I tried to simplify this code, but it doesn't seem to be possible
    def load_cutmix_image_and_boxes(self, index, resize_factor):
        image, _, _, _, _, _, _, _ = self.load_image_and_labels(index=index)
        orig_image = image.copy()
        # Resize the image
        image = cv2.resize(image, resize_factor)
        h, w, c = image.shape
        s = h // 2
    
        xc, yc = [int(random.uniform(h * 0.25, w * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]
        
        # Create empty image with the above resized image.
        result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, index in enumerate(indexes):
            image, image_resized, orig_boxes, boxes, labels, area, iscrowd, dims = self.load_image_and_labels(index=index)
            # Resize the current image according to the above resize,
            # else `result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]`
            # will give error when image sizes are different.
            image = cv2.resize(image, resize_factor)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # copying a portion of the original image to the corresponding region of a new image
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # update bounding boxes
            boxes[:, 0] += x1a-x1b
            boxes[:, 1] += y1a-y1b
            boxes[:, 2] += x1a-x1b
            boxes[:, 3] += y1a-y1b
            
            result_boxes.append(boxes)
            for class_name in labels:
                result_classes.append(class_name)

        # filtering out any bounding boxes with zero area 
        final_classes = []
        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)

        for idx in range(len(result_boxes)):
            box_w, box_h = result_boxes[idx, 2]-result_boxes[idx, 0], result_boxes[idx, 3]-result_boxes[idx, 1]
            if box_w > 0 and box_h > 0:
                final_classes.append(result_classes[idx])
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        
        return orig_image, result_image/255., torch.tensor(result_boxes), torch.tensor(np.array(final_classes)), area, iscrowd, dims

    def __getitem__(self, idx):
        if not self.train:
            # for validation and test sets
            image, image_resized, orig_boxes, boxes, labels, area, iscrowd, dims = self.load_image_and_labels(index=idx)
            
        else:
            # for training set
            # make sure each mosaic-ed image has at least one handicap labels
            for i in range(100):
                image, image_resized, boxes, labels, area, iscrowd, dims = self.load_cutmix_image_and_boxes(index=idx, resize_factor=(self.height, self.width))    
                if len(boxes) > 0:
                    break
            if i == 100:
                print('max attempt reached')

        # set the target label
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        
        trans = albumentations.Compose([albumentations.pytorch.ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc','label_fields': ['labels']})
        trans_result = trans(image=image_resized, bboxes=target['boxes'], labels=labels)
        image_resized = trans_result['image']
        target['boxes'] = torch.Tensor(trans_result['bboxes'])
            
        return image_resized, target

    def __len__(self):
        return len(self.all_images)
