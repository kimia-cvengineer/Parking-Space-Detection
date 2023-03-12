import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torch_utils.engine import train_one_epoch, evaluate
from datasets import custom_dataset

import argparse
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import logging
import pandas as pd

'''
Codes adapted from: 
https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
'''

torch.multiprocessing.set_sharing_strategy('file_system')

np.random.seed(42)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch', dest='batch', default=25, type=int)
    parser.add_argument('--name',dest='name', default=None, type=str)
    args = vars(parser.parse_args())
    return args

def main(args):
    # Set folder
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    OUT_DIR = "outputs/" + str(args['name'])
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data config
    with open('/home/yuchen/venv/Faster_RCNN/handicap.yaml') as file:
        data_configs = yaml.safe_load(file)
    
    # set parameters and other info
    TRAIN_DIR_IMAGES = data_configs['TRAIN_DIR_IMAGES']
    TRAIN_DIR_LABELS = data_configs['TRAIN_DIR_LABELS']
    VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
    VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = 4
    NUM_EPOCHS = args['epochs']
    BATCH_SIZE = args['batch']
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 1280
  
    # set dataloader
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_dataset = custom_dataset(TRAIN_DIR_IMAGES, TRAIN_DIR_LABELS, IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES, train=True)
    valid_dataset =  custom_dataset(VALID_DIR_IMAGES, VALID_DIR_LABELS, IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,collate_fn=collate_fn)

    print("Training images: " + str(len(train_dataset)))
    print("Validation images: " + str(len(valid_dataset)) + '\n')
    
    # set model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers = 2)    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES) 

    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0004, weight_decay=0.0005)

    steps = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps,T_mult=1,verbose=False)

    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s',filename=f"{OUT_DIR}/model_log.log", filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    # initilize mAP
    val_map_05 = []
    val_map_05095 = []
    
    best_mAP_05 = 0
    best_mAP_05095 = 0

    # train
    for epoch in range(NUM_EPOCHS):

        # note: my dataset is in pascal_voc format, and train_one_epoch converts it to coco format 
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=100,scheduler=scheduler)
        coco_evaluator, stats = evaluate(model, valid_loader, device=DEVICE,out_dir=OUT_DIR,classes=CLASSES)

        val_map_05.append(stats[1])
        val_map_05095.append(stats[0])

        # store the best model (best 0.5mAP - I also tried to store the bext 0.5:0.95mAP model, but it didn't work very well)
        if val_map_05[-1] > best_mAP_05:
            best_mAP_05095 = val_map_05095[-1]
            best_mAP_05 = val_map_05[-1]

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'config': data_configs,
                'model_name': 'fasterrcnn_resnet50_fpn_v2'
                }, OUT_DIR +"/best_model.pth")
    
        print("\nBest 0.5 mAP for validation set: " + str(best_mAP_05095))
        print("Corresponding 0.5:0.95 mAP for validation set: " + str(best_mAP_05) + '\n')
        
        # save mAP plots
        figure = plt.figure(figsize=(10, 8))
        axes = figure.add_subplot()
        axes.plot(val_map_05, color='green', linestyle='-', label='mAP@0.5')
        axes.plot(val_map_05095, color='blue', linestyle='-', label='mAP@0.5:0.95')
        axes.set_xlabel('Epochs')
        axes.set_ylabel('mAP')
        axes.legend()
        figure.savefig(f"{OUT_DIR}/map.png")
        plt.close()

        # save log
        log_dict_keys = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    ]
        log_dict = {}
        with open(f"{OUT_DIR}/model_log.log", 'a+') as f:
            f.writelines('\n')
            for i, key in enumerate(log_dict_keys):
                out_str = str(key) + ' = ' + str(stats[i])
                logger.debug(out_str)
            logger.debug('\n')

        if epoch == 0:
            model_map = pd.DataFrame(columns=['epoch', 'map', 'map_05'])
            model_map.to_csv(os.path.join(OUT_DIR, 'model_map.csv'), mode='w', index=False)

        df_temp = pd.DataFrame({
                'epoch': int(epoch+1),
                'map_05': [float(stats[0])],
                'map': [float(stats[1])]})
        df_temp.to_csv(os.path.join(OUT_DIR, 'model_map.csv'), mode='a', index=False)

if __name__ == '__main__':
    args = parse_opt()
    main(args)
