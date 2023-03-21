import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_prediction(image_path):
    # set up model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # put the weight path here
    checkpoint = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # load image
    img = np.array(Image.open(image_path).copy())

    # feature engineering
    img_copy = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    img[:, :, -1] = img_copy
    img = Image.fromarray(img).copy()

    # predict
    transform = transforms.ToTensor()
    input = transform(img)
    input = input.unsqueeze_(0)
    output = model(input)

    return output