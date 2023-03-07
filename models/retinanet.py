import math

import torch
from torchvision.models.detection import retinanet_resnet50_fpn


def create_model():
    # load a model pre-trained on COCO
    model = retinanet_resnet50_fpn(weights='DEFAULT')

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # occupied + vacant

    # replace classification layer
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    # out_channels
    #
    # cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    # torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    # torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
    # # assign cls head to model
    # model.head.classification_head.cls_logits = cls_logits

    return model
