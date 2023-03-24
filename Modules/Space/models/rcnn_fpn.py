from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn, \
    MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2


def create_model(with_mask=True, visionT=False):
    # load a model pre-trained on COCO
    if with_mask:
        if visionT:
            model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        else:
            model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        if visionT:
            model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        else:
            model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 1 + 2  # BG + occupied + vacant

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained model head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if with_mask:
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

    return model
