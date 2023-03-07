from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2

from .utils import pooling
from .utils.class_head import ClassificationHead


class FasterRCNN_FPN(nn.Module):
    """
    A Faster R-CNN FPN inspired parking lot classifier.
    Passes the whole image through a CNN -->
    pools ROIs from the feature pyramid --> passes
    each ROI separately through a classification head.
    """
    def __init__(self, roi_res=7, pooling_type='square'):
        super().__init__()
        
        # backbone
        # by default, uses frozen batchnorm and 3 trainable layers
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        hidden_dim = 256
        
        # pooling
        self.roi_res = roi_res
        self.pooling_type = pooling_type
        
        # classification head
        in_channels = hidden_dim * self.roi_res**2
        self.class_head = ClassificationHead(in_channels)
        
        # load coco weights
        # url taken from: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
        weights_url = 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        state_dict = load_state_dict_from_url(weights_url, progress=False)
        self.load_state_dict(state_dict, strict=False)
        
        
    def forward(self, image, rois):
        # get backbone features
        features = self.backbone(image[None])
        
        # pool ROIs from features pyramid
        features = list(features.values())
        features = pooling.pool_FPN_features(features, rois, self.roi_res, self.pooling_type)
        
        # pass pooled ROIs through classification head to get class logits
        features = features.flatten(1)
        class_logits = self.class_head(features)

        return class_logits

def create_model():
    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # occupied + vacant

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained model head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



