import numpy
import torch

import Modules.Sign.inference as SignDetector
import Modules.Space.inference as SpaceDetector
from Utils.utils import get_sign_spot_correspondences, filter_model_output
from Utils.visualize import draw_predictions

device = torch.device('cpu')
# img_path = './data/images/GOPR6754.JPG'
# img_path = './data/images/GOPR6741.JPG'
img_path = './data/images/GOPR6720.JPG'
# Get Parking Sign prediction
signs = SignDetector.get_prediction(img_path, device)[0]
print("sign preds : ", signs['labels'].shape)

# Get Parking Space prediction
spots = SpaceDetector.get_prediction(img_path, device)[0]

print("spot preds : ", spots['labels'].shape)

# filter out predictions
signs = filter_model_output([signs], score_threshold=.2)[0]
print("filtered sign preds : ", signs['labels'].shape)

spots = filter_model_output([spots], score_threshold=.5)[0]
print("filtered spot preds : ", spots['labels'].shape)

signs_len = signs['labels'].nelement()
# Merge predictions
corrs_indices = []
if signs_len > 0:
    corrs_indices, single_indices = get_sign_spot_correspondences(spots['boxes'], signs['boxes'])
    corrs_indices = numpy.array(corrs_indices)
    print("corrs_indices : ", corrs_indices)

# Check existence of corresponding signs
if len(corrs_indices) > 0:
    sign_spots_indices, signs_indices = corrs_indices[:, 0], corrs_indices[:, 1]

    # Handicapped spots
    corr_spots, corr_signs = {}, {}
    for spot_key, spot_val in spots.items():
        corr_spots[spot_key] = spot_val[sign_spots_indices]

if len(corrs_indices) > 0:
    # Extract regular spots
    reg_spots = {}
    for spot_key, spot_val in spots.items():
        reg_spots[spot_key] = spot_val[single_indices]
    draw_predictions(img_path, reg_spots, corr_spots)
else:
    draw_predictions(img_path, spots)
