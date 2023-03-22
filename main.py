import numpy
import torch

import Modules.Sign.inference as SignDetector
import Modules.Space.inference as SpaceDetector
from Utils.utils import get_sign_spot_correspondences, filter_model_output
from Utils.visualize import draw_predictions

device = torch.device('cpu')
img_path = './data/images/GOPR6761.JPG'
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
# Merge predictions
corrs_indices, single_indices = get_sign_spot_correspondences(spots['boxes'], signs['boxes'])
corrs_indices = numpy.array(corrs_indices)
print("corrs_indices : ", corrs_indices)
sign_spots_indices, signs_indices = corrs_indices[:, 0], corrs_indices[:, 1]
# print("corrs spots : ", spots['boxes'][spots_indices])
print("len corrs spots : ", len(spots['boxes'][sign_spots_indices]))
print("spots : ", spots)

# Handicapped spots
corr_spots, corr_signs = {}, {}
for spot_key, spot_val in spots.items():
    corr_spots[spot_key] = spot_val[sign_spots_indices]
print("corr_spots: ", corr_spots)

reg_spots = {}
for spot_key, spot_val in spots.items():
    reg_spots[spot_key] = spot_val[single_indices]

draw_predictions(img_path, reg_spots, corr_spots)
