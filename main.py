import numpy
import torch

import Modules.Mark.inference as MarkDetector
import Modules.Space.inference as SpaceDetector
from Utils.utils import get_mark_spot_correspondences, filter_model_output
from Utils.visualize import draw_predictions

device = torch.device('cpu')
# img_path = './data/images/GOPR6754.JPG'
img_path = './data/images/GOPR6743.JPG'
img_path = 'C:\\Users\\kimia\\Desktop\\parking_lot2.jpg'
# Get Parking Sign prediction
marks = MarkDetector.get_prediction(img_path, device)[0]
print("mark preds : ", marks['labels'].shape)

# Get Parking Space prediction
spots = SpaceDetector.get_prediction(img_path, device)[0]

print("spot preds : ", spots['labels'].shape)

# filter out predictions
marks = filter_model_output([marks], score_threshold=.2)[0]
print("filtered mark preds : ", marks['labels'].shape)

spots = filter_model_output([spots], score_threshold=.6)[0]
print("filtered spot preds : ", spots['labels'].shape)

marks_len = marks['labels'].nelement()
# Merge predictions
corrs_indices = []
if marks_len > 0:
    corrs_indices, single_indices = get_mark_spot_correspondences(spots['boxes'], marks['boxes'])
    corrs_indices = numpy.array(corrs_indices)

# Check existence of corresponding marks
if len(corrs_indices) > 0:
    sign_spots_indices, marks_indices = corrs_indices[:, 0], corrs_indices[:, 1]

    # Handicapped spots
    corr_spots, corr_marks = {}, {}
    for spot_key, spot_val in spots.items():
        corr_spots[spot_key] = spot_val[sign_spots_indices]

    # Extract regular spots
    reg_spots = {}
    for spot_key, spot_val in spots.items():
        reg_spots[spot_key] = spot_val[single_indices]
    draw_predictions(img_path, reg_spots, corr_spots)
else:
    draw_predictions(img_path, spots)
