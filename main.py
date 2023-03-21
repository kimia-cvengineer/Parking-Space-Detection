import torch

import Modules.Sign.inference as SignDetector
import Modules.Space.inference as SpaceDetector

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
img_path = './data/images/GOPR6543.JPG'
# Get Parking Sign prediction
signs = SignDetector.get_prediction(img_path, device)

# Get Parking Space prediction
spots = SpaceDetector.get_prediction(img_path, device)

print("sign preds : ", signs)
print("spot preds : ", spots)
# TODO Filter out predictions

# Merge predictions
# spot_categories = get_sign_spot_correspondences(spots, signs)
#
# draw_predictions(spot_categories)
