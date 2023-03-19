import Modules.Sign.inference as SignDetector
import Modules.Space.inference as SpaceDetector


def get_spot_categories(spot_preds, sing_preds):
    pass


def visualize_predictions(preds):
    pass


# Get Parking Sign prediction
signs = SignDetector.get_prediction(img_path)

# Get Parking Space prediction
spots = SpaceDetector.get_prediction(img_path)

# Merge predictions
spot_categories = get_spot_categories()

visualize_predictions(spot_categories)
