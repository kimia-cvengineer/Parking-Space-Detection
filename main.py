import Modules.Sign.inference as SignDetector
import Modules.Space.inference as SpaceDetector

from Utils.Box import calculate_iou_matrix


def get_spot_categories(spot_preds, sign_preds):
    pass


def visualize_predictions(preds):
    pass


def get_sign_spot_correspondences(spot_preds, sign_preds):
    """
    Returns the row and column indices with the maximum IoU value for each row in the IoU matrix to get the best
    correspondence

    Arguments:
    iou_matrix -- a matrix of size (num_rows, num_cols) representing the IoU calculation for each box pair.

    Returns:
    A list of tuples (spot_preds, sign_preds) representing the pair of spot and sign boxes.
    """

    # Calculate the IoU for each spot, sign pair
    iou_matrix = calculate_iou_matrix(spot_preds, sign_preds)

    correspondences = []

    # Iterate over each row in the matrix
    for row in range(len(iou_matrix)):
        # Find the colum of the maximum IoU value in the row
        max_col = iou_matrix[row].index(max(iou_matrix[row]))

        # Add the pair to the list of correspondences
        correspondences.append((spot_preds[row], sign_preds[max_col]))

    return correspondences


img_path = './data/images/GOPR6543.JPG'
# Get Parking Sign prediction
signs = SignDetector.get_prediction(img_path)

# Get Parking Space prediction
spots = SpaceDetector.get_prediction(img_path)


# TODO Filter out predictions

# Merge predictions
# spot_categories = get_sign_spot_correspondences(spots, signs)
#
# visualize_predictions(spot_categories)
