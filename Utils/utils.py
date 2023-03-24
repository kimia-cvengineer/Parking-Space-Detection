from Utils.Box import calculate_iou_matrix


def filter_model_output(output, score_threshold=.3):
    filtred_output = list()
    for image in output:
        filtred_image = dict()
        for key in image.keys():
            filtred_image[key] = image[key][image['scores'] >= score_threshold]
        filtred_output.append(filtred_image)
    return filtred_output


def get_mark_spot_correspondences(spot_preds, mark_preds, thresh=.01):
    """
    Returns the row and column indices with the maximum IoU value for each row in the IoU matrix to get the best
    correspondence

    Arguments
    iou_matrix: a matrix of size (num_rows, num_cols) representing the IoU calculation for each box pair.

    Returns: A list of tuples (row_index, col_index) representing the row and column indices with the maximum IoU
    value for each row in the matrix.
    """

    # Calculate the IoU for each spot, sign pair
    iou_matrix = calculate_iou_matrix(spot_preds, mark_preds)

    corrs_indices, single_indices = [], []

    # Iterate over each row in the matrix
    for row in range(len(iou_matrix)):
        # Find the colum of the maximum IoU value in the row
        max_iou = max(iou_matrix[row])
        max_col = iou_matrix[row].index(max_iou)
        if max_iou.item() > thresh:
            # Add the pair to the list of correspondences
            corrs_indices.append((row, max_col))
        else:
            single_indices.append(row)

    return corrs_indices, single_indices

