
def calculate_iou(bb1, bb2):
    """
    Calculates the confusion matrix between two given bounding boxes using the Intersection over Union (IoU) calculation.

    Arguments:
    bb1 -- a tuple (x1, y1, x2, y2) representing the coordinates of the first bounding box
    bb2 -- a tuple (x1, y1, x2, y2) representing the coordinates of the second bounding box

    Returns:
    The IoU calculation for the box pair
    """

    # Calculate the intersection of the two bounding boxes
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    # Calculate the area of the intersection and the union of the two bounding boxes
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)
    union_area = bb1_area + bb2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area
    return iou


def calculate_iou_matrix(bbox1, bbox2):
    """
    Calculates the IoU matrix for each pair of bounding boxes between two lists of bounding boxes.

    Arguments:
    bounding_boxes_1 -- a list of tuples (x1, y1, x2, y2) representing the coordinates of the bounding boxes in the first list
    bounding_boxes_2 -- a list of tuples (x1, y1, x2, y2) representing the coordinates of the bounding boxes in the second list

    Returns:
    A matrix of size (len(bounding_boxes_1), len(bounding_boxes_2)) representing the IoU calculation for each box pair.
    """

    # Initialize the matrix with zeros
    iou_matrix = [[0 for _ in range(len(bbox2))] for _ in range(len(bbox1))]

    # Calculate the IoU for each pair of bounding boxes
    for i in range(len(bbox1)):
        for j in range(len(bbox2)):
            iou_matrix[i][j] = calculate_iou(bbox1[i], bbox2[j])

    return iou_matrix

