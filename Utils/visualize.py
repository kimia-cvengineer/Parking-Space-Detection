from PIL import Image
import torchvision.transforms.functional as F
import torch
from matplotlib import pyplot as plt, patches
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from Modules.Space.utils_funcs.visualize import get_space_colors, get_boolean_mask


def draw_predictions(img_path, spots, marks=None):
    """
    Draw parking spaces predictions

    Arguments
    img_path: image path to draw on
    spots: a dict containing spot information in a tensor format
    marks: a dict containing marks information in a tensor format
    if there is no marks, set mark to None

    Returns None
    """
    img = read_image(img_path)
    fig, ax = plt.subplots(figsize=[12, 8])

    # Draw regular spots masks on top of the img using torch built-in func
    bool_masked_spots = get_boolean_mask([spots])[0]
    img_masks = draw_segmentation_masks(img, bool_masked_spots.get('masks'), alpha=0.6, colors=get_space_colors(spots.get('labels'), colors=('lightgreen', 'salmon')))
    # Draw regular spot boxes on top of the img using torch built-in func
    img_spots = draw_bounding_boxes(img_masks, spots.get('boxes'), width=15, colors=get_space_colors(spots.get('labels')))

    # Display confidence score for each predicted spot
    for score, label, box in zip(spots.get('scores'), spots.get('labels'), spots.get('boxes')):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        occupancy, color = ("empty", 'green') if label == 1 else ("occupied", 'red')
        ax.annotate(f"{occupancy} {score.item():.2f}", (x, y + height / 2), color=color, weight='bold', fontsize=7)

    # Draw accessible spots masks
    if marks is not None:
        bool_masked_marks = get_boolean_mask([marks])[0]
        img_spots = draw_segmentation_masks(img_spots, bool_masked_marks.get('masks'), alpha=0.6,
                                            colors=get_space_colors(marks.get('labels'), colors=('lightcyan', 'salmon')))

    img_spots = img_spots.detach()
    img_spots = F.to_pil_image(img_spots)
    ax.imshow(img_spots)

    # Draw accessible spots boxes
    if marks is not None:
        sign_boxes = marks['boxes']
        sign_labels = marks['labels']
        sign_boxes = sign_boxes.detach().numpy()
        for label, box in zip(sign_labels, sign_boxes):
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            # Draw the bounding box on top of the image
            # Blue boxes represent empty handicap sign, red the occupied ones
            color = 'blue' if label == 1 else 'red'
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth=2,
                                     edgecolor=color,
                                     facecolor='none', rotation_point='center')
            ax.add_patch(rect)
            ax.annotate(f"Accessible", (x+width/3, y+height/2), color=color, weight='bold', fontsize=7)
    plt.show()
