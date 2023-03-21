from PIL import Image
from matplotlib import pyplot as plt, patches
from torchvision.utils import draw_bounding_boxes

from Modules.Space.utils_funcs.visualize import get_mask_colors


def draw_predictions(img_path, spots, signs):
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=[12, 8])

    # TODO draw spot masks
    # spot_masks = spots['masks']
    spot_boxes = spots['boxes']
    sign_boxes = signs['boxes']

    # Draw spot boxes on top of the img using torch built-in func
    # TODO reformat this
    img_spots = draw_bounding_boxes(img, spot_boxes, width=15,
                                    colors=get_mask_colors(spots.get('labels')))

    ax.imshow(img_spots)

    for box in sign_boxes:
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        # mask = mask[0]
        # Draw the bounding box on top of the image
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none', rotation_point='center')
        ax.add_patch(rect)
        # poly = patches.Polygon([(mask[0], mask[1]), (mask[2], mask[3]),
        #                         (mask[4], mask[5]), (mask[6], mask[7])], edgecolor='g', linewidth=2, facecolor='none')
        # ax.add_patch(poly)
    plt.show()
