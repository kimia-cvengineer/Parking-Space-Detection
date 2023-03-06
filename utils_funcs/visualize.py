import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection


def image_pt_to_np(image):
    """
    Convert a PyTorch image to a NumPy image (in the OpenCV format).
    """
    image = image.cpu().clone()
    image = image.permute(1, 2, 0)
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_sd = torch.tensor([0.229, 0.224, 0.225])
    image = image * image_sd + image_mean
    image = image.clip(0, 1)
    return image


def show_warps(warps, nrow=8, fname=None, show=False):
    """
    Plot a tensor of image patches / warps.
    """
    image_grid = torchvision.utils.make_grid(warps, nrow=nrow)
    fig, ax = plt.subplots(figsize=[8, 8])
    ax.imshow(image_pt_to_np(image_grid))
    ax.axis('off')
    save_fig(fig, fname, show)


def occupancy_colors(scores):
    """
    Set the coloring scheme for occupancy plots.
    """
    colors = np.zeros([len(scores), 3])
    colors[:, 0] = scores  # red
    colors[:, 1] = 1 - scores  # green
    colors[:, 2] = 0  # blue
    return colors


def save_fig(fig, fname, show=False):
    """
    A helper function to show or save a figure. 
    """
    if fname is not None:
        plt.savefig(fname, dpi=300, pil_kwargs={'quality': 80}, bbox_inches='tight', pad_inches=0)
        if not show:
            plt.close(fig)
    else:
        plt.show()


def plot_ds_image(image, rois, occupancy, true_occupancy=None, fname=None, show=False):
    """
    Plot an annotated dataset image with occupancy equal to `occupancy`.
    If `true_occupancy` is specified, `occupancy` is assumed to represent the
    predicted occupancy.
    """
    # plot image
    fig, ax = plt.subplots(figsize=[12, 8])
    ax.imshow(image_pt_to_np(image))
    ax.axis('off')

    # convert rois
    C, H, W = image.shape
    rois = rois.cpu().clone()
    rois[..., 0] *= (W - 1)
    rois[..., 1] *= (H - 1)
    rois = rois.numpy()

    # plot annotations
    polygons = []
    colors = occupancy_colors(occupancy.cpu().numpy())
    i = 0
    mid_roi = rois.shape[0] // 2
    for roi, color in zip(rois, colors):
        if i == mid_roi:
            print(f"x1, y1: ({roi[0][0]}, {roi[0][1]})")
            print(f"x2, y2: ({roi[1][0]}, {roi[1][1]})")
            print(f"x3, y3: ({roi[2][0]}, {roi[2][1]})")
            print(f"x4, y4: ({roi[3][0]}, {roi[3][1]})")
            polygon = Polygon(roi, fc=color, alpha=0.3)
            polygons.append(polygon)
        i += 1
    p = PatchCollection(polygons, match_original=True)
    ax.add_collection(p)

    # plot prediction
    if true_occupancy is not None:
        # only show those crosses where the predictions are incorrect
        pred_inc = occupancy.round() != true_occupancy
        rois_subset = rois[pred_inc]
        ann_subset = true_occupancy[pred_inc]

        # create an array of crosses for each parking space
        lines = np.array(rois_subset)[:, [0, 2, 1, 3], :].reshape(len(rois_subset) * 2, 2, 2)
        colors = occupancy_colors(ann_subset.cpu().numpy())

        # add the crosses to the plot
        colors = np.repeat(colors, 2, axis=0)
        lc = LineCollection(lines, colors=colors, lw=1)
        ax.add_collection(lc)

    # save figure
    save_fig(fig, fname, show)


# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(figsize=[12, 8])
    # fig.set_size_inches(5, 5)
    a.imshow(image_pt_to_np(img))

    for box in (target['boxes']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()


def plot_log_per_epoch(epochs, values, y_label):
    plt.plot(epochs, values)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.show()
