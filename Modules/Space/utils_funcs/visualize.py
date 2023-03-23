import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection

from utils_funcs import transforms
from torchvision.utils import make_grid
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks


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
def plot_img_bbox(img, target, title=None):
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

    # print("area = ", target['area'])
    # print("min area = ", torch.min(target['area']))
    # print("max area = ", torch.max(target['area']))
    if title is not None:
        plt.title(title)
    plt.show()


def plot_log_per_epoch(epochs, y_label, values1, values2=None):
    if values2 is not None:
        # Plotting both the curves simultaneously
        plt.plot(epochs, values1, color='g', label='mAP@0.5')
        plt.plot(epochs, values2, color='b', label='mAP@0.5:0.95')
    else:
        plt.plot(epochs, values1)

    ax = plt.axes()
    # Setting the background color of the plot
    # using set_facecolor() method
    ax.set_facecolor("gainsboro")

    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.show()


# the function takes the original prediction and the iou threshold.

def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def show_predictions(model, model_path, ds, device, num_images=4, iou_thresh=0.2):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    i = 0
    for image_batch, target_batch in ds:
        if i == num_images:
            break
        image_batch, _ = transforms.preprocess(image_batch, device=device)
        for image, target in zip(image_batch, target_batch):
            if i == num_images // len(ds):
                i -= 1
                break
            with torch.no_grad():
                prediction = model([image.to(device)])[0]
            print("preds: ", prediction)
            print('predicted #boxes: ', len(prediction['labels']))
            print('real #boxes: ', len(target['labels']))
            nms_prediction = apply_nms(prediction, iou_thresh=iou_thresh)
            print('NMS APPLIED MODEL OUTPUT')
            print('predicted #boxes: ', len(prediction['labels']))
            plot_img_bbox(image, target, title='Original boxes')
            plot_img_bbox(image, nms_prediction, title='Predicted boxes')
        i += 1


def filter_model_output(output, score_threshold):
    filtred_output = list()
    for image in output:
        filtred_image = dict()
        for key in image.keys():
            filtred_image[key] = image[key][image['scores'] >= score_threshold]
        filtred_output.append(filtred_image)
    return filtred_output


def get_boolean_mask(output):
    for index, pred in enumerate(output):
        output[index]['masks'] = pred['masks'] > 0.5
        output[index]['masks'] = output[index]['masks'].squeeze(1)
    return output


def get_space_colors(output, colors=('green', 'red')):
    # label 1 = empty, 2 =occupied
    labels = output.numpy()
    colors = np.where(labels == 1, colors[0], colors[1]).tolist()
    return colors


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(12, 12))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_mask_predictions(image_list, preds, score_threshold=.8):
    output = filter_model_output(output=preds, score_threshold=score_threshold)
    output = get_boolean_mask(output)
    show([
        draw_segmentation_masks(image, prediction.get('masks'), alpha=0.6, colors=get_space_colors(prediction.get('labels')))
        for index, (image, prediction) in enumerate(zip(image_list, output))
    ])


def show_box_predictions(image_list, preds, score_threshold=.8, box_width=10):
    output = filter_model_output(output=preds, score_threshold=score_threshold)
    show([
        draw_bounding_boxes(image, prediction.get('boxes'), width=box_width, colors=get_space_colors(prediction.get('labels')))
        for index, (image, prediction) in enumerate(zip(image_list, output))
    ])
