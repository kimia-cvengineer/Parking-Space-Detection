import json
import numpy as np
import shapely
import math

import torch
from PIL import Image


def load_data(dataset_path, ds_type):
    # load all annotations
    with open(f'{dataset_path}/annotations.json', 'r') as f:
        all_annotations = json.load(f)

    # select train, valid, test, or all annotations
    if ds_type in ['train', 'valid', 'test']:
        # select train, valid, or test annotations
        annotations = all_annotations[ds_type]
    else:
        # select all annotations
        assert ds_type == 'all'
        # if using all annotations, combine the train, valid, and test dicts
        annotations = {k: [] for k in all_annotations['train'].keys()}
        for ds_type in ['train', 'valid', 'test']:
            for k, v in all_annotations[ds_type].items():
                annotations[k] += v

    fname_list = annotations['file_names']
    rois_list = annotations['rois_list']
    occupancy_list = annotations['occupancy_list']

    return fname_list, rois_list, occupancy_list


def convert2coco(file_names, rois_list, label_list, save_path):
    # COCO dataset format dictionary
    coco_dataset = {
        "info": {
            "description": "ACPDS",
            "url": "",
            "version": "",
            "year": 2023,
            "contributor": "Kimia.A",
            "date_created": "2023-03-11"
        },
        "licenses": [
            {
                "url": "",
                "id": 0,
                "name": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Open an image and get its width and height
    # all images are of the same W, H
    with Image.open("./data/images/" + file_names[0]) as img:
        width, height = img.size


    # Loop through each image folder
    i = 0
    for fname, rois, labels in zip(file_names, rois_list, label_list):

        # Add image information to COCO dataset
        coco_dataset["images"].append({
            "id": i,
            "file_name": fname,
            "width": width,
            "height": height
        })

        # Loop through each annotation
        for roi, label in zip(rois, labels):
            # If category label is new, add it to category dictionary
            category_id = 1 if label else 0

            # Un-normalize points to the size of the img
            roi = np.array(roi)
            roi[..., 0] *= (width - 1)
            roi[..., 1] *= (height - 1)

            # Add annotation information to COCO dataset
            coco_dataset["annotations"].append({
                "id": len(coco_dataset["annotations"]),
                "image_id": i,
                "category_id": category_id,
                "bbox": get_rotatedbox(roi),  # TODO make it like (x, y, w, h, theta)
                "area": polygon_area(roi),
                "iscrowd": 1
            })

        i += 1

    coco_dataset["categories"].append({
        "supercategory": "Occupancy",
        "id": 0,
        "name": "empty"
    })
    coco_dataset["categories"].append({
        "supercategory": "Occupancy",
        "id": 1,
        "name": "occupied"
    })

    # Save COCO dataset to output file
    with open(save_path, "w") as outfile:
        json.dump(coco_dataset, outfile)


def polygon_area(poly):
    """Calculates the area of a polygon using the Shoelace formula"""
    n = len(poly)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return round(abs(area) / 2.0, 2)


def get_rotatedbox(roi):
    reg_box = _irregularbox_2_rectangularbox(roi)
    rotated_box = _corners2rotatedbbox(reg_box)
    return rotated_box


def _convert_points_2_8(W, H, rois):
    return torch.flatten(rois, start_dim=1)


def _irregularbox_2_rectangularbox(corners):
    rect = shapely.MultiPoint(corners).minimum_rotated_rectangle
    coords = [(round(x, 2), round(y, 2)) for x, y in rect.exterior.coords]
    return coords


def _corners2rotatedbbox(corners):
    centre = np.mean(np.array(corners), 0)
    theta = calc_bearing(corners[0], corners[1])
    theta = math.radians(theta)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    out_points = np.matmul(corners - centre, rotation) + centre
    x, y = list(out_points[0, :])
    # w, h = list(out_points[2, :] - out_points[0, :])
    # print(f"w , h : {w}, {h}")

    # Find the minimum and maximum x and y coordinates of the rotated vertices
    min_x = min(out_points[:, 0])
    max_x = max(out_points[:, 0])
    min_y = min(out_points[:, 1])
    max_y = max(out_points[:, 1])

    # Calculate the width and height of the rotated rectangle
    w = max_x - min_x
    h = max_y - min_y
    return [x, y, w, h, theta]


def calc_bearing(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    # Calculate the angle in radians using arctan2
    angle = math.atan2(x2 - x1, y2 - y1)

    # Convert the angle to degrees and normalize it to the range [-90, 90]
    angle_degrees = math.degrees(angle)

    # Return the angle in degrees
    return angle_degrees


if __name__ == '__main__':
    # Generate train ds
    fname_list, rois_list, occupancy_list = load_data("./data", 'train')
    convert2coco(fname_list, rois_list, occupancy_list, "./data/train.json")

    # Generate test ds
    fname_list, rois_list, occupancy_list = load_data("./data", 'valid')
    convert2coco(fname_list, rois_list, occupancy_list, "./data/valid.json")

    fname_list, rois_list, occupancy_list = load_data("./data", 'test')
    convert2coco(fname_list, rois_list, occupancy_list, "./data/test.json")

    # print(get_rotatedbox([(0,0),(1,1),(2,0.5)]))
