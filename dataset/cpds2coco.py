import json
import os
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

    # Loop through each image folder
    i = 0
    for fname, rois, labels in zip(file_names, rois_list, label_list):

        # Open image and get its width and height
        with Image.open("./data/images/" + fname) as img:
            width, height = img.size

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

            # Add annotation information to COCO dataset
            coco_dataset["annotations"].append({
                "id": len(coco_dataset["annotations"]),
                "image_id": i,
                "category_id": category_id,
                "bbox": roi,  # TODO make it like (x, y, w, h, theta)
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
    return abs(area) / 2.0


if __name__=='__main__':
    # Generate train ds
    fname_list, rois_list, occupancy_list = load_data("./data", 'train')
    convert2coco(fname_list, rois_list, occupancy_list, "./data/train.json")

