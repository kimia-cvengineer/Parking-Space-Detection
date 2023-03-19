import json

from matplotlib import pyplot as plt, patches

from PIL import Image

with open("./data/train.json", 'r') as f:
    anns = json.load(f)

images = anns['images']
annotation = anns['annotations']

for idx in range(0, len(images)):
    print("idx : ", idx)
    # idx = 74
    img = Image.open(f"./data/images/{images[idx]['file_name']}")
    W, H = img.size
    boxes = []
    masks = []
    for ann in annotation:
        if ann['image_id'] == idx:
            boxes.append(ann['bbox'])
            masks.append(ann['segmentation'])

    point = [(2901.4444575, 2774.003024), (1905.6934575, 2326.15235733), (2151.6319575, 1952.277024),
             (3039.4099575, 2306.159024)]

    fig, ax = plt.subplots(figsize=[12, 8])
    # fig.set_size_inches(5, 5)
    ax.imshow(img)
    ax.annotate(images[idx]['file_name'], (W - 350, H - 250))

    for mask, box in zip(masks, boxes):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        mask = mask[0]
        # Draw the bounding box on top of the image
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none', rotation_point='center')
        # , angle= -math.degrees(theta))
        ax.add_patch(rect)
        ax.annotate(f"x, y = ({x:.2f},{y:.2f}", (x, y), color='black', weight='bold', fontsize=7)
        # ax.annotate(f"w, h = ({width:.2f},{height:.2f}", (x+width/2, y+height/2), color='black', weight='bold',
        # fontsize=7, ha='center', va='center') Draw the quadrilateral on top of the image
        poly = patches.Polygon([(mask[0], mask[1]), (mask[2], mask[3]),
                                (mask[4], mask[5]), (mask[6], mask[7])], edgecolor='g', linewidth=2, facecolor='none')
        poly2 = patches.Polygon(point, facecolor='none', edgecolor='b', linewidth=2)
        # ax.add_patch(poly)
        # ax.add_patch(poly2)

    # print("area = ", target['area'])
    # print("min area = ", torch.min(target['area']))
    # print("max area = ", torch.max(target['area']))
    plt.show()
