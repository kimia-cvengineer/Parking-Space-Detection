import json
import math

from matplotlib import pyplot as plt, patches

from PIL import Image

with open("./data/valid.json", 'r') as f:
    anns = json.load(f)


images = anns['images']
annotation = anns['annotations']

for idx in range(0, len(images)):
    print("idx : ", idx)
    # idx = 12
    img = Image.open(f"./data/images/{images[idx]['file_name']}")
    W, H = img.size
    boxes = []
    for ann in annotation:
        if ann['image_id'] == idx:
            boxes.append(ann['bbox'])
    # print("box : ", box)
    # box = [3045.677407881576, 2577.8249571958895, -1085.9948157631507, -496.2399143917796, 2.7189330877965925]


    point = [(2901.4444575, 2774.003024), (1905.6934575, 2326.15235733), (2151.6319575 , 1952.277024), (3039.4099575, 2306.159024 )]

    fig, ax = plt.subplots(figsize=[12, 8])
    # fig.set_size_inches(5, 5)
    ax.imshow(img)
    ax.annotate(images[idx]['file_name'], (W-350, H-250))

    # for box in (target['boxes']):
    # x, y, width, height, theta = box[0], box[1], box[2], box[3], box[4]
    for box in boxes:
        x, y, width, height = box[0], box[1], box[2], box[3]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none', rotation_point='center')
                                 # , angle= -math.degrees(theta))
        ax.add_patch(rect)
        ax.annotate(f"x, y = ({x:.2f},{y:.2f}", (x, y), color='black', weight='bold', fontsize=7)
        # ax.annotate(f"w, h = ({width:.2f},{height:.2f}", (x+width/2, y+height/2), color='black', weight='bold', fontsize=7, ha='center', va='center')

    poly = patches.Polygon(point, edgecolor='g', facecolor='none', linewidth=2)

    # print(math.degrees(theta))

    # Draw the bounding box on top of the image

    # a.add_patch(poly)

    # print("area = ", target['area'])
    # print("min area = ", torch.min(target['area']))
    # print("max area = ", torch.max(target['area']))
    plt.show()

