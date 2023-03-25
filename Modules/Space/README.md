# Image-Based Parking Space Detection and Occupancy Classification

In this repository, we provide:
- code to reproduce all of our results
- ACPDS [paper](https://arxiv.org/pdf/2107.12207.pdf) and [repo](https://github.com/martin-marek/parking-space-occupancy)
- download links for the [ACPDS dataset](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip), [training logs](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Fpaper_training_output.zip), and [model weights](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2FRCNN_128_square_gopro.pt)
- Colab notebooks to [explore the dataset and models](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/model_playground.ipynb), [train a model](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/train.ipynb), and [plot the training logs](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/train_log_analysis.ipynb)

# Introcudtion

In the aerial view of parking lots, parking spots are small and densely packed, resulting in the foreground and background class imbalance. In addition, due to the complexity and irregular shape of parking spaces, working with well-aligned bounding boxes resulted in a high overlap between each spot and its adjacents, making detection a challenging task. To address this issue, we used the Mask R-CNN model to consider segmentation masks. This improves the detection performance, because by using only Faster R-CNN, we got the x- and y-axis aligned bounding boxes which includes a larger area around each spot. However, the actual spots are not well aligned and have orientations with respect to the x and y axes. So, predicting masks for parking spots yields better performance.


# Model Builder

We support two model builders relying on MaskRCNN. To boost the performance and prevent from overfitting, we trained the model with pre-trained weights.

* Mask R-CNN Resnet50_FPN

  Mask R-CNN model with a ResNet-50-FPN backbone from the Mask R-CNN paper.

* Mask R-CNN Resnet50_FPN_V2

  Improved Mask R-CNN model with a ResNet-50-FPN backbone from the Benchmarking Detection Transfer Learning with Vision Transformers paper.


# Dataset

ACPDS
The dataset contains 293 images captured at a roughly 10-meter height using a GoPro Hero 6 camera. Here is a sample from the dataset:

<img src="/Modules/Space/illustrations/dataset_sample.jpg" width="500">

# Inference

Here's an example to run inference on a trained model and visualize the model predictions on an image. For more, please see the [demo notebook](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/model_playground.ipynb).

```python
from inference import get_mask_rcnn_model as MaskRCNN
from utils_funcs import visualize as vis
from inference import predict

device = torch.device('cpu')

# create model
model = MaskRCNN(weights='model_weights', device=device)

# inference
img_path = 'dataset/data/images/GOPR6543.JPG'
img = read_image(img_path)
preds = predict(model=model, img_path=img_path, device=device)

# Predictions visualization
# Draw predicted masks
vis.show_mask_predictions([img], preds, score_threshold=.4)

# Draw predicted boxes
vis.show_box_predictions([img], preds, score_threshold=.4, box_width=15)
```

# Training

Models were trained and evaluated on 12th Gen Intel(R) Core(TM) i9-12900K CPU, 64 GB RAM, and Nvidia **GeForce RTX 3060** graphics card. 

Here's an example to train a model given ACPDS dataset. You can also use the provided [Colab notebook](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/train.ipynb).

```python
from dataset import acpds
from utils_funcs.engine import train_model
from models.rcnn_fpn import create_model

# load dataset
train_ds, valid_ds, test_ds = acpds.create_datasets('dataset/data', 1, res=1800)

# train model
model = create_model()
train_model(, train_ds, valid_ds, test_ds, out_dir, device, epochs=30, lr=0.00008)

torch.cuda.empty_cache()
```

