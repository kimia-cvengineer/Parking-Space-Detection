# Image-Based Parking Space Detection and Occupancy Classification

In this repository, we provide:
- code to reproduce all of our results
- ACPDS [paper](https://arxiv.org/pdf/2107.12207.pdf) and [repo](https://github.com/martin-marek/parking-space-occupancy)
- download links for the [ACPDS dataset](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip), [training logs](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Fpaper_training_output.zip), and [model weights](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2FRCNN_128_square_gopro.pt)
- Colab notebooks to [explore the dataset and models](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/model_playground.ipynb), [train a model](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/train.ipynb), and [plot the training logs](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/train_log_analysis.ipynb)

# Datasets

1. ACPDS
The dataset contains 293 images captured at a roughly 10-meter height using a GoPro Hero 6 camera. Here is a sample from the dataset:

![alt text](/Modules/Space/illustrations/dataset_sample.jpg)

2. Custom dataset
Combination of ACPDS and online resourses. Here is a sample from the dataset:
![alt text](/Modules/Mark/illustrations/dataset_sample.png)

# Inference

Here's a minimal example to run inference on a trained model. For more, please see the [demo notebook](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/model_playground.ipynb).

```python
import torch, os, requests
from models.rcnn import RCNN
from utils_funcs import transforms

device = torch.device('cpu')

# create model
model = MaskRCNN(weights='model_weights', device=device)

# load model weights
weights_path = 'weights.pt'
if not os.path.exists(weights_path):
    r = requests.get('https://storage.googleapis.com/pd-models/RCNN_128_square_gopro.pt')
    with open(weights_path, 'wb') as f:
        f.write(r.content)
model.load_state_dict(torch.load(weights_path, map_location='cpu'))

# inference
img_path = 'dataset/data/images/GOPR6543.JPG'
img = read_image(img_path)
preds = predict(model=model, img_path=img_path, device=device)

# predictions visualization
# Draw predicted masks
vis.show_mask_predictions([img], preds, score_threshold=.4)

# Draw predicted boxes
vis.show_box_predictions([img], preds, score_threshold=.4, box_width=15)
```

# Training

To reproduce our full results from the paper, please run the [train_all_models](train_all_models.py) script locally. To train just a single model, please use the provided [Colab notebook](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/train.ipynb) – Google Colab is sufficient for this.
