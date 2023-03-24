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

# Inference and Visualization

To run inference on the trained model and get the prediction, run [main](main.py). It gets the model predictions given an image and draw predictions on the images to visualize the outputs. 

| ![alt text](/Illustrations/prediction_visualiztion_sample_img1.png) | ![alt text](/Illustrations/prediction_visualiztion_sample_img2.png) |
| ------------ | ------------ |


# Training

To reproduce our full results or further improve the models performance, please refer to the modules '''README.md''' files.

# Citation

```bibtex
@misc{marek2021imagebased,
      title={Image-Based Parking Space Occupancy Classification: Dataset and Baseline}, 
      author={Martin Marek},
      year={2021},
      eprint={2107.12207},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
