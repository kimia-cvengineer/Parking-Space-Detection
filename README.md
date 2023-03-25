# Image-Based Parking Space Detection and Occupancy Classification

<!-- <p align="center"> -->
<img src="/Illustrations/PSDet%20Architecture.png" alt="Model Architecture" height="400">
<!-- </p> -->

In this repository, we provide:
- code to reproduce all of our results
- project report hat describes all the methods used in the developement of this project 
- download links for the [ACPDS dataset](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip)
- download links for the [Custom dataset](https://drive.google.com/file/d/1__tQI7GGbzt4KL0cv6gR6UmJyzVDQPoE/view?usp=sharing)
- Colab notebooks to [explore the dataset and models](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/model_playground.ipynb), [train a model](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/train.ipynb), and [plot the training logs](https://colab.research.google.com/github/martin-marek/parking-space-occupancy/blob/main/notebooks/train_log_analysis.ipynb)


# Introduction

   With the increasing number of vehicles on the road, parking spaces have become scarce resources in urban areas, and it can be challenging to find available spots, especially for people with disabilities. Automatic parking space detection can not only facilitate the process of looking for an available parking spot but also help drivers save time and effectively reduce emissions as well as traffic congestion by navigating directly to the pre-suggested spot.
   
   In parking lots, there are some spaces allocated to people with disabilities to improve their accessibility to the spots. To help them automatically find those spaces, we proposed an image-based smart parking system that looks for available regular and accessible parking spots. Thus, this project highlights the importance of developing automatic parking systems with special attention to accessible parking spots in order to create a more equitable and accessible urban environment.


# Datasets

1. ACPDS:
The dataset contains 293 images captured at a roughly 10-meter height using a GoPro Hero 6 camera ([paper](https://arxiv.org/pdf/2107.12207.pdf) and [repo](https://github.com/martin-marek/parking-space-occupancy)).

Here is a sample from the dataset:

<img src="/Modules/Space/illustrations/dataset_sample.jpg" width="500" alt="alt_text">

2. Custom dataset:
Combination of ACPDS and online resourses. 

Here is a sample from the dataset:

<img src="/Modules/Mark/illustrations/dataset_sample.png" width="500" alt="alt_text">

# Inference and Visualization

To run inference on the trained model and get the prediction, run [main.py](main.py). It gets the model predictions given an image and draw predictions on the images to visualize the outputs. 

| ![alt text](/Illustrations/prediction_visualiztion_sample_img1.png) | ![alt text](/Illustrations/prediction_visualiztion_sample_img2.png) |
| ------------ | ------------ |


# Training

To reproduce our full results or further improve the models performance, please refer to the modules *README.md* files.

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
