# Image-Based Parking Space Detection and Occupancy Classification

<!-- <p align="center"> -->
<img src="/Illustrations/PSDet%20Architecture.png" alt="Model Architecture" height="400">
<!-- </p> -->

In this repository, we provide:
- Codes to reproduce all of our results
- [Project report](Project%20Report.pdf) that describes all the methods used in the developement of this project 
- Links for the [ACPDS dataset](https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip)
- Links for the [Custom dataset](https://drive.google.com/file/d/1__tQI7GGbzt4KL0cv6gR6UmJyzVDQPoE/view?usp=sharing)


## Introduction

   With the increasing number of vehicles on the road, parking spaces have become scarce resources in urban areas, and it can be challenging to find available spots, especially for people with disabilities. Automatic parking space detection can not only facilitate the process of looking for an available parking spot but also help drivers save time and effectively reduce emissions as well as traffic congestion by navigating directly to the pre-suggested spot.
   
   In parking lots, there are some spaces allocated to people with disabilities to improve their accessibility to the spots. To help them automatically find those spaces, we proposed an image-based smart parking system that looks for available regular and accessible parking spots. Thus, this project highlights the importance of developing automatic parking systems with special attention to accessible parking spots in order to create a more equitable and accessible urban environment.

## Methodology

We split the project into three main tasks. 
* Parking space detection

   - Detects the location of each parking spot in a parking lot
   - Classifies the occupancy of each spot
   
* Handicap mark detection

   - Finds the location of the painted handicap marks on the ground
   
* Final detector

   - Combines outputs of parking space and handicap mark detectors
   - Outputs regular and accessible parking spot along with their occupancy


## Datasets

* ACPDS:
The dataset contains 293 images captured at a roughly 10-meter height using a GoPro Hero 6 camera ([paper](https://arxiv.org/pdf/2107.12207.pdf) and [repo](https://github.com/martin-marek/parking-space-occupancy)).

   Here is a sample from the dataset:

<img src="/Modules/Space/illustrations/dataset_sample.jpg" width="500" alt="alt_text">

* Custom dataset:
Combination of ACPDS and online resourses. 

   Here is a sample from the dataset:

<img src="/Modules/Mark/illustrations/dataset_sample.png" width="500" alt="alt_text">

## Results

* Parking Space Detector 

| Metric  | IoU | Valid Box IoU | Valid Segm IoU | Test Box IoU | Test Segm IoU | 
| --- | --- | --- | --- | --- | --- | 
| mAP | 0.5 | 0.622 | 0.563 | 0.537 | 0.473 | 
| mAP  | 0.5:0.95 | 0.334 | 0.300 | 0.260 | 0.229 | 
| mAR  | 0.5:0.95 | 0.421 | 0.368 | 0.339 | 0.291 | 

* Handicap Mark Detector 

| Metric  | IoU | Valid Box IoU | Test Box IoU | 
| --- | --- | --- | --- | 
| mAP | 0.5 | 0.830 | 0.732 | 
| mAP  | 0.5:0.95 | 0.617 | 0.501 | 
| mAR  | 0.5:0.95 | 0.702 | 0.573 |


## Inference and Visualization

To run inference on the trained model and get the prediction, run [main.py](main.py). It gets the model predictions given an image and draw predictions on the images to visualize the outputs. 

| ![alt text](/Illustrations/prediction_visualiztion_sample_img1.png) | ![alt text](/Illustrations/prediction_visualiztion_sample_img2.png) |
| ------------ | ------------ |


## Training

To reproduce our full results or further improve the models performance, please refer to the modules *README.md* files.

# Citation

```bibtex
@misc{kimiayuchen2023imagebased,
      title={Image-Based Parking Space Detection and Occupancy Classification}, 
      authors={Kimia Afshari, Yuchen Hou},
      year={2023}
}
```
