# Robotics Dataset Compression 
## Project Overview
Robotics datasets during training often contain repeated information, with only certain parts of data changing. In this project we aim to find ways to compress sequential robotic datasets, we aim to compress sequential images of a robotic arm performing pick-and-place action. Given that these sequences contain significant temporal redundancy—where only the robotic arm's movement changes and the background remains constant—compression techniques can significantly reduce data size. We first convert the images to tensors of size (180,320,3) where 3 refers to the RGB channels and then apply Frame of Reference(FOR) and Delta encoding for lossless compression.
## How to use the tool
First clone the repo using:
```
git clone https://github.com/suman1209/robotics_dataset_compression.git
```
Open the terminal ( command prompt) and go to the directory
```
cd robotics_dataset_compression
```

Install required packages
```
python -m pip install -r requirement.txt
```

To run main.py
```
python main.py
```
To measure the performance of compression methods
```
python compare_compression_algo.py
```

## Results

## Architecture
The system starts with a raw dataset, which is processed by the RDC tool to produce a compressed dataset. Within the tool, tensor storage is used to manage data, where an original dataset is parsed through an encoding scheme (such as FOR/DELTA). The tensor storage interacts with a sparse representation module, which can convert delta images into sparse representations and vice versa.
![Archiitecture of RDC](https://github.com/suman1209/robotics_dataset_compression/blob/akash/datasets/Results/rdc_architecture.jpeg)
## Future Works