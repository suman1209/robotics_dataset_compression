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
To run main.py and measure the performance of compression methods
```
python main.py
```
## Results
Performance of our compression algo on Dataset 1
![Dataset 1 Results](https://github.com/suman1209/robotics_dataset_compression/blob/akash/datasets/Results/dataset1.png)

Performance of our compression algo on Dataset 1 converted to gray scale
![Dataset 1 gs Results](https://github.com/suman1209/robotics_dataset_compression/blob/akash/datasets/Results/dataset1_gs.png) 


## Architecture
In this architecture, we first load the dataset. Then, the dataset is parsed by the RDC tool. In this tool, we compress original data using one of the encoding schemes (FOR, delta). Then, we use the decompression method for the same encoding scheme to reconstruct the original data. 
![Archiitecture of RDC](https://github.com/suman1209/robotics_dataset_compression/blob/akash/datasets/Results/rdc_architecture.jpeg)
## Future Works
