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

Performance of our compression algo on Dataset 1 converted to grayscale
![Dataset 1 gs Results](https://github.com/suman1209/robotics_dataset_compression/blob/akash/datasets/Results/dataset1_gs.png) 


Performance of our compression algo on Dataset 2
![Dataset 2 Results](https://github.com/suman1209/robotics_dataset_compression/blob/akash/datasets/Results/result_d2.png)

## Architecture
In this architecture, First, we load the dataset. Then, the dataset is parsed by the RDC tool. In this tool, First, we compress original data using one of the encoding schemes (FOR, delta). Then we convert it to a sparse representation. To reconstruct the image, we decompress the data using decoding of the applied encoding scheme.
![Archiitecture of RDC](https://github.com/suman1209/robotics_dataset_compression/blob/akash/datasets/Results/rdc_architecture.jpeg)
## Future Works
 - Improving current compression algorithm (for both RGB and Grayscale Datasets)
 - Trying different encoding techniques
 - Testing our compression algorithm on different datasets
