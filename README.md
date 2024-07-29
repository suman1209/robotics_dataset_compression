# Robotics Dataset Compression 
## Project Overview
Robotics datasets during training often contain repeated information, with only certain parts of data changing. In this project we aim to find ways to compress sequential robotic datasets, we aim to compress sequential images of a robotic arm performing pick-and-place action. Given that these sequences contain significant temporal redundancy—where only the robotic arm's movement changes and the background remains constant—compression techniques can significantly reduce data size. We first convert the images to tensors of size (180,320,3) where 3 refers to the RGB channels and then apply Frame of Reference(FOR) and Delta encoding for lossless compression.
## How to use the tool
First clone the repo using:
```
git clone https://github.com/suman1209/robotics_dataset_compression.git
```
Open terminal and go to directory
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


