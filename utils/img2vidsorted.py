import os
import cv2
from PIL import Image
import re

# Function to extract the numerical part from filenames for sorting
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else -1

# Image folder path
image_folder = "/home/priyam/UTN/Cloud Database/cdb/droid_100_sample_pictures"
# Video output path
output_folder = "/home/priyam/UTN/Cloud Database/cdb"
output_video_name = 'mygeneratedvideo_sorted.avi'

# Get list of images and sort them by the number in the filename
images = sorted(
    [img for img in os.listdir(image_folder) if img.endswith(".png")],
    key=extract_number
)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Reading the first image to get dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define video codec and create VideoWriter object
video_path = os.path.join(output_folder, output_video_name)
codec = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_path, codec, 40, (width, height))

# Write each image to the video
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    video.write(frame)

# Release the video writer and clean up
cv2.destroyAllWindows()
video.release()

print(f"Video saved at {video_path}")
