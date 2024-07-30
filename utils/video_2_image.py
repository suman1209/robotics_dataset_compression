import cv2
import os

class Video2Image:
    """This is a utility class to convert a video into images."""
    
    def __init__(self, video_path):
        self.video_path = video_path

    def save_images(self, directory: str):
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if no frame is captured

            filename = os.path.join(directory, f"frame_{i:04d}.png")
            cv2.imwrite(filename, frame)
            i += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/priyam/UTN/Cloud Database/cdb/mygeneratedvideo_sorted.avi"
    vd = Video2Image(video_path)
    vd.save_images(directory='/home/priyam/UTN/Cloud Database/cdb/mygeneratedvidoe_sorted_frames')
