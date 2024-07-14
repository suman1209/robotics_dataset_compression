import cv2


class Video2Image:
    """This is a utility class to convert a video into an image."""
    def __init__(self, video_path):
        self.video_path = video_path

    def save_images(self, directory: str):
        cap = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        i = 0
        while (cap.isOpened()):
            filename = directory + str(i) + ".jpg"
            ret, frame = cap.read()
            if ret == True:
                # write the flipped frame
                cv2.imwrite(filename, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # Release everything if job is finished
        cap.release()


if __name__ == "__main__":
    video_path = "../datasets/2023-01-23-01_14_55_color_image3.mp4"
    vd = Video2Image(video_path)
    vd.save_images(directory='../datasets/sample_images')



