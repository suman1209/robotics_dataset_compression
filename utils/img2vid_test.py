import os

fps = 30
dataset_path = "./dataset/droid_100_sample_pictures/"
image_format = """idx_%d.png"""
out_dir = "./dataset/"

img_list = [img for img in os.listdir(dataset_path) if img.endswith(".png")]

total_size = 0
for img in img_list:
    total_size += os.path.getsize(os.path.join(dataset_path, img))

print(f"{total_size=}")

os.system("ls")

# os.system(f"ffmpeg -f image2 -framerate {fps} -i {dataset_path}{image_format} -c:v libx265 {out_dir}output.mp4")

# os.system(f"ffmpeg -f image2 -framerate {fps} -i {dataset_path}{image_format} -c:v libx264 -preset veryslow -qp 0 {out_dir}output_lossless_h264.avi")

os.system(f"ffmpeg -f image2 -framerate {fps} -i {dataset_path}{image_format} -c:v libx264 -preset ultrafast -qp 0 {out_dir}output_lossless_h264_fast.mkv")
os.system(f"ffmpeg -f image2 -framerate {fps} -i {dataset_path}{image_format} -c:v libx264 -preset veryslow -qp 0 {out_dir}output_lossless_h264_slow.mkv")
os.system(f"ffmpeg -f image2 -framerate {fps} -i {dataset_path}{image_format} -c:v libx265 -preset ultrafast -x265-params lossless=1 {out_dir}output_lossless_h265_fast.mp4")
os.system(f"ffmpeg -f image2 -framerate {fps} -i {dataset_path}{image_format} -c:v libx265 -preset veryslow -x265-params lossless=1 {out_dir}output_lossless_h265_slow.mp4")

# os.system("""ffmpeg -framerate 30 -i image_%03d.png -f yuv4mpegpipe -pix_fmt yuv420p - | vvc_encoder_app -c lossless.cfg -i - -o output_lossless_h266.bin""")
h264_slow_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h264_slow.mkv"))
h264_fast_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h264_fast.mkv"))
h265_slow_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h265_slow.mp4"))
h265_fast_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h265_fast.mp4"))
print(f"{total_size=}")
print(f"{h264_slow_size=}")
print(f"{h264_fast_size=}")
print(f"{h265_slow_size=}")
print(f"{h265_fast_size=}")
