import os

def get_img2vid_size(data_path, fps=30, out_dir="./temp/", delete_tmp=True, image_format="frame_%d.png"):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_size_list = [os.path.getsize(os.path.join(data_path, img)) for img in os.listdir(data_path) if img.endswith(".png")]

    total_img_size = sum(img_size_list)

    # os.system(f"ffmpeg -f image2 -framerate {fps} -i {dataset_path}{image_format} -c:v libx265 {out_dir}output.mp4")
    # os.system(f"ffmpeg -f image2 -framerate {fps} -i {dataset_path}{image_format} -c:v libx264 -preset veryslow -qp 0 {out_dir}output_lossless_h264.avi")

    os.system(f"ffmpeg -f image2 -framerate {fps} -i {data_path}{image_format} -c:v libx264 -preset ultrafast -qp 0 {out_dir}output_lossless_h264_fast.mkv")
    os.system(f"ffmpeg -f image2 -framerate {fps} -i {data_path}{image_format} -c:v libx264 -preset veryslow -qp 0 {out_dir}output_lossless_h264_slow.mkv")
    os.system(f"ffmpeg -f image2 -framerate {fps} -i {data_path}{image_format} -c:v libx265 -preset ultrafast -x265-params lossless=1 {out_dir}output_lossless_h265_fast.mp4")
    os.system(f"ffmpeg -f image2 -framerate {fps} -i {data_path}{image_format} -c:v libx265 -preset veryslow -x265-params lossless=1 {out_dir}output_lossless_h265_slow.mp4")

    # os.system("""ffmpeg -framerate 30 -i image_%03d.png -f yuv4mpegpipe -pix_fmt yuv420p - | vvc_encoder_app -c lossless.cfg -i - -o output_lossless_h266.bin""")
    h264_slow_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h264_slow.mkv"))
    h264_fast_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h264_fast.mkv"))
    h265_slow_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h265_slow.mp4"))
    h265_fast_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h265_fast.mp4"))

    if delete_tmp:
        os.system(f"rm -r {out_dir}")
        # os.system(f"rm {os.path.join(out_dir, "output_lossless_h264_slow.mkv")}")
        # os.system(f"rm {os.path.join(out_dir, "output_lossless_h264_fast.mkv")}")
        # os.system(f"rm {os.path.join(out_dir, "output_lossless_h265_slow.mp4")}")
        # os.system(f"rm {os.path.join(out_dir, "output_lossless_h265_fast.mp4")}")

    print(f"{total_img_size / 10** 6=}")
    print(f"{h264_slow_size / 10** 6=}")
    print(f"{h264_fast_size / 10** 6=}")
    print(f"{h265_slow_size / 10** 6=}")
    print(f"{h265_fast_size / 10** 6=}")
    return {
        "total_img_size": total_img_size,
        "h264_slow_size": h264_slow_size,
        "h264_fast_size": h264_fast_size,
        "h265_slow_size": h265_slow_size,
        "h265_fast_size": h265_fast_size,
    }


if __name__ == "__main__":
    # ori_order = get_img2vid_size("./datasets/droid_100_sample_pictures/")
    # reorder = get_img2vid_size("./datasets/droid_100_sample_pictures_reorder/", image_format="idx_new_%d.png", delete_tmp=False)
    # print(f"{ori_order=}")
    # print(f"{reorder=}")

    ori_order = get_img2vid_size("/home/suman/Desktop/robotics_dataset_compression/datasets/chips_droid/")
