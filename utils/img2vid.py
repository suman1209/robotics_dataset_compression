import os
import numpy as np
from utils.dataset_utils import OriginalDataset

def get_img2vid_size(data_path, fps=30, out_dir="./temp/", delete_tmp=True, image_format="frame_%d.png", only_h264_fast=False):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_size_list = [os.path.getsize(os.path.join(data_path, img)) for img in os.listdir(data_path) if img.endswith(".png")]

    total_img_size = sum(img_size_list) / 2 ** 20

    if only_h264_fast:
        os.system(f"ffmpeg -f image2 -framerate {fps} -i {os.path.join(data_path, image_format)} -c:v libx264 -preset veryslow -qp 0 {os.path.join(out_dir, 'output_lossless_h264_slow.mp4 -loglevel panic')}")
    else:
        os.system(f"ffmpeg -f image2 -framerate {fps} -i {os.path.join(data_path, image_format)} -c:v libx264 -preset ultrafast -qp 0 {os.path.join(out_dir, 'output_lossless_h264_fast.mp4 -loglevel panic')}")
        os.system(f"ffmpeg -f image2 -framerate {fps} -i {os.path.join(data_path, image_format)} -c:v libx264 -preset veryslow -qp 0 {os.path.join(out_dir, 'output_lossless_h264_slow.mp4 -loglevel panic')}")
        os.system(f"ffmpeg -f image2 -framerate {fps} -i {os.path.join(data_path, image_format)} -c:v libx265 -preset ultrafast -x265-params lossless=1 {os.path.join(out_dir, 'output_lossless_h265_fast.mp4 -loglevel panic')}")
        os.system(f"ffmpeg -f image2 -framerate {fps} -i {os.path.join(data_path, image_format)} -c:v libx265 -preset veryslow -x265-params lossless=1 {os.path.join(out_dir, 'output_lossless_h265_slow.mp4 -loglevel panic')}")

    if only_h264_fast:
        h264_slow_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h264_slow.mkv")) / 2 ** 20
        h264_fast_size = 0
        h265_slow_size = 0
        h265_fast_size = 0
    else:
        h264_slow_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h264_slow.mkv")) / 2 ** 20
        h264_fast_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h264_fast.mkv")) / 2 ** 20
        h265_slow_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h265_slow.mp4")) / 2 ** 20
        h265_fast_size = os.path.getsize(os.path.join(out_dir, "output_lossless_h265_fast.mp4")) / 2 ** 20

    if delete_tmp:
        os.system(f"rm -r {out_dir}")
        # os.system(f"rm {os.path.join(out_dir, "output_lossless_h264_slow.mkv")}")
        # os.system(f"rm {os.path.join(out_dir, "output_lossless_h264_fast.mkv")}")
        # os.system(f"rm {os.path.join(out_dir, "output_lossless_h265_slow.mp4")}")
        # os.system(f"rm {os.path.join(out_dir, "output_lossless_h265_fast.mp4")}")

    # print(f"{total_img_size=}")
    # print(f"{h264_slow_size=}")
    # print(f"{h264_fast_size=}")
    # print(f"{h265_slow_size=}")
    # print(f"{h265_fast_size=}")
    return {
        "total_img_size": total_img_size,
        "h264_slow_size": h264_slow_size,
        "h264_fast_size": h264_fast_size,
        "h265_slow_size": h265_slow_size,
        "h265_fast_size": h265_fast_size,
    }


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def mean_square_error(vec1, vec2):
    mse = np.mean((vec1 - vec2) ** 2)
    return mse


def diff_num_zeros(vec1, vec2):
    diff_vec = vec1 - vec2
    diff = 0
    non_zero = int(np.count_nonzero(diff_vec))
    # print(f"{non_zero=}")
    # print(f"{int(diff_vec.sum())=}")
    diff = non_zero - (int(diff_vec.sum())/ non_zero)
    return diff


def compress_size_metric(data_path, vec1_id, vec2_id):
    os.mkdir('temp_com')
    os.system(f"cp {os.path.join(data_path, f'idx_{vec1_id}.png')} temp_com/frame_0.png")
    os.system(f"cp {os.path.join(data_path, f'idx_{vec2_id}.png')} temp_com/frame_1.png")
    img2vid_size = get_img2vid_size("temp_com/", only_h264=True)
    os.system("rm -r temp_com")
    return img2vid_size['h264_slow_size']


def shuffel_img(data_path, out_dir, sim_type='cosine'):
    original_dataset = OriginalDataset('../datasets/droid_100_sample_pictures')
    len_ = (original_dataset.__len__())
    # print(len_)
    if sim_type == 'cosine':
        sim_func = cosine_similarity
    elif sim_type == 'mse':
        sim_func = mean_square_error
    elif sim_type == 'zero_diff':
        sim_func = diff_num_zeros
    elif sim_type == 'compress_size':
        sim_func = compress_size_metric
    else:
        raise ValueError("Invalid similarity type. Please choose from 'cosine', 'mse', 'zero_diff'")

    img_list = [i for i in range(len_)]
    img_order = []
    current_img = 0
    img_prv = None
    img_order.append(current_img)
    for i in range(len_-1):
        print('current_img: ', current_img)
        img_list.remove(current_img)
        sim_list = []
        if i != 0:
            img_prv = original_dataset[img_order[-1]]
            img_prv = img_prv.flatten()
        else:
            pass
        for img_idx in img_list:
            img1 = original_dataset[current_img]
            img2 = original_dataset[img_idx]
            img1 = img1.flatten()
            img2 = img2.flatten()
            if sim_type == 'compress_size':
                score = sim_func(data_path, current_img, img_idx)
            else:
                score = sim_func(img1, img2)
            sim_list.append((score, img_idx))

        if sim_type == 'cosine':
            sorted_sim_list = sorted(sim_list, key=lambda x: x[0], reverse=True)
        else:
            sorted_sim_list = sorted(sim_list, key=lambda x: x[0], reverse=False)
        print('sorted_sim_list: ', sorted_sim_list[:2])
        print(f"Most similar img for image {current_img} is {sorted_sim_list[0][1]}\n")
        current_img = sorted_sim_list[0][1]
        img_order.append(current_img)

    if os.path.exists(out_dir):
        os.system(f'rm -rf {out_dir}')
        os.mkdir(out_dir)
        print(f'removed and created {out_dir}')
    else:
        os.mkdir(out_dir)
        print(f'created {out_dir}')

    for i, img_idx in enumerate(img_order):
        # print(f"Image {i}: {img_idx}")
        os.system(f'cp {os.path.join(data_path, f"idx_{img_idx}.png")} {os.path.join(out_dir, f"idx_{i}.png")}')
        # os.system(f'mv {img_reorder_path}idx_{img_idx}.png {img_reorder_path}idx_new_{i}.png')


    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    # img_list = [img for img in os.listdir(data_path) if img.endswith(".png")]

    # for i, img in enumerate(img_list):
    #     os.system(f"cp {os.path.join(data_path, img)} {os.path.join(out_dir, image_format % i)}")

    return True


if __name__ == "__main__":
    ori_order = get_img2vid_size("./datasets/droid_100_sample_pictures/", image_format="idx_%d.png")
    # reorder = get_img2vid_size("./datasets/droid_100_sample_pictures_reorder/", image_format="idx_new_%d.png", delete_tmp=False)
    # print(f"{ori_order=}")
    # print(f"{reorder=}")

    # ori_order = get_img2vid_size("/home/suman/Desktop/robotics_dataset_compression/datasets/chips_droid/")
