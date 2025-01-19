import random
import cv2
import numpy as np
import torch
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import logging
from torch import Tensor
from jaxtyping import Float

# 在主进程中设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logfile.txt',  # 指定日志文件名
    filemode='a',
)


def prepare_frames(video_file):
    # 创建视频捕捉对象
    cap = cv2.VideoCapture(video_file)

    # 获取视频的宽度和高度
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 检查视频是否成功打开
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_file}")

    frames = []

    while True:
        # 逐帧读取
        ret, frame = cap.read()

        # 如果没有读取到帧，则退出循环
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # 释放视频捕捉对象
    cap.release()

    return frames, width, height


def prepare_camera_poses(camera_pose_file):
    whole_camera_para = []

    with open(camera_pose_file, 'r', encoding='utf-8') as file:
        # 读取所有行
        lines = file.readlines()

        title = lines[0].strip()

        # 确保文件至少有两行
        if len(lines) < 2:
            logging.info("文件内容不足两行，无法读取数据。")
            return whole_camera_para

        # 跳过第一行，从第二行开始处理
        for idx, line in enumerate(lines[1:], start=2):
            # 去除首尾空白字符并按空格分割
            parts = line.strip().split()

            # 检查每行是否有19个数字
            if len(parts) != 19:
                logging.info(f"警告：第 {idx} 行的数字数量不是19，跳过该行。")
                continue

            try:
                # 将字符串转换为浮点数
                numbers = [float(part) for part in parts]
                whole_camera_para.append(numbers)
            except ValueError as ve:
                logging.info(f"警告：第 {idx} 行包含非数字字符，跳过该行。错误详情: {ve}")
                continue

    return title, whole_camera_para


def get_projection_matrix(fovy: Float[Tensor, 'B'], aspect_wh: float, near: float, far: float) -> Float[Tensor, 'B 4 4']:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(fovy / 2.0)  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_src_proj_mtx(focal_length_x_norm, focal_length_y_norm, height, width, res, src_image):
    """
    根据相机内参和图像处理步骤计算投影矩阵。

    参数:
    - focal_length_x_norm (float): 归一化的x方向焦距 (fx / width)
    - focal_length_y_norm (float): 归一化的y方向焦距 (fy / height)
    - height (int): 原始图像高度
    - width (int): 原始图像宽度
    - res (int): 图像缩放后的尺寸 (res, res)
    - src_image (torch.Tensor): 源图像张量，用于确定设备类型

    返回:
    - src_proj_mtx (torch.Tensor): 投影矩阵，形状为 (1, 4, 4)
    """
    # 将归一化焦距转换为像素单位
    focal_length_x = focal_length_x_norm * width
    focal_length_y = focal_length_y_norm * height

    # 裁剪为中心正方形
    cropped_size = min(width, height)
    scale_crop_x = cropped_size / width
    scale_crop_y = cropped_size / height

    # 调整焦距以适应裁剪后的图像
    focal_length_x_cropped = focal_length_x * scale_crop_x
    focal_length_y_cropped = focal_length_y * scale_crop_y

    # 缩放图像
    scale_resize = res / cropped_size
    focal_length_x_resized = focal_length_x_cropped * scale_resize
    focal_length_y_resized = focal_length_y_cropped * scale_resize

    # 计算垂直视场角 (fovy) 使用调整后的焦距和缩放后的高度
    fovy = 2.0 * torch.atan(torch.tensor(res / (2.0 * focal_length_y_resized)))
    fovy = fovy.unsqueeze(0)  # 形状调整为 (1,)

    near, far = 0.01, 100.0
    aspect_wh = 1.0  # 因为图像被缩放为正方形 (res, res)

    # 获取投影矩阵
    src_proj_mtx = get_projection_matrix(fovy=fovy, aspect_wh=aspect_wh, near=near, far=far).to(src_image)

    return src_proj_mtx


def convert_camera_extrinsics(w2c):
    # 获取设备和数据类型，以确保缩放矩阵与w2c在同一设备和数据类型
    device = w2c.device
    dtype = w2c.dtype

    # 定义缩放矩阵，x和y轴取反，z轴保持不变
    S = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=dtype))

    # 将缩放矩阵应用于旋转和平移部分
    R = w2c[:, :3]  # 3x3
    t = w2c[:, 3]  # 3

    new_R = S @ R  # 矩阵乘法
    new_t = S @ t  # 向量乘法

    # 构建新的外参矩阵
    new_w2c = torch.cat((new_R, new_t.unsqueeze(1)), dim=1)  # 3x4

    return new_w2c


def get_rel_view_mtx(src_wc, tar_wc, src_image):
    src_wc = convert_camera_extrinsics(src_wc)
    tar_wc = convert_camera_extrinsics(tar_wc)

    # 将第一个 W2C 矩阵扩展为 4x4 齐次变换矩阵
    T1 = torch.eye(4, dtype=src_wc.dtype, device=src_wc.device)
    T1[:3, :3] = src_wc[:, :3]
    T1[:3, 3] = src_wc[:, 3]

    # 将第二个 W2C 矩阵扩展为 4x4 齐次变换矩阵
    T2 = torch.eye(4, dtype=tar_wc.dtype, device=tar_wc.device)
    T2[:3, :3] = tar_wc[:, :3]
    T2[:3, 3] = tar_wc[:, 3]

    # 计算第一个视图矩阵的逆
    T1_inv = torch.inverse(T1)

    # 计算相对视图矩阵
    rel_view_mtx = T2 @ T1_inv

    return rel_view_mtx.to(src_image)


def pre_process_frames_poses(frames, camera_poses, output_frames):
    total_frames = len(frames)

    # 计算有效采样区域的起始和结束帧数
    start_drop = int(0.1 * total_frames)
    end_drop = int(0.9 * total_frames)
    valid_length = end_drop - start_drop  # 有效采样区域的总帧数

    if valid_length <= 0:
        return False, frames, camera_poses

    # 动态计算 stride，使其在1到3之间，并尽可能接近 video_sample_n_frames
    # 尝试从 stride=3 开始，如果无法满足，则减小 stride
    for stride in range(3, 0, -1):
        possible_max_frames = (valid_length + stride - 1) // stride  # 向上取整
        if possible_max_frames >= output_frames:
            chosen_stride = stride
            break
    else:
        # 如果 stride=1 也无法满足，则选择 stride=1 并尽可能采样多的帧
        chosen_stride = 1

    # 计算实际可以采样的帧数
    possible_frames = (valid_length + chosen_stride - 1) // chosen_stride

    if possible_frames < output_frames:
        return False, frames, camera_poses

    min_sample_n_frames = min(output_frames, possible_frames)

    if min_sample_n_frames == 0:
        return False, frames, camera_poses

    # 计算片段的总长度
    clip_length = (min_sample_n_frames - 1) * chosen_stride + 1
    if clip_length > valid_length:
        clip_length = valid_length
        min_sample_n_frames = (clip_length + chosen_stride - 1) // chosen_stride

    # 随机选择起始索引
    if valid_length != clip_length:
        start_idx_lower = start_drop
        start_idx_upper = end_drop - clip_length
        if start_idx_upper < start_idx_lower:
            start_idx_upper = start_idx_lower  # 防止随机范围出现负值
        start_idx = random.randint(start_idx_lower, start_idx_upper)
    else:
        start_idx = start_drop

    # 生成采样帧的索引
    batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

    frames = [frames[i] for i in batch_index]
    camera_poses = [camera_poses[i] for i in batch_index]

    return True, frames, camera_poses


def main(
    video_file,
    camera_pose_file,
    output_frames,
    device,
    batch_size,
    schedule,
    lr,
    niter,
    model_name,
):

    # preprocess data
    frames, width, height = prepare_frames(video_file)
    title, camera_poses = prepare_camera_poses(camera_pose_file)
    well_done, frames, camera_poses = pre_process_frames_poses(frames, camera_poses, output_frames)

    if not well_done:
        return False

    if len(frames) != output_frames:
        return False

    if len(frames) != len(camera_poses):
        return False

    # define model
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    # load_images can take a list of images or a directory
    images = load_images(
        [
            'assets/770352d5c0066b2e/frame_000001.png',
            'assets/770352d5c0066b2e/frame_000002.png',
            'assets/770352d5c0066b2e/frame_000003.png',
            'assets/770352d5c0066b2e/frame_000004.png',
            'assets/770352d5c0066b2e/frame_000005.png',
        ],
        size=512,
        square_ok=True,
    )
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()


if __name__ == '__main__':
    # data
    video_file = "assets/xoWH4gEHdog/404a47d74c89b6a0.mp4"
    camera_pose_file = "assets/xoWH4gEHdog/404a47d74c89b6a0.txt"
    output_frames = 49

    # model
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    main(
        video_file,
        camera_pose_file,
        output_frames,
        device,
        batch_size,
        schedule,
        lr,
        niter,
        model_name,
    )
