import cv2
import sys
import os
import subprocess
import numpy as np
import numpy as np
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def extract_frames(video_path, save_path, image_format='jpg'):
    """
    Extracts frames from a video and saves them as images.

    :param video_path: Path to the input video file.
    :param save_path: Directory where frames will be saved.
    :param image_format: Image format for saved frames (e.g., 'png', 'jpg').
    """
    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    # Create the save directory if it doesn't exist
    if os.path.exists(save_path):
        return
    os.makedirs(save_path, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'.")
        sys.exit(1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total Frames: {frame_count}, FPS: {fps}")

    count = 0
    while count < frame_count:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames to read

        # Construct the filename with leading zeros
        filename = os.path.join(save_path, f"frame_{count:06d}.{image_format}")
        
        # Save the frame as an image
        cv2.imwrite(filename, frame)

        count += 1

    cap.release()
    print(f"Extraction complete. {count} frames saved to '{save_path}'.")

def load_camera_extrinsics(frames_dir):
    """
    Loads camera extrinsic parameters from COLMAP's images.txt and returns
    rotation and translation matrices sorted in temporal order.

    :param images_file: Path to COLMAP's images.txt file.
    :return: 
        rotations: NumPy array of shape (n, 3, 3) containing rotation matrices.
        translations: NumPy array of shape (n, 3, 1) containing translation vectors.
    """
    image_data = []

    # Regular expression to extract frame number from image name
    # Modify this regex based on your image naming convention

    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    model_name = "/home/lingcheng/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(
        frames_dir,
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

    # 获取相机内参和外参
    extrinsics = poses.cpu().detach().numpy().copy()  # 相机外参矩阵（相机到世界） [n_imgs, 4, 4]
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3:4]
    
    return rotations, translations

def process_extrinsics(video_path, project_dir):
    frames_dir = os.path.join(project_dir, 'extracted_frames')
    extract_frames(video_path, frames_dir)
    rotations, translations = load_camera_extrinsics(frames_dir)
    return rotations, translations

def RotationError(R_gen, R_gt, frames):
    """
    Calculate rotation error

    Args:
        R_gen and R_gt: n * 3 * 3
        frames: only calculate the error of first {frames} frames
    """
    R_gen = R_gen[:frames, :, :]
    R_gt = R_gt[:frames, :, :]
    R_gt= np.transpose(R_gt, axes = (0, 2, 1))
    RotErr = np.sum(
        np.arccos(
            0.5 * (np.trace(
                np.matmul(R_gen, R_gt), axis1 = 1, axis2 = 2
            ) - 1)
        )
    ) / frames
    return RotErr

def TransationError(T_gen, T_gt, frames):
    """
    Calculate translation Error

    Args:
        T_gen and T_gt: n * 3 * 1
        frames: only calculate the error of first {frames} frames
    """
    T_gen = T_gen[:frames, :, :]
    T_gt = T_gt[:frames, :, :]
    TransErr = np.sum(np.sum((T_gen - T_gt) ** 2, axis = (1, 2)) ** 0.5) / frames
    return TransErr

def calc_error(video_path_gen, video_path_gt, project_dir_gen, project_dir_gt, frames = None):
    '''
    Calculate Rotation and Translation Error over the first {frames} frames
    '''
    rots_gen, trans_gen = process_extrinsics(video_path_gen, project_dir_gen)
    rots_gt, trans_gt = process_extrinsics(video_path_gt, project_dir_gt)
    if frames is None:
        frames = rots_gen.shape[0]
    RotErr = RotationError(rots_gen, rots_gt, frames)
    TransErr = TransationError(trans_gen, trans_gt, frames)
    return RotErr, TransErr

if __name__ == "__main__":
    video_path_gen = '/home/lingcheng/dust3r/camerapose/test/input_video.mp4'
    project_dir_gen = '/home/lingcheng/dust3r/camerapose/test/gen'
    video_path_gt = '/home/lingcheng/dust3r/camerapose/test/input_video.mp4'
    project_dir_gt = '/home/lingcheng/dust3r/camerapose/test/gt'
    RotErr, TransErr = calc_error(video_path_gen, video_path_gt, project_dir_gen, project_dir_gt, 49)
    print(RotErr)
    print(TransErr)