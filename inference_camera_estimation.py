import numpy as np
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def save_camera_parameters(intrinsics, poses, output_prefix="camera"):
    """
    将相机内参和外参保存为文本文件。

    参数:
        intrinsics (torch.Tensor): 相机内参矩阵，形状为 [n_imgs, 3, 3]
        poses (torch.Tensor): 相机外参矩阵（相机到世界），形状为 [n_imgs, 4, 4]
        output_prefix (str): 保存文件的前缀名称
    """
    intrinsics_np = intrinsics.cpu().detach().numpy()
    poses_np = poses.cpu().detach().numpy()

    for i in range(intrinsics_np.shape[0]):
        K = intrinsics_np[i]
        pose = poses_np[i]

        # 保存内参
        K_filename = f"{output_prefix}_{i+1}_intrinsic.txt"
        np.savetxt(K_filename, K, fmt='%.6f')
        print(f"Saved intrinsic matrix for Image {i+1} to {K_filename}")

        # 保存外参
        pose_filename = f"{output_prefix}_{i+1}_extrinsic.txt"
        np.savetxt(pose_filename, pose, fmt='%.6f')
        print(f"Saved extrinsic matrix for Image {i+1} to {pose_filename}")


def main():
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
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

    # 获取相机内参和外参
    intrinsics = scene.get_intrinsics()  # 相机内参矩阵 [n_imgs, 3, 3]
    extrinsics = poses  # 相机外参矩阵（相机到世界） [n_imgs, 4, 4]

    # 打印相机内参和外参
    for i in range(intrinsics.shape[0]):
        K = intrinsics[i].cpu().detach().numpy()
        pose = extrinsics[i].cpu().detach().numpy()

        print(f"Image {i+1}:")
        print("Intrinsic Matrix (K):")
        print(K)
        print("Extrinsic Matrix (Camera-to-World Pose):")
        print(pose)
        print("-" * 50)

    # 可选：将相机内参和外参保存为文件
    save_camera_parameters(intrinsics, extrinsics, output_prefix="camera")


if __name__ == '__main__':
    main()
