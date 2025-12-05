import os
import json
import gzip
import torch
import numpy as np
import imageio.v2 as imageio
from gsplat import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def load_co3d_annotations(path):
    """Loads CO3D annotations from a .jgz or .gz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")
    
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    return data

def get_split_data(annotations, sequence_name, dataset_root):
    """
    Extracts train/test splits for a given sequence.
    Returns dictionaries containing image paths and camera parameters.
    """
    train_data = []
    test_data = []
    
    for frame in annotations:
        if frame['sequence_name'] != sequence_name:
            continue
            
        # Check split
        # "frame_splits": ["singlesequence_apple_test_0_unseen"] -> test
        # "frame_splits": ["singlesequence_apple_test_0_seen"] -> train (or similar)
        # Usually "unseen" is test, others are train.
        is_test = any('unseen' in split for split in frame['meta']['frame_splits'])
        
        # Parse camera
        vp = frame['viewpoint']
        R = torch.tensor(vp['R'])
        T = torch.tensor(vp['T'])
        
        # CO3D cameras are World-to-Camera?
        # "R": Rotation matrix. "T": Translation vector.
        # Usually P = K [R | T]
        
        # Intrinsics
        # ndc_isotropic means:
        # focal_length in NDC.
        # principal_point in NDC.
        # We need image size to convert.
        image_path = os.path.join(dataset_root, frame['image']['path'])
        if not os.path.exists(image_path):
            continue
            
        # We delay loading image size until needed or assume consistent?
        # CO3D images can vary in size.
        # But for a single sequence they might be consistent or we resize.
        # We'll read image size from the file or annotation if available.
        # Annotation has "image": {"size": [H, W]}
        H, W = frame['image']['size']
        
        focal_ndc = vp['focal_length']
        pp_ndc = vp['principal_point']
        
        # Convert NDC to pixels
        # min_dim = min(H, W)
        # fx = focal_ndc[0] * min_dim / 2
        # fy = focal_ndc[1] * min_dim / 2
        # cx = -pp_ndc[0] * min_dim / 2 + W / 2
        # cy = -pp_ndc[1] * min_dim / 2 + H / 2
        
        min_dim = min(H, W)
        fx = focal_ndc[0] * min_dim / 2.0
        fy = focal_ndc[1] * min_dim / 2.0
        cx = -pp_ndc[0] * min_dim / 2.0 + W / 2.0
        cy = -pp_ndc[1] * min_dim / 2.0 + H / 2.0
        
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Construct 4x4 World-to-Camera matrix
        # w2c = [R | T]
        w2c = torch.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        
        item = {
            'image_path': image_path,
            'w2c': w2c,
            'K': K,
            'H': H,
            'W': W,
            'frame_number': frame['frame_number']
        }
        
        if is_test:
            test_data.append(item)
        else:
            train_data.append(item)
            
    # Sort by frame number
    train_data.sort(key=lambda x: x['frame_number'])
    test_data.sort(key=lambda x: x['frame_number'])
    
    return train_data, test_data

def compute_umeyama(source_points, target_points):
    """
    Computes Sim3 alignment (s, R, t) such that:
    target = s * R * source + t
    
    source_points: (N, 3)
    target_points: (N, 3)
    """
    assert source_points.shape == target_points.shape
    
    # Centering
    source_mean = source_points.mean(dim=0)
    target_mean = target_points.mean(dim=0)
    
    src_centered = source_points - source_mean
    tgt_centered = target_points - target_mean
    
    # Variance
    src_var = (src_centered ** 2).sum() / len(source_points)
    
    # Covariance
    cov = (tgt_centered.T @ src_centered) / len(source_points)
    
    # SVD
    U, D, Vh = torch.linalg.svd(cov)
    V = Vh.T
    
    # Rotation
    S = torch.eye(3, device=source_points.device)
    if torch.det(U @ V.T) < 0:
        S[2, 2] = -1
    
    R = U @ S @ V.T
    
    # Scale
    scale = torch.trace(torch.diag(D) @ S) / src_var
    
    # Translation
    t = target_mean - scale * (R @ source_mean)
    
    return scale, R, t

def align_and_transform_cameras(train_est_w2c, train_gt_w2c, test_gt_w2c):
    """
    Aligns estimated cameras to GT cameras using training set,
    then transforms test GT cameras to the estimated coordinate system.
    
    Note: We usually align ESTIMATED to GT to evaluate in GT frame.
    But here we want to render using the RECONSTRUCTION (which is in est frame).
    So we need to transform GT cameras to ESTIMATED frame.
    
    Let T_est be pose in est frame.
    Let T_gt be pose in gt frame.
    We want T_gt -> T_est transformation.
    
    We use camera centers for alignment.
    C_est = -R_est^T * t_est
    C_gt = -R_gt^T * t_gt
    
    Find Sim3 (s, R_align, t_align) such that:
    C_est ~ s * R_align * C_gt + t_align
    
    Then for test views, we map C_gt_test to C_est_test using this Sim3.
    And we also need to rotate the orientation.
    R_est = R_align * R_gt
    
    Wait, Sim3 applies to points.
    X_est = s * R_align * X_gt + t_align
    
    Camera center is a point.
    Orientation: R_est = R_align * R_gt ?
    Let's verify.
    X_cam = R_gt * X_gt + t_gt
    X_cam = R_est * X_est + t_est
    
    Substitute X_est:
    X_cam = R_est * (s * R_align * X_gt + t_align) + t_est
          = s * R_est * R_align * X_gt + R_est * t_align + t_est
    
    We want this to match R_gt * X_gt + t_gt (up to scale? No, X_cam is metric in camera frame).
    Actually, if we just scale the world, the camera intrinsics might need adjustment if focal length is in world units?
    No, focal length is in pixels.
    
    So:
    R_gt = R_est * R_align
    t_gt = R_est * t_align + t_est  (divided by s? No)
    
    Actually, it's easier to align the reconstruction to GT.
    But we have the reconstruction fixed (the Gaussians).
    So we must transform the GT cameras to the reconstruction frame.
    
    So we find (s, R, t) such that:
    C_est = s * R * C_gt + t
    
    Then for a test camera (R_gt, t_gt):
    C_gt_test = -R_gt^T * t_gt
    C_est_test = s * R * C_gt_test + t
    
    And rotation?
    R_est_test = R_gt_test * R^T ?
    Let's check:
    Global rotation of the world by R maps X to R*X.
    Camera rotation R_cam maps world vector v to camera vector R_cam * v.
    If we rotate world by R, the new coordinates are X' = R * X.
    The camera vector should stay same.
    R_cam' * X' = R_cam * X
    R_cam' * R * X = R_cam * X
    => R_cam' * R = R_cam
    => R_cam' = R_cam * R^T
    
    So yes:
    R_est = R_gt * R^T
    
    And translation t_est?
    t_est = -R_est * C_est
    
    So the procedure:
    1. Extract C_est and C_gt for training set.
    2. Compute Sim3 (s, R, t) mapping C_gt -> C_est.
    3. For test set:
       C_gt_test = ...
       C_est_test = s * R * C_gt_test + t
       R_est_test = R_gt_test * R.T
       t_est_test = -R_est_test @ C_est_test
       w2c_est_test = [R_est_test | t_est_test]
       
    Also need to scale the Gaussians?
    If we map cameras to Gaussian frame, we don't touch Gaussians.
    But we need to handle the scale `s`.
    If the world is scaled by `s`, the depth is scaled by `s`.
    But `gsplat` takes `means` (positions).
    If we move cameras to the `means` space, we are good.
    
    One detail: `run_splatter-sfm.py` (MASt3R) might output a "scaled" reconstruction where the scale is arbitrary.
    The Sim3 scale `s` handles this.
    
    Returns: List of test w2c matrices in estimated frame.
    """
    
    # Extract centers
    def get_center(w2c):
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        return -R.T @ t
        
    train_C_est = torch.stack([get_center(p) for p in train_est_w2c])
    train_C_gt = torch.stack([get_center(p) for p in train_gt_w2c])
    
    # Compute Sim3: C_est = s * R * C_gt + t
    s, R_align, t_align = compute_umeyama(train_C_gt, train_C_est)
    
    transformed_test_w2c = []
    for gt_w2c in test_gt_w2c:
        C_gt = get_center(gt_w2c)
        
        # Transform center
        C_est = s * R_align @ C_gt + t_align
        
        # Transform rotation
        # R_est = R_gt * R_align^T
        R_gt = gt_w2c[:3, :3]
        R_est = R_gt @ R_align.T
        
        # Reconstruct t
        t_est = -R_est @ C_est
        
        w2c_est = torch.eye(4, device=gt_w2c.device)
        w2c_est[:3, :3] = R_est
        w2c_est[:3, 3] = t_est
        
        transformed_test_w2c.append(w2c_est)
        
    return transformed_test_w2c

def render_and_evaluate(scene, test_data, transformed_test_w2c, device='cuda'):
    """
    Renders the scene from transformed test viewpoints and computes metrics.
    """
    # Extract Gaussian parameters from scene
    # We need to aggregate them like in run_splatter-sfm.py
    # But scene.get_dense_pts3d() returns per-view points.
    # We should use the same aggregation logic or just use all points.
    
    # Actually, `run_splatter-sfm.py` aggregates them into a single PLY.
    # We can do the same here in memory.
    
    poses = scene.get_im_poses()
    optimized_means, _, confidences = scene.get_dense_pts3d()
    imgs = scene.imgs # List of dicts
    
    all_xyz, all_opac, all_scale, all_rot, all_sh = [], [], [], [], []
    
    # Re-implement aggregation logic (simplified)
    # We assume `scene` has everything we need.
    # We need to handle SH correctly (residual + base).
    # This is complex to duplicate.
    # Maybe we can just load the saved PLY?
    # But we need to know the alignment.
    # The PLY is in the same frame as `scene.get_im_poses()`.
    # So we can use the PLY and the alignment we computed from `scene.get_im_poses()`.
    
    pass # We will implement this logic in the main script using the PLY loader
    
    
class Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
        
    def compute_metrics(self, pred, target):
        # pred, target: (C, H, W) in [0, 1]
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            
        # Clamp to [0, 1]
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
            
        p = self.psnr(pred, target)
        s = self.ssim(pred, target)
        l = self.lpips(pred, target)
        
        return {'psnr': p.item(), 'ssim': s.item(), 'lpips': l.item()}

def load_ply(path, device='cuda'):
    """Loads Gaussian Splat PLY file."""
    from plyfile import PlyData
    
    plydata = PlyData.read(path)
    
    xyz = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']), axis=1)
    opac = plydata['vertex']['opacity']
    
    scale_names = [p.name for p in plydata['vertex'].properties if p.name.startswith('scale_')]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[1]))
    scale = np.stack([plydata['vertex'][n] for n in scale_names], axis=1)
    
    rot_names = [p.name for p in plydata['vertex'].properties if p.name.startswith('rot_')]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[1]))
    rot = np.stack([plydata['vertex'][n] for n in rot_names], axis=1)
    
    # SH / Color
    # We assume degree 0 (RGB) for now as run_splatter-sfm saves f_dc_0..2
    sh_names = ['f_dc_0', 'f_dc_1', 'f_dc_2']
    sh = np.stack([plydata['vertex'][n] for n in sh_names], axis=1)
    
    # Convert to torch
    means = torch.from_numpy(xyz).float().to(device)
    opacities = torch.from_numpy(opac).float().to(device)
    scales = torch.exp(torch.from_numpy(scale).float().to(device)) # Log scale in PLY
    quats = torch.from_numpy(rot).float().to(device)
    colors = torch.from_numpy(sh).float().to(device) # RGB
    
    return means, quats, scales, opacities, colors

def render_view(means, quats, scales, opacities, colors, w2c, K, H, W, device='cuda'):
    """
    Renders a single view using gsplat.
    w2c: (4, 4) World-to-Camera matrix
    K: (3, 3) Intrinsics
    """
    viewmat = w2c.unsqueeze(0).to(device) # (1, 4, 4)
    K_tensor = K.unsqueeze(0).to(device) # (1, 3, 3)
    
    # gsplat.rasterization expects:
    # means: (N, 3)
    # quats: (N, 4)
    # scales: (N, 3)
    # opacities: (N,)
    # colors: (N, 3) or (N, C, 3) ? 
    # If colors is (N, 3), output is (1, H, W, 3)
    
    # Ensure shapes
    if opacities.ndim == 2:
        opacities = opacities.squeeze(-1)
        
    render_colors, render_alphas, meta = rasterization(
        means, quats, scales, opacities, colors, viewmat, K_tensor, W, H
    )
    
    # render_colors: (1, H, W, 3)
    return render_colors.squeeze(0)


