import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from huggingface_hub import hf_hub_download

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# === Path Setup ===
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add paths with HIGH PRIORITY (insert at index 0)
mast3r_root = os.path.join(current_dir, 'src', 'mast3r_src')
dust3r_repo_root = os.path.join(mast3r_root, 'dust3r')

# Insert in reverse order of priority (last insert = highest priority)
for path in [current_dir, dust3r_repo_root, mast3r_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# === Imports ===
try:
    from main import MAST3RGaussians
except ImportError as e:
    print(f"Error importing MAST3RGaussians: {e}")
    sys.exit(1)

try:
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from mast3r.utils.misc import hash_md5
except ImportError as e:
    print(f"Error importing mast3r/dust3r: {e}")
    sys.exit(1)

# Check for retrieval support
try:
    from mast3r.retrieval.processor import Retriever
    has_retrieval = True
    print("✓ Retrieval module loaded successfully\n")
except ImportError as e:
    has_retrieval = False
    print(f"✗ Retrieval module import failed: {e}\n")

def save_splat_ply(path, xyz, opacity, scale, rot, sh):
    """Saves Gaussian Splats to a .ply file compatible with 3DGS viewers."""
    print(f"Saving {len(xyz)} splats to {path}...")
    
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    if sh.ndim == 3:
        sh = sh[:, 0, :]
    
    # Ensure sh has 3 channels (RGB)
    if sh.shape[-1] == 1:
        # If only 1 channel, replicate it to all 3 channels (grayscale)
        sh = np.repeat(sh, 3, axis=-1)
    elif sh.shape[-1] != 3:
        # If not 3 channels, take first 3 or pad with zeros
        if sh.shape[-1] > 3:
            sh = sh[:, :3]
        else:
            # Pad with zeros to make 3 channels
            padding = np.zeros((sh.shape[0], 3 - sh.shape[-1]))
            sh = np.concatenate([sh, padding], axis=-1)
    
    elements = np.empty(len(xyz), dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0
    elements['f_dc_0'] = sh[:, 0]
    elements['f_dc_1'] = sh[:, 1]
    elements['f_dc_2'] = sh[:, 2]
    elements['opacity'] = 1 / (1 + np.exp(-opacity[:, 0])) 
    elements['scale_0'] = np.log(np.clip(scale[:, 0], 1e-6, None))
    elements['scale_1'] = np.log(np.clip(scale[:, 1], 1e-6, None))
    elements['scale_2'] = np.log(np.clip(scale[:, 2], 1e-6, None))
    elements['rot_0'] = rot[:, 0] 
    elements['rot_1'] = rot[:, 1] 
    elements['rot_2'] = rot[:, 2] 
    elements['rot_3'] = rot[:, 3] 

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_gaussian_attributes(cache_path, img1_instance, img2_instance, is_img1=True):
    """Load saved Gaussian attributes from cache."""
    idx1 = hash_md5(img1_instance)
    idx2 = hash_md5(img2_instance)
    
    if is_img1:
        path_gauss = os.path.join(cache_path, 'forward', idx1, f'{idx2}_gauss.pth')
    else:
        path_gauss = os.path.join(cache_path, 'forward', idx2, f'{idx1}_gauss.pth')
    
    if os.path.isfile(path_gauss):
        gauss_attrs = torch.load(path_gauss)
        return gauss_attrs[0] if is_img1 else gauss_attrs[1]
    return None


def get_reconstructed_scene_splatt3r(outdir, model, retrieval_model, device, filelist, 
                                     scenegraph_type='complete', winsize=20, refid=10,
                                     lr1=0.07, niter1=500, lr2=0.01, niter2=300, 
                                     matching_conf_thr=5.0):
    """
    Custom version of get_reconstructed_scene specifically for Splatt3R CLI usage.
    """
    # 1. Load Images
    print(f">> Loading {len(filelist)} images...")
    imgs = load_images(filelist, size=512, verbose=True)
    if len(imgs) == 0:
        raise ValueError("No images found!")

    # 2. Build Scene Graph Params
    scene_graph_params = [scenegraph_type]
    if scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))
        scene_graph_params.append(str(refid))
    
    scene_graph = '-'.join(scene_graph_params)

    # 3. Retrieval (Optional)
    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        if not has_retrieval:
            print("WARNING: Retrieval module not available, falling back to 'complete' mode")
            scenegraph_type = 'complete'
            scene_graph = 'complete'
        elif retrieval_model is None or not os.path.exists(retrieval_model):
            print(f"WARNING: Retrieval checkpoint not found at {retrieval_model}, falling back to 'complete' mode")
            scenegraph_type = 'complete'
            scene_graph = 'complete'
        else:
            print(">> Running Image Retrieval...")
            try:
                # FIXED: Pass only the encoder to the retriever
                backbone = model.encoder if hasattr(model, 'encoder') else model
                retriever = Retriever(retrieval_model, backbone=backbone, device=device)
                with torch.no_grad():
                    sim_matrix = retriever(filelist)
                del retriever
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"WARNING: Retrieval failed with error: {e}")
                print(f"Falling back to 'complete' scene graph mode...")
                scenegraph_type = 'complete'
                scene_graph = 'complete'
                sim_matrix = None

    # 4. Make Pairs
    print(f">> Making pairs ({scenegraph_type})...")
    if sim_matrix is not None:
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    else:
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

    # 5. Run Sparse Global Alignment
    print(">> Running Sparse Global Alignment with full Splatt3R model...")
    cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model,  # Pass full model (will be handled correctly now)
                                    lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, 
                                    device=device,
                                    opt_depth=True, 
                                    matching_conf_thr=matching_conf_thr)
    
    return scene, imgs, pairs, cache_dir

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get file list
    if os.path.isdir(args.input_dir):
        filelist = [os.path.join(args.input_dir, f) for f in sorted(os.listdir(args.input_dir)) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        filelist = [args.input_dir]
    
    if len(filelist) == 0:
        raise ValueError(f"No images found in {args.input_dir}")
    
    print(f"Found {len(filelist)} images")

    # Load Splatt3R Model
    print(f">> Loading Splatt3R model...")
    if args.use_huggingface:
        model_name = "brandonsmart/splatt3r_v1.0"
        filename = "epoch=19-step=1200.ckpt"
        print(f"   Downloading from HuggingFace: {model_name}/{filename}")
        weights_path = hf_hub_download(repo_id=model_name, filename=filename)
    else:
        weights_path = args.ckpt
    
    print(f"   Loading checkpoint from: {weights_path}")
    model = MAST3RGaussians.load_from_checkpoint(weights_path, strict=False).to(device)
    model.eval()
    print("   Model loaded successfully!")

    # Run Reconstruction
    scene, imgs, pairs, cache_dir = get_reconstructed_scene_splatt3r(
        args.output_dir, model, args.retrieval_ckpt, device, filelist,
        scenegraph_type=args.scenegraph_type,
        winsize=args.winsize,
        refid=args.refid
    )

    # Extract & Save Splats
    print(">> Extracting optimized geometry...")
    poses = scene.get_im_poses()
    optimized_means, _, _ = scene.get_dense_pts3d()
    
    all_xyz, all_opac, all_scale, all_rot, all_sh = [], [], [], [], []

    print(">> Aggregating Gaussians with predicted attributes from Splatt3R...")
    with torch.no_grad():
        for i, img_obj in enumerate(tqdm(imgs)):
            img_instance = filelist[i]
            pose = poses[i].detach()
            pts_global = optimized_means[i].reshape(-1, 3).to(device)
            N_points = pts_global.shape[0]
            
            # Get image dimensions - handle different formats
            true_shape = img_obj['true_shape']
            if isinstance(true_shape, (tuple, list)) and len(true_shape) == 2:
                H, W = true_shape
            elif isinstance(true_shape, np.ndarray):
                if true_shape.shape == (2,):
                    H, W = int(true_shape[0]), int(true_shape[1])
                else:
                    # Fallback: get from image tensor
                    H, W = img_obj['img'].shape[-2:]
            else:
                # Fallback: get from image tensor
                H, W = img_obj['img'].shape[-2:]
            
            # **CHANGED: Load Gaussian attributes from saved pairs**
            gauss_attrs = None
            for img1, img2 in pairs:
                if img1['instance'] == img_instance:
                    gauss_attrs = load_gaussian_attributes(cache_dir, img1['instance'], img2['instance'], is_img1=True)
                    if gauss_attrs is not None and len(gauss_attrs) > 0:
                        break
                elif img2['instance'] == img_instance:
                    gauss_attrs = load_gaussian_attributes(cache_dir, img1['instance'], img2['instance'], is_img1=False)
                    if gauss_attrs is not None and len(gauss_attrs) > 0:
                        break
            
            # Debug: print what attributes we have
            if i == 0:  # Only print for first image
                if gauss_attrs is not None:
                    print(f"\nDebug - Gaussian attributes found: {list(gauss_attrs.keys())}")
                    for key, val in gauss_attrs.items():
                        if isinstance(val, torch.Tensor):
                            print(f"  {key}: shape={val.shape}, dtype={val.dtype}, range=[{val.min():.3f}, {val.max():.3f}]")
                else:
                    print("\nDebug - No Gaussian attributes found, using fallback colors")
            
            # Extract spherical harmonics (colors) - ALWAYS use fallback for now
            # Fallback: extract colors from original image
            img_rgb = img_obj['img'][0].permute(1, 2, 0).cpu().numpy()
            img_rgb = (img_rgb * 0.5 + 0.5).clip(0, 1)
            
            # Subsample image to match point count
            step = max(1, int(np.sqrt(H * W / N_points)))
            img_rgb_sub = img_rgb[::step, ::step].reshape(-1, 3)[:N_points]
            sh = torch.from_numpy(img_rgb_sub).float().to(device)
            
            # Ensure we have exactly N_points colors
            if sh.shape[0] < N_points:
                # Pad with last color
                padding = sh[-1:].repeat(N_points - sh.shape[0], 1)
                sh = torch.cat([sh, padding], dim=0)
            elif sh.shape[0] > N_points:
                sh = sh[:N_points]
            
            # Extract opacities
            if gauss_attrs is not None and 'opacities' in gauss_attrs:
                opac = gauss_attrs['opacities'].to(device)
                if opac.ndim == 3:
                    opac = opac.reshape(-1, opac.shape[-1])[:N_points]
                elif opac.ndim == 1:
                    opac = opac.reshape(-1, 1)[:N_points]
            else:
                opac = torch.ones((N_points, 1), device=device) * 2.0
            
            # Extract scales
            if gauss_attrs is not None and 'scales' in gauss_attrs:
                scale = gauss_attrs['scales'].to(device)
                if scale.ndim == 3:
                    scale = scale.reshape(-1, scale.shape[-1])[:N_points]
            else:
                scale = torch.ones((N_points, 3), device=device) * 0.01
            
            # Extract rotations
            if gauss_attrs is not None and 'rotations' in gauss_attrs:
                q = gauss_attrs['rotations'].to(device)
                if q.ndim == 3:
                    q = q.reshape(-1, q.shape[-1])[:N_points]
            else:
                import roma
                cam_rot = pose[:3, :3]
                q = roma.rotmat_to_unitquat(cam_rot).unsqueeze(0).repeat(N_points, 1)

            all_xyz.append(pts_global)
            all_opac.append(opac)
            all_scale.append(scale)
            all_rot.append(q)
            all_sh.append(sh)

    final_xyz = torch.cat(all_xyz).cpu().numpy()
    final_opac = torch.cat(all_opac).cpu().numpy()
    final_scale = torch.cat(all_scale).cpu().numpy()
    final_rot = torch.cat(all_rot).cpu().numpy()
    final_sh = torch.cat(all_sh).cpu().numpy()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'scene.ply')
    save_splat_ply(out_path, final_xyz, final_opac, final_scale, final_rot, final_sh)
    print(f">> Done! Saved to {out_path}")
    print(f"   Total Gaussians: {len(final_xyz)}")
    print(f"   Color range: [{final_sh.min():.3f}, {final_sh.max():.3f}]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Splatt3R-SfM pipeline to generate Gaussian Splat .ply')
    parser.add_argument('--input_dir', required=True, help='Path to images folder')
    parser.add_argument('--output_dir', required=True, help='Where to save .ply')
    parser.add_argument('--ckpt', default='checkpoints/splatt3r.ckpt', help='Local Splatt3R weights path')
    parser.add_argument('--use_huggingface', action='store_true', help='Download model from HuggingFace')
    parser.add_argument('--retrieval_ckpt', default=None, help='Retrieval model weights (optional, for retrieval mode)')
    parser.add_argument('--scenegraph_type', default='complete', choices=['complete', 'retrieval', 'swin'], 
                        help='Scene graph type')
    parser.add_argument('--winsize', type=int, default=20, help='Retrieval: Num key images')
    parser.add_argument('--refid', type=int, default=10, help='Retrieval: Num neighbors')
    
    args = parser.parse_args()
    main(args)
