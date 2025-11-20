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
                retriever = Retriever(retrieval_model, backbone=model.encoder, device=device)
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
    # IMPORTANT: Pass the underlying encoder model, not the full Splatt3R model
    print(">> Running Sparse Global Alignment...")
    cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Extract the encoder from Splatt3R model
    # Splatt3R wraps a MASt3R encoder - we need to pass that to sparse_global_alignment
    encoder_model = model.encoder  # This is the actual MASt3R model
    
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    encoder_model,  # Pass encoder, not full model
                                    lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, 
                                    device=device,
                                    opt_depth=True, 
                                    matching_conf_thr=matching_conf_thr)
    
    return scene, imgs

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
    scene, imgs = get_reconstructed_scene_splatt3r(
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

    print(">> Aggregating Gaussians...")
    with torch.no_grad():
        for i, img_obj in enumerate(tqdm(imgs)):
            pose = poses[i].detach()
            pts_global = optimized_means[i].reshape(-1, 3).to(device)
            
            # --- Placeholder Attributes ---
            N_points = pts_global.shape[0]
            opac = torch.ones((N_points, 1), device=device) * 5.0
            scale = torch.ones((N_points, 3), device=device) * 0.01
            
            import roma
            cam_rot = pose[:3, :3]
            q = roma.rotmat_to_unitquat(cam_rot).unsqueeze(0).repeat(N_points, 1)
            sh = torch.ones((N_points, 3), device=device) * 0.5

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