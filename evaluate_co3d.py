import os
import sys
import argparse
import torch
import numpy as np
import json
import shutil
import imageio.v2 as imageio
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# === Path Setup ===
# Ensure we can import from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
mast3r_root = os.path.join(current_dir, 'src', 'mast3r_src')
dust3r_repo_root = os.path.join(mast3r_root, 'dust3r')

for path in [current_dir, dust3r_repo_root, mast3r_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import Splatt3R modules
try:
    from main import MAST3RGaussians
    from run_splatter_sfm import get_reconstructed_scene_splatt3r, aggregate_and_save_ply
except ImportError:
    # Try importing with underscores if file name is different
    try:
        from run_splatter_sfm import get_reconstructed_scene_splatt3r, aggregate_and_save_ply
    except ImportError:
        # The file is run_splatter-sfm.py (hyphen)
        # We can't import with hyphen easily.
        # We will use importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_splatter_sfm", os.path.join(current_dir, "run_splatter-sfm.py"))
        run_splatter_sfm = importlib.util.module_from_spec(spec)
        sys.modules["run_splatter_sfm"] = run_splatter_sfm
        spec.loader.exec_module(run_splatter_sfm)
        get_reconstructed_scene_splatt3r = run_splatter_sfm.get_reconstructed_scene_splatt3r
        aggregate_and_save_ply = run_splatter_sfm.aggregate_and_save_ply

# Import Eval Utils
import eval_utils

def load_model(ckpt_path, device):
    print(f">> Loading Splatt3R model from {ckpt_path}...")
    model = MAST3RGaussians.load_from_checkpoint(ckpt_path, strict=False).to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='Run evaluations on CO3D dataset')
    parser.add_argument('--dataset_root', default='/ssd_scratch/cvit/Eshaan', help='Root directory of CO3D dataset')
    parser.add_argument('--output_root', default='/ssd_scratch/cvit/eval_outputs', help='Root directory for outputs')
    parser.add_argument('--weights', nargs='+', type=float, default=[0, 1, 2, 4], help='Parametric loss weights to evaluate')
    parser.add_argument('--conf_thresh', type=float, default=2.0, help='Confidence threshold')
    parser.add_argument('--scenegraph_type', default='swin', help='Scene graph type')
    parser.add_argument('--use_huggingface', action='store_true', help='Download model from HuggingFace')
    parser.add_argument('--ckpt', default='checkpoints/splatt3r.ckpt', help='Local checkpoint path')
    parser.add_argument('--num_views', type=int, default=None, help='Number of training views for few-view task (e.g. 3, 6, 9)')
    parser.add_argument('--max_categories', type=int, default=None, help='Maximum number of categories to process')
    parser.add_argument('--max_sequences', type=int, default=None, help='Maximum number of sequences per category')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for subsampling')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load Model
    if args.use_huggingface:
        model_name = "brandonsmart/splatt3r_v1.0"
        filename = "epoch=19-step=1200.ckpt"
        ckpt_path = hf_hub_download(repo_id=model_name, filename=filename)
    else:
        ckpt_path = args.ckpt
        
    model = load_model(ckpt_path, device)
    
    # Initialize Evaluator
    evaluator = eval_utils.Evaluator(device)
    
    # Load Annotations
    # We assume annotations are in dataset_root/category/frame_annotations.jgz
    # We iterate over categories
    categories = sorted([d for d in os.listdir(args.dataset_root) if os.path.isdir(os.path.join(args.dataset_root, d)) and not d.startswith('_')])
    
    if args.max_categories is not None:
        categories = categories[:args.max_categories]
        print(f"Limiting to {len(categories)} categories.")
    
    results = {}
    
    for category in tqdm(categories, desc="Categories"):
        category_path = os.path.join(args.dataset_root, category)
        annot_path = os.path.join(category_path, 'frame_annotations.jgz')
        
        if not os.path.exists(annot_path):
            # Try .gz
            annot_path = os.path.join(category_path, 'frame_annotations.gz')
            if not os.path.exists(annot_path):
                print(f"Skipping {category}: No annotations found")
                continue
                
        print(f"Loading annotations for {category}...")
        try:
            annotations = eval_utils.load_co3d_annotations(annot_path)
        except Exception as e:
            print(f"Error loading annotations for {category}: {e}")
            continue
            
        # Get sequences
        sequences = sorted(list(set(a['sequence_name'] for a in annotations)))
        
        if args.max_sequences is not None:
            sequences = sequences[:args.max_sequences]
            print(f"  Limiting to {len(sequences)} sequences for {category}.")
        
        for sequence in sequences:
            print(f"Processing sequence: {category}/{sequence}")
            
            # Get split
            train_data, test_data = eval_utils.get_split_data(annotations, sequence, args.dataset_root)
            
            if not train_data or not test_data:
                print(f"  Skipping: Missing train or test data (Train: {len(train_data)}, Test: {len(test_data)})")
                continue
                
            # Few-view Subsampling
            if args.num_views is not None:
                if len(train_data) > args.num_views:
                    # Evenly space them out (linspace) to ensure coverage
                    indices = np.linspace(0, len(train_data) - 1, args.num_views, dtype=int)
                    # indices = np.random.choice(len(train_data), args.num_views, replace=False)
                    # indices.sort() 
                    train_data = [train_data[i] for i in indices]
                    print(f"  Subsampled to {len(train_data)} views.")
                else:
                    print(f"  Warning: Requested {args.num_views} views but only have {len(train_data)}.")
            
            train_files = [x['image_path'] for x in train_data]
            
            # Run for each weight
            for weight in args.weights:
                output_dir = os.path.join(args.output_root, category, sequence, f"w_{weight}")
                os.makedirs(output_dir, exist_ok=True)
                
                ply_path = os.path.join(output_dir, 'scene.ply')
                
                # Run Reconstruction
                print(f"  Running reconstruction (w={weight})...")
                try:
                    scene, imgs, pairs, cache_dir = get_reconstructed_scene_splatt3r(
                        output_dir, model, None, device, train_files,
                        scenegraph_type=args.scenegraph_type,
                        photometric_loss_w=weight,
                        matching_conf_thr=args.conf_thresh
                    )
                except Exception as e:
                    print(f"  Reconstruction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Save PLY
                print("  Aggregating Gaussians...")
                aggregate_and_save_ply(scene, imgs, pairs, cache_dir, output_dir, device, args.conf_thresh, train_files)
                
                # Load PLY for rendering
                final_means, final_quats, final_scales, final_opacities, final_colors = eval_utils.load_ply(ply_path, device)
                
                # Get Estimated Poses (Train)
                est_poses = scene.get_im_poses().detach().cpu() # (N, 4, 4)
                
                # Get GT Poses (Train)
                gt_poses = torch.stack([x['w2c'] for x in train_data])
                
                # Get GT Poses (Test)
                test_gt_poses = torch.stack([x['w2c'] for x in test_data])
                
                # Align
                print("  Aligning...")
                try:
                    aligned_test_w2c = eval_utils.align_and_transform_cameras(est_poses, gt_poses, test_gt_poses)
                except Exception as e:
                    print(f"  Alignment failed: {e}")
                    continue
                
                # Save PLY


                
                # Render and Evaluate
                print("  Evaluating on test set...")
                seq_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
                
                for i, item in enumerate(test_data):
                    w2c = aligned_test_w2c[i].to(device)
                    K = item['K'].to(device)
                    H, W = item['H'], item['W']
                    
                    # Render
                    render = eval_utils.render_view(final_means, final_quats, final_scales, final_opacities, final_colors, w2c, K, H, W, device)
                    
                    # Load GT Image
                    gt_img = imageio.imread(item['image_path'])
                    gt_img = torch.from_numpy(gt_img).float().to(device) / 255.0
                    if gt_img.shape[-1] == 3:
                        gt_img = gt_img.permute(2, 0, 1) # (C, H, W)
                    
                    render = render.permute(2, 0, 1) # (C, H, W)
                    
                    # Metrics
                    m = evaluator.compute_metrics(render, gt_img)
                    for k, v in m.items():
                        seq_metrics[k].append(v)
                        
                    # Save Visualization (First 5 views)
                    if i < 5:
                        vis_dir = os.path.join(output_dir, 'vis')
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # Convert back to [0, 1] HWC numpy
                        render_np = render.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
                        gt_np = gt_img.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
                        
                        # Concatenate side-by-side
                        combined = np.concatenate([gt_np, render_np], axis=1)
                        
                        # Save
                        vis_path = os.path.join(vis_dir, f'view_{i:03d}.png')
                        imageio.imwrite(vis_path, (combined * 255).astype(np.uint8))

                        
                # Average metrics
                avg_metrics = {k: np.mean(v) for k, v in seq_metrics.items()}
                print(f"  Results (w={weight}): {avg_metrics}")
                
                # Save results
                res_key = f"{category}/{sequence}/w_{weight}"
                results[res_key] = avg_metrics
                
                with open(os.path.join(args.output_root, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                    
                # Cleanup Cache
                if os.path.exists(cache_dir):
                    print(f"  Cleaning up cache: {cache_dir}")
                    shutil.rmtree(cache_dir)

if __name__ == '__main__':
    main()
