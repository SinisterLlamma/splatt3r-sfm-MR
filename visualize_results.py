import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from glob import glob

def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--output_root', default='/ssd_scratch/cvit/eval_outputs', help='Root directory of outputs')
    parser.add_argument('--category', type=str, required=True, help='Category to visualize (e.g. apple)')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence to visualize')
    args = parser.parse_args()

    # Load Results
    results_path = os.path.join(args.output_root, 'results.json')
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Filter for specific sequence
    seq_key_prefix = f"{args.category}/{args.sequence}/w_"
    
    weights = []
    psnrs = []
    ssims = []
    lpips_vals = []

    for key, metrics in results.items():
        if key.startswith(seq_key_prefix):
            w = float(key.split('_')[-1])
            weights.append(w)
            psnrs.append(metrics['psnr'])
            ssims.append(metrics['ssim'])
            lpips_vals.append(metrics['lpips'])

    if not weights:
        print("No results found for this sequence.")
        return

    # Sort by weight
    sorted_indices = np.argsort(weights)
    weights = np.array(weights)[sorted_indices]
    psnrs = np.array(psnrs)[sorted_indices]
    ssims = np.array(ssims)[sorted_indices]
    lpips_vals = np.array(lpips_vals)[sorted_indices]

    # Plot Metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(weights, psnrs, marker='o')
    axes[0].set_title('PSNR vs Weight')
    axes[0].set_xlabel('Photometric Loss Weight')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].grid(True)

    axes[1].plot(weights, ssims, marker='o', color='orange')
    axes[1].set_title('SSIM vs Weight')
    axes[1].set_xlabel('Photometric Loss Weight')
    axes[1].set_ylabel('SSIM')
    axes[1].grid(True)

    axes[2].plot(weights, lpips_vals, marker='o', color='green')
    axes[2].set_title('LPIPS vs Weight')
    axes[2].set_xlabel('Photometric Loss Weight')
    axes[2].set_ylabel('LPIPS')
    axes[2].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(args.output_root, args.category, args.sequence, 'metrics_plot.png')
    plt.savefig(plot_path)
    print(f"Saved metrics plot to {plot_path}")

    # Visual Comparison (Image Grid)
    # We need to render images or find saved renders. 
    # The evaluation script currently doesn't save rendered images, it just computes metrics.
    # We should probably modify the eval script to save at least one example render.
    # For now, I will assume we might have saved them or I can't do it yet without re-running.
    
    # However, I can check if 'scene.ply' exists and maybe render it? 
    # No, that's too complex for a vis script.
    # I will rely on the user having run the eval script which hopefully I can modify to save an image.
    
    # Let's modify this script to assume there might be a 'render_0.png' in the output dir if we add it to eval script.
    # If not, we just skip image generation.
    
    pass

if __name__ == '__main__':
    main()
