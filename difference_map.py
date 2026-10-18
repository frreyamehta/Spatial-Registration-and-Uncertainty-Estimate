import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio

#ADDED
def to_diff_map(diff):
    """Convert difference map to a visible heatmap."""
    diff = np.mean(diff, axis=0)  # average across channels → (H, W)
    diff = diff / (diff.max() + 1e-8)  # normalize to [0,1]
    return diff

def load_fused_image(fused_dir, date, shift_tag):
    """Load a fused .tif by date and shift tag."""
    fname = os.path.join(fused_dir, f'swin_LGC_{date}_{shift_tag}.tif')
    with rasterio.open(fname) as src:
        img = src.read().astype(np.float32) / 10000.
    return img

def load_gt_image(root_dir, date):
    fname = os.path.join(root_dir, 'landsat', f'LAN_{date}.npy')
    img = np.load(fname).astype(np.float32) / 10000.
    return img  # (C, H, W)

def to_rgb(image, r=2, g=1, b=0):
    """Extract RGB bands and normalize to [0,1] for display."""
    rgb = np.stack([image[r], image[g], image[b]], axis=-1)
    rgb = np.clip(rgb, 0, 0.3)   # clip reflectance for visual contrast
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb

def extract_patch(image, h_start, w_start, patch_size=256):
    return image[:, h_start:h_start+patch_size, w_start:w_start+patch_size]

def visualize_patches(root_dir, fused_dir, dates, shift_tags, shift_labels,
                      patch_h=400, patch_w=400, patch_size=256,
                      save_dir='./difference_maps'):
    os.makedirs(save_dir, exist_ok=True)

    for date in dates:
        gt = load_gt_image(root_dir, date)
        gt_patch = extract_patch(gt, patch_h, patch_w, patch_size)

        n_cols = len(shift_tags) + 1   # GT + one per shift
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
        fig.suptitle(f'LGC — {date}', fontsize=13, y=1.01)

        # Ground truth
        axes[0].imshow(to_rgb(gt_patch))
        axes[0].set_title('Ground truth', fontsize=10)
        axes[0].axis('off')

        # Fused outputs per shift
        for ax, tag, label in zip(axes[1:], shift_tags, shift_labels):
            try:
                fused = load_fused_image(fused_dir, date, tag)
                fused_patch = extract_patch(fused, patch_h, patch_w, patch_size)
                
                #ADDED
                diff = np.abs(fused_patch - gt_patch)
                diff_map = to_diff_map(diff)

                im = ax.imshow(diff_map, cmap='hot')  # heatmap
                ax.set_title(f'{label}', fontsize=10)
                #ADDED till here

            except FileNotFoundError:
                ax.set_title(f'{label}\n(not found)', fontsize=9, color='red')
            ax.axis('off')
        
        #ADDED:
        cbar = fig.colorbar(im, ax=axes[1:], fraction=0.02, pad=0.04)
        cbar.set_label("Absolute Error", fontsize=10, labelpad=10)
        cbar.ax.text(1.5, 1.0, "Higher\n error",  transform=cbar.ax.transAxes, va='top', ha='left', fontsize=9)
        cbar.ax.text(1.5, 0.0, "Lower\n error",  transform=cbar.ax.transAxes, va='bottom', ha='left', fontsize=9)
        cbar.set_ticks([])

        #plt.tight_layout()
        save_path = os.path.join(save_dir, f'diff_map_{date}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {save_path}')


if __name__ == '__main__':

    ROOT_DIR   = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/LGC'
    FUSED_DIR  = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/fusedimg_lgc'
    SAVE_DIR   = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/difference_maps_lgc'

    DATES = ['2004_363_Dec28', '2004_347_Dec12', '2004_331_Nov26']

    # Tags must match exactly how you saved them in test.py
    SHIFT_TAGS   = ['shift0px', 'shift8px', 'shift15px']
    SHIFT_LABELS = ['0 px shift', '8 px shift', '15 px shift']

    # Patch location — pick a region with interesting land cover
    PATCH_H = 400
    PATCH_W = 400
    PATCH_SIZE = 256

    visualize_patches(
        root_dir=ROOT_DIR,
        fused_dir=FUSED_DIR,
        dates=DATES,
        shift_tags=SHIFT_TAGS,
        shift_labels=SHIFT_LABELS,
        patch_h=PATCH_H,
        patch_w=PATCH_W,
        patch_size=PATCH_SIZE,
        save_dir=SAVE_DIR
    )