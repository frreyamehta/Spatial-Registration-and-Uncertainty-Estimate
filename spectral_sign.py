import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

MASK_PATH  = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/LGC_MASK_inun/data/LGC/Mask_inun/LGC_inun_mask'
LAN_DIR    = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/LGC/Landsat'
FUSED_DIR  = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/fusedimg_lgc'
SAVE_DIR   = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/spectral_plots_lgc'

DATES      = ['2004_363_Dec28', '2004_347_Dec12', '2004_331_Nov26']
SHIFTS     = [0, 8, 15]
BAND_NAMES = ['Band 1\n(Blue)', 'Band 2\n(Green)', 'Band 3\n(Red)', 
              'Band 4\n(NIR)', 'Band 5\n(SWIR1)', 'Band 6\n(SWIR2)']
#CLASSES    = {0: 'Non-irrigated', 1: 'Irrigated'}
CLASSES = {
    0: 'Water',
    1: 'Bare soil',
    2: 'Sparse vegetation',
    3: 'Dense vegetation'
}
COLORS     = {'GT': 'black', 'shift0px': '#1D9E75', 'shift8px': '#BA7517', 'shift15px': '#A32D2D'}
LINESTYLE  = {'GT': '-', 'shift0px': '--', 'shift8px': '-.', 'shift15px': ':'}

os.makedirs(SAVE_DIR, exist_ok=True)

def derive_lc_mask(landsat_img):
    """
    Derive a 4-class land cover mask from Landsat reflectance using NDVI and NDWI.
    landsat_img: (6, H, W), reflectance values
    Band order (from readme): B1=Blue, B2=Green, B3=Red, B4=NIR, B5=SWIR1, B6=SWIR2
    
    NDVI = (NIR - Red) / (NIR + Red)  → bands 3 and 2 (0-indexed)
    NDWI = (Green - NIR) / (Green + NIR) → bands 1 and 3 (0-indexed)
    
    Classes:
      0 = Water          NDWI > 0.1
      1 = Bare soil      NDVI < 0.1  (and not water)
      2 = Sparse veg     0.1 <= NDVI < 0.4
      3 = Dense veg      NDVI >= 0.4
    """
    nir  = landsat_img[3].astype(np.float32)
    red  = landsat_img[2].astype(np.float32)
    green = landsat_img[1].astype(np.float32)

    ndvi = (nir - red)  / (nir + red  + 1e-8)
    ndwi = (green - nir) / (green + nir + 1e-8)

    mask = np.zeros(landsat_img.shape[1:], dtype=np.uint8)
    mask[ndwi > 0.1]                          = 0  # water
    mask[(ndwi <= 0.1) & (ndvi < 0.1)]        = 1  # bare soil
    mask[(ndwi <= 0.1) & (ndvi >= 0.1) & (ndvi < 0.4)] = 2  # sparse veg
    mask[(ndwi <= 0.1) & (ndvi >= 0.4)]       = 3  # dense veg

    # print class distribution
    for c, name in CLASSES.items():
        print(f'  {name}: {np.sum(mask == c)} pixels ({100*np.mean(mask==c):.1f}%)')

    return mask

def load_gt(date):
    path = os.path.join(LAN_DIR, f'LAN_{date}.npy')
    img = np.load(path).astype(np.float32) / 10000.
    return img  # (C, H, W)


def load_fused(date, shift):
    path = os.path.join(FUSED_DIR, f'swin_LGC_{date}_shift{shift}px.tif')
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32) / 10000.
    return img  # (C, H, W)


def compute_signature(image, mask, class_val):
    """Mean reflectance per band for pixels belonging to class_val."""
    sig = []
    for b in range(image.shape[0]):
        band = image[b]
        pixels = band[mask == class_val]
        pixels = pixels[(pixels > 0) & (pixels < 1)]  # exclude invalid
        sig.append(np.mean(pixels) if len(pixels) > 0 else np.nan)
    return np.array(sig)


def plot_signatures(date, mask):
    gt = load_gt(date)

    fused = {'GT': gt}
    for shift in SHIFTS:
        tag = f'shift{shift}px'
        try:
            fused[tag] = load_fused(date, shift)
        except FileNotFoundError:
            print(f'Missing: {tag} for {date}')

    x = np.arange(len(BAND_NAMES))
    n_classes = len(CLASSES)

    fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5), sharey=False)
    fig.suptitle(f'Spectral signatures — LGC {date}', fontsize=13)

    for ax, (class_val, class_name) in zip(axes, CLASSES.items()):
        for tag, img in fused.items():
            sig = compute_signature(img, mask, class_val)
            ax.plot(x, sig,
                    label=tag,
                    color=COLORS.get(tag, 'gray'),
                    linestyle=LINESTYLE.get(tag, '-'),
                    linewidth=2,
                    marker='o',
                    markersize=5)

        ax.set_title(class_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(BAND_NAMES, fontsize=9)
        ax.set_ylabel('Mean reflectance', fontsize=10)
        ax.set_xlabel('Band', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'spectral_sig_{date}_LGC.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# def main():
#     # load mask
#     mask = np.fromfile(MASK_PATH, dtype=np.uint8).reshape(2040, 1720)
#     print(f'Mask loaded — irrigated pixels: {np.sum(mask==1)}, non-irrigated: {np.sum(mask==0)}')

#     for date in DATES:
#         plot_signatures(date, mask)

def main():
    # Use first test date GT to derive the land cover mask
    # (mask is derived once from GT and applied consistently across all dates)
    ref_gt = load_gt(DATES[0])
    print(f'Deriving land cover mask from {DATES[0]}...')
    mask = derive_lc_mask(ref_gt)

    for date in DATES:
        print(f'\nPlotting: {date}')
        plot_signatures(date, mask)

if __name__ == '__main__':
    main()