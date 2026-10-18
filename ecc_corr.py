import os
import random
import time
import csv
import cv2
import argparse
import rasterio
import numpy as np
from tqdm import tqdm
from sewar import rmse, ssim, sam
from scipy.ndimage import rotate as scipy_rotate  # ADDED
import torch

from datasets import load_image_pair, transform_image
from models import SwinSTFM

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR  = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/LGC'
CKPT_PATH = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/LGC_best.pth'
SAVE_DIR  = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/ecc_results'
os.makedirs(SAVE_DIR, exist_ok=True)

TEST_DATES = ['2004_363_Dec28', '2004_347_Dec12', '2004_331_Nov26']
REF_DATE   = '2005_013_Jan13'

EXPERIMENTS = {
    "SHIFTS"   :[0, 8, 15],       # pixel shift experiment 
    "ROTATIONS":  [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]   # degrees 
}

PATCH_SIZE = 256
IMAGE_SIZE = [2720, 3200]


def uiqi(im1, im2, block_size=64, return_map=False):
    if len(im1.shape)==3:
        return np.array([uiqi(im1[:,:,i], im2[:,:,i], block_size, return_map=return_map) for i in range(im1.shape[2])])
    delta_x = np.std(im1, ddof=1)
    delta_y = np.std(im2, ddof=1)
    delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
    mu_x = np.mean(im1)
    mu_y = np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
    q = q1 * q2 * q3
    return q


# ORIGINAL shift misregistration (kept as reference)
def apply_shift(image, shift):
    if shift == 0:
        return image
    shifted = np.zeros_like(image)
    shifted[:, shift:, shift:] = image[:, :-shift, :-shift]
    return shifted


# ADDED: rotational misregistration
def apply_rotation_misregistration(image, angle):
    """Apply rotational misregistration to a (C,H,W) image."""
    if angle == 0.0:
        return image
    rotated = np.zeros_like(image)
    for c in range(image.shape[0]):
        rotated[c] = scipy_rotate(
            image[c],
            angle=angle,
            reshape=False,
            mode='nearest',
            order=3
        )
    return rotated


def ecc_correct_image(misreg_img, ref_img, misreg_val, is_rotation=False,
                      warp_mode=cv2.MOTION_HOMOGRAPHY, 
                      number_of_iterations=200,
                      termination_eps=1e-6,
                      ref_band=3):
    """
    Attempt ECC-based homography correction on a (C,H,W) misregistered image.
    corrects misreg_img → ref_img (ref Landsat, images[3]).
    misreg_val: shift in px (if is_rotation=False) or angle in degrees (if True)
    Supports single-band (H,W) and multi-band (H,W,C) or (C,H,W) arrays.

    Returns:
        corrected: same shape and dtype as moving
        warp_matrix: the estimated transform (useful for logging/debugging)
    """

    success      = False
    failure_mode = 'none'
    corrected    = misreg_img.copy()
    H            = np.eye(3, dtype=np.float32)

    t0 = time.time()

    orig_dtype = misreg_img.dtype
    

    # --- Normalise axis order to (H, W, C) internally ---
    def to_hwc(img):
        if img.ndim == 2:
            return img[:, :, np.newaxis]          # (H,W) → (H,W,1)
        if img.ndim == 3 and img.shape[0] < img.shape[-1]:  #C < W
            return np.transpose(img, (1, 2, 0))   # (C,H,W) → (H,W,C)
        return img                                 # already (H,W,C)


    def to_original_shape(img, original):
        if original.ndim == 2:
            return img[:, :, 0]
        if original.ndim == 3 and original.shape[0] < original.shape[-1]:  #if condition true, then [0]=C & [-1]=W ==> original image: (C,H,W) 
            return np.transpose(img, (2, 0, 1))  #img: (H,W,C) → (C,H,W) 
        return img


    misreg_hwc = to_hwc(misreg_img)
    ref_hwc = to_hwc(ref_img)
    h, w, n_bands = misreg_hwc.shape

     # --- Pick representative band for ECC ---
    band_idx = min(ref_band, n_bands - 1)  #safety condition, prevent IndexError
    misreg_band_img = misreg_hwc[:, :, band_idx].astype(np.float32)  #take all (h,w) for specific channel
    ref_band_img = ref_hwc[:, :, band_idx].astype(np.float32)

    
    # Normalize to [0, 1]
    def safe_norm(img):
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn + 1e-8)

    misreg_norm = safe_norm(misreg_band_img)
    ref_norm = safe_norm(ref_band_img)
    
    
    # --- Initialize warp matrix ---
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # --- ECC termination criteria ---
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)


    # --- Run ECC ---
    try:
        cc, H = cv2.findTransformECC(
            ref_norm, misreg_norm, warp_matrix, warp_mode, criteria
        )
        print(f"  ECC converged | correlation coeff: {cc:.6f}")
        success = True

    except cv2.error as e:
        print(f"  ECC failed: {e} — returning original image")    #ECC can fail: images too different, low texture, clouds, noise, etc.
        return misreg_img, None, 0.0, 0.0, False, 'ecc_failed'
    
    
    # --- Apply warp to every band individually ---
    corrected_bands = []
    for b in range(n_bands):
        band = misreg_hwc[:, :, b].astype(np.float32)
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warped = cv2.warpPerspective(
                band, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE
            )
        else:
            warped = cv2.warpAffine(
                band, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        corrected_bands.append(warped)
    

    corrected_hwc = np.stack(corrected_bands, axis=-1)
    corrected = to_original_shape(corrected_hwc, misreg_img)


    # --- Restore original dtype (clip to valid range first) ---
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        corrected = np.clip(corrected, info.min, info.max)
    corrected = corrected.astype(orig_dtype)

    H = warp_matrix
    tx, ty = H[0, 2], H[1, 2]
    
    
    # sanity check — different logic for shift vs rotation
    if is_rotation:
        # for rotation: check if homography is close to a pure rotation matrix
        # extract rotation angle from homography
        estimated_angle = abs(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
        expected_angle  = abs(misreg_val)
        if estimated_angle > 2 * expected_angle + 1:
            failure_mode = f'overcorrection: estimated={estimated_angle:.2f}° expected~{expected_angle:.2f}°'
            success = False
        elif estimated_angle < 0.3 * expected_angle and misreg_val != 0:
            failure_mode = f'undercorrection: estimated={estimated_angle:.2f}° expected~{expected_angle:.2f}°'
            success = False
    else:
        estimated_shift = np.sqrt(tx**2 + ty**2)
        expected_shift  = misreg_val * np.sqrt(2)
        if estimated_shift > 2 * expected_shift + 5:
            failure_mode = f'overcorrection: estimated={estimated_shift:.1f}px expected~{expected_shift:.1f}px'
            success = False
        elif estimated_shift < 0.3 * expected_shift and misreg_val > 0:
            failure_mode = f'undercorrection: estimated={estimated_shift:.1f}px expected~{expected_shift:.1f}px'
            success = False

    
    return corrected, H, cc, time.time()-t0, success, failure_mode
    


def compute_metrics(real_im, real_predict):
    result = {'rmse': [], 'ssim': [], 'cc': [], 'uiqi': [], 'ergas': 0}
    for i in range(6):
        result['rmse'].append(rmse(real_im[i], real_predict[i]))
        result['ssim'].append(ssim(real_im[i], real_predict[i], MAX=1.0)[0])
        result['uiqi'].append(uiqi(real_im[i], real_predict[i]))
        cc = np.sum((real_im[i]-np.mean(real_im[i]))*(real_predict[i]-np.mean(real_predict[i]))) / \
             np.sqrt(np.sum(np.square(real_im[i]-np.mean(real_im[i]))) *
                     np.sum(np.square(real_predict[i]-np.mean(real_predict[i]))) + 1e-100)
        result['cc'].append(cc)
        result['ergas'] += rmse(real_im[i], real_predict[i])**2 / (np.mean(real_im[i])**2 + 1e-100)
    result['ergas'] = np.sqrt(result['ergas'] / 6.) * 6
    result['sam']   = sam((real_im*10000.).transpose(1,2,0),
                          (real_predict*10000.).transpose(1,2,0)) * 180 / np.pi
    return {k: np.mean(v) if isinstance(v, list) else v for k, v in result.items()}


def run_swinstfm(model, images):
    C, H, W = images[0].shape
    pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE

    def pad_img(img):
        if pad_h > 0 or pad_w > 0:
            return np.pad(img, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
        return img

    images_padded = [pad_img(im) for im in images]
    H_pad = H + pad_h
    W_pad = W + pad_w

    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (H_pad - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (W_pad - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_list = list(range(0, end_h, PATCH_STRIDE))
    w_list = list(range(0, end_w, PATCH_STRIDE))
    if (H_pad - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_list.append(H_pad - PATCH_SIZE)
    if (W_pad - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_list.append(W_pad - PATCH_SIZE)

    output_image = np.zeros(images_padded[1].shape)
    image_mask   = np.ones(images_padded[1].shape)
    for i in range(4):
        image_mask[np.where(images_padded[i] < 0)]      = 0
        image_mask[np.where(images_padded[i] > 10000.)] = 0

    for i, h_start in enumerate(h_list):
        for j, w_start in enumerate(w_list):
            input_lr  = images_padded[0][:, h_start:h_start+PATCH_SIZE, w_start:w_start+PATCH_SIZE]
            target_hr = images_padded[1][:, h_start:h_start+PATCH_SIZE, w_start:w_start+PATCH_SIZE]
            ref_lr    = images_padded[2][:, h_start:h_start+PATCH_SIZE, w_start:w_start+PATCH_SIZE]
            ref_hr    = images_padded[3][:, h_start:h_start+PATCH_SIZE, w_start:w_start+PATCH_SIZE]

            input_lr, _ = transform_image(input_lr, 0, 0, 0)
            target_hr,_ = transform_image(target_hr,0, 0, 0)
            ref_lr,   _ = transform_image(ref_lr,   0, 0, 0)
            ref_hr,   _ = transform_image(ref_hr,   0, 0, 0)

            output = model(
                ref_lr.unsqueeze(0).to(device),
                ref_hr.unsqueeze(0).to(device),
                input_lr.unsqueeze(0).to(device)
            ).squeeze()

            ch_s = PATCH_SIZE // 4 if i != 0 else 0
            ch_e = PATCH_SIZE - PATCH_SIZE // 4 if i != len(h_list)-1 else PATCH_SIZE
            cw_s = PATCH_SIZE // 4 if j != 0 else 0
            cw_e = PATCH_SIZE - PATCH_SIZE // 4 if j != len(w_list)-1 else PATCH_SIZE

            dst_h_start = h_start + ch_s
            dst_h_end   = min(h_start + ch_e, output_image.shape[1])
            dst_w_start = w_start + cw_s
            dst_w_end   = min(w_start + cw_e, output_image.shape[2])

            actual_h = dst_h_end - dst_h_start
            actual_w = dst_w_end - dst_w_start

            output_image[:, dst_h_start:dst_h_end, dst_w_start:dst_w_end] = \
                output[:, ch_s:ch_s+actual_h, cw_s:cw_s+actual_w].cpu().detach().numpy()

    # crop back to original size
    output_image = output_image[:, :H, :W]
    image_mask   = image_mask[:, :H, :W]
    real_im      = (images[1] * image_mask) / 10000.
    real_predict = (output_image + 1) * 0.5 * image_mask
    return real_im, real_predict



def main():
    model = SwinSTFM()
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict({k: v for k, v in ckpt.items() if k in model.state_dict()})
    model.to(device)
    model.eval()

    csv_path = os.path.join(SAVE_DIR, 'ecc_shift_results.csv')
    csv_file  = open(csv_path, 'w', newline='')
    writer    = csv.writer(csv_file)
    writer.writerow([
        'exp_type','value', 'date', 'ecc_success', 'failure_mode',
        'ecc_time_s', 'correlation_coeff', 'tx_estimated', 'ty_estimated',
        'rmse_after', 'ssim_after', 'uiqi_after', 'cc_after', 'ergas_after', 'sam_after'
    ])

    # ── EXPERIMENT ──────────────────────────────────────────────
    for exp_type, values in EXPERIMENTS.items():
        print(f"\n=== Experiment: {exp_type.upper()} ===")

        for val in values:
            print(f"\n--- {exp_type}: {val} ---")
            for cur_date in TEST_DATES:
                images = load_image_pair(ROOT_DIR, cur_date, REF_DATE)

                if exp_type == "shift":
                    images[0] = apply_shift(images[0], int(val))
                    images[2] = apply_shift(images[2], int(val))
                    is_rotation = False
                else:
                    images[0] = apply_rotation_misregistration(images[0], val)
                    images[2] = apply_rotation_misregistration(images[2], val)
                    is_rotation = True  
                # ECC correction — correct MODIS to ref Landsat (images[3])
                corrected_input, H_input, cc_input, t_input, ok_input, fm_input = \
                    ecc_correct_image(images[0], images[3], val, is_rotation=is_rotation)
                corrected_ref, H_ref, cc_input, t_ref, ok_ref, fm_ref = \
                    ecc_correct_image(images[2], images[3], val, is_rotation=is_rotation)

                ecc_time    = t_input + t_ref
                ecc_success = ok_input and ok_ref
                failure_mode = fm_input if not ok_input else (fm_ref if not ok_ref else 'none')
                tx, ty       = H_input[0, 2], H_input[1, 2]

                # run SwinSTFM on corrected inputs
                images_corrected    = list(images)
                images_corrected[0] = corrected_input
                images_corrected[2] = corrected_ref
                real_im_after, pred_after = run_swinstfm(model, images_corrected)
                met_after = compute_metrics(real_im_after, pred_after)

                print(f'[{cur_date}] ECC={ecc_success} ({failure_mode}) '
                    f't={ecc_time:.2f}s correlation_coeff={cc_input} '
                    f'tx={tx:.2f} ty={ty:.2f}')
                print(f'After  — RMSE:{met_after["rmse"]:.4f} SSIM:{met_after["ssim"]:.4f} '
                    f'CC:{met_after["cc"]:.4f} SAM:{met_after["sam"]:.4f}')

                writer.writerow([
                    exp_type, val, cur_date, ecc_success, failure_mode,
                    f'{ecc_time:.3f}', cc_input,
                    f'{tx:.3f}', f'{ty:.3f}',
                    f'{met_after["rmse"]:.4f}', f'{met_after["ssim"]:.4f}',
                    f'{met_after["uiqi"]:.4f}', f'{met_after["cc"]:.4f}',
                    f'{met_after["ergas"]:.4f}', f'{met_after["sam"]:.4f}'
                ])

    csv_file.close()
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()