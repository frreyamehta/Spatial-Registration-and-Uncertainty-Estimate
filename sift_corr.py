import os
import time
import csv
import numpy as np
import cv2
import torch
from scipy.ndimage import rotate as scipy_rotate
from sewar import rmse, ssim, sam

from datasets import load_image_pair, transform_image
from models import SwinSTFM

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR  = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/LGC'
CKPT_PATH = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/LGC_best.pth'
SAVE_DIR  = '/Users/medihse/Documents/Projects/CV_project/swinmod-main/swinstfm/sift_results'
os.makedirs(SAVE_DIR, exist_ok=True)

TEST_DATES = ['2004_363_Dec28', '2004_347_Dec12', '2004_331_Nov26']
REF_DATE   = '2005_013_Jan13'

EXPERIMENTS = {
    "SHIFTS"   :[0, 8, 15],       # pixel shift experiment 
    "ROTATIONS":  [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]   # degrees 
}

PATCH_SIZE = 256
IMAGE_SIZE = [2720, 3200]

# ── SIFT parameters ────────────────────────────────────────────────────────
MIN_MATCH_COUNT      = 10
LOWE_RATIO           = 0.75
RANSAC_REPROJ_THRESH = 5.0


# ── helpers ────────────────────────────────────────────────────────────────
def to_pseudorgb(image):
    """Pseudo-RGB: NIR=R, Red=G, Green=B. bands: 0=Blue,1=Green,2=Red,3=NIR"""
    r = image[3]
    g = image[2]
    b = image[1]
    return np.stack([r, g, b], axis=-1)


#def to_gray_uint8(image):
#    """Pseudo-RGB → grayscale → uint8 for SIFT."""
#    rgb = to_pseudorgb(image)
#    rgb = rgb - rgb.min()
#    rgb = rgb / (rgb.max() + 1e-8)
#    rgb_uint8 = (rgb * 255).astype(np.uint8)
#    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

def to_gray_uint8(image):
    rgb = to_pseudorgb(image)

    # FIXED scaling (don’t normalize per image)
    rgb = np.clip(rgb / 10000.0, 0, 1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    # CRITICAL: CLAHE (not simple equalization)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])

    gray = cv2.filter2D(gray, -1, kernel)

    return gray
    

def uiqi(im1, im2):
    if len(im1.shape) == 3:
        return np.array([uiqi(im1[:,:,i], im2[:,:,i]) for i in range(im1.shape[2])])
    delta_x  = np.std(im1, ddof=1)
    delta_y  = np.std(im2, ddof=1)
    delta_xy = np.sum((im1-np.mean(im1))*(im2-np.mean(im2))) / (im1.shape[0]*im1.shape[1]-1)
    mu_x, mu_y = np.mean(im1), np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y + 1e-100)
    q2 = 2*mu_x*mu_y / (mu_x**2 + mu_y**2 + 1e-100)
    q3 = 2*delta_x*delta_y / (delta_x**2 + delta_y**2 + 1e-100)
    return q1 * q2 * q3


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
            order=1
        )
    return rotated


def sift_correct_image(misreg_img, ref_img, misreg_val, is_rotation=False):
    """
    Attempt SIFT-based homography correction on a (C,H,W) misregistered image.
    Aligns misreg_img → ref_img (ref Landsat, images[3]).
    misreg_val: shift in px (if is_rotation=False) or angle in degrees (if True)
    """
    ref_band = to_gray_uint8(ref_img)
    mis_band = to_gray_uint8(misreg_img)

    print("MIS std:", np.std(mis_band))
    print("MIS unique:", len(np.unique(mis_band)))

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04, edgeThreshold=10)# limit number of keypoints
 
    kp_ref, des_ref = sift.detectAndCompute(ref_band, None)
    kp_mis, des_mis = sift.detectAndCompute(mis_band, None)

    print(f"kp_ref: {len(kp_ref)} | kp_mis: {len(kp_mis)}")

    success      = False
    failure_mode = 'none'
    corrected    = misreg_img.copy()
    H            = np.eye(3, dtype=np.float32)

    t0 = time.time()

    if des_ref is None or des_mis is None or \
       len(kp_ref) < MIN_MATCH_COUNT or len(kp_mis) < MIN_MATCH_COUNT:
        failure_mode = f'insufficient keypoints: ref={len(kp_ref) if kp_ref else 0} mis={len(kp_mis) if kp_mis else 0}'
        return corrected, H, time.time()-t0, success, failure_mode

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))
    try:
        matches = flann.knnMatch(des_mis, des_ref, k=2)
    except cv2.error as e:
        return corrected, H, time.time()-t0, success, f'FLANN error: {str(e)}'

    good_matches = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

    if len(good_matches) < MIN_MATCH_COUNT:
        failure_mode = f'insufficient good matches: {len(good_matches)} (need {MIN_MATCH_COUNT})'
        return corrected, H, time.time()-t0, success, failure_mode

    src_pts = np.float32([kp_mis[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)

    if H is None:
        return corrected, np.eye(3, dtype=np.float32), time.time()-t0, \
               False, 'homography estimation failed (RANSAC returned None)'

    inlier_count = int(mask.sum()) if mask is not None else 0
    inlier_ratio = inlier_count / len(good_matches)

    if inlier_ratio < 0.3:
        failure_mode = f'low inlier ratio: {inlier_ratio:.2f} ({inlier_count}/{len(good_matches)} matches)'
        return misreg_img, np.eye(3, dtype=np.float32), time.time()-t0, False, failure_mode

    success = True

    h, w = misreg_img.shape[1], misreg_img.shape[2]
    for c in range(misreg_img.shape[0]):
        corrected[c] = cv2.warpPerspective(
            misreg_img[c], H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

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

    return corrected, H, time.time()-t0, success, failure_mode


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

    csv_path = os.path.join(SAVE_DIR, 'sift_results.csv')
    csv_file  = open(csv_path, 'w', newline='')
    writer    = csv.writer(csv_file)
    writer.writerow([
        'exp_type','value', 'date', 'sift_success', 'failure_mode',
        'sift_time_s', 'n_good_matches', 'inlier_ratio', 'tx_estimated', 'ty_estimated',
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
                # SIFT correction — align MODIS to ref Landsat (images[3])
                corrected_input, H_input, t_input, ok_input, fm_input = \
                    sift_correct_image(images[0], images[3], val, is_rotation=is_rotation)
                corrected_ref, H_ref, t_ref, ok_ref, fm_ref = \
                    sift_correct_image(images[2], images[3], val, is_rotation=is_rotation)

                sift_time    = t_input + t_ref
                sift_success = ok_input and ok_ref
                failure_mode = fm_input if not ok_input else (fm_ref if not ok_ref else 'none')
                tx, ty       = H_input[0, 2], H_input[1, 2]

                # count good matches for logging
                ref_band = to_gray_uint8(images[3])
                mis_band = to_gray_uint8(images[0])
                sift_det = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04, edgeThreshold=10)# limit number of keypoints
    
                kp_r, d_r = sift_det.detectAndCompute(ref_band, None)
                kp_m, d_m = sift_det.detectAndCompute(mis_band, None)
                n_good = 0
                inlier_ratio = 0.0
                if d_r is not None and d_m is not None:
                    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
                    try:
                        raw  = flann.knnMatch(d_m, d_r, k=2)
                        good = [m for m, n in raw if m.distance < LOWE_RATIO * n.distance]
                        n_good = len(good)
                        if n_good >= MIN_MATCH_COUNT:
                            src = np.float32([kp_m[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                            dst = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                            _, msk = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)
                            if msk is not None:
                                inlier_ratio = float(msk.sum()) / n_good
                    except cv2.error:
                        pass

                # run SwinSTFM on corrected inputs
                images_corrected    = list(images)
                images_corrected[0] = corrected_input
                images_corrected[2] = corrected_ref
                real_im_after, pred_after = run_swinstfm(model, images_corrected)
                met_after = compute_metrics(real_im_after, pred_after)

                print(f'  [{cur_date}] SIFT={sift_success} ({failure_mode}) '
                    f't={sift_time:.2f}s matches={n_good} inliers={inlier_ratio:.2f} '
                    f'tx={tx:.2f} ty={ty:.2f}')
                print(f'    After  — RMSE:{met_after["rmse"]:.4f} SSIM:{met_after["ssim"]:.4f} '
                    f'CC:{met_after["cc"]:.4f} SAM:{met_after["sam"]:.4f}')

                writer.writerow([
                    exp_type, val, cur_date, sift_success, failure_mode,
                    f'{sift_time:.3f}', n_good, f'{inlier_ratio:.3f}',
                    f'{tx:.3f}', f'{ty:.3f}',
                    f'{met_after["rmse"]:.4f}', f'{met_after["ssim"]:.4f}',
                    f'{met_after["uiqi"]:.4f}', f'{met_after["cc"]:.4f}',
                    f'{met_after["ergas"]:.4f}', f'{met_after["sam"]:.4f}'
                ])

    csv_file.close()
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
