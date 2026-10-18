import os
import random
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


# ADDED: helper for rotational misregistration
def apply_rotation_misregistration(image, angle):
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


def test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE):
    cur_result = {}
    model.eval()

    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    # ADDED: loop over rotation angles
    #rotation_angles = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    rotation_angles = [0.0]
    for rotation_angle in rotation_angles:                        # ADDED
        #print(f'\n=== Rotational misregistration: {rotation_angle:+.1f}° ===')  # ADDED

    # so 3 dates kept for testing in which 347 pred target and 331 and 363 is ref dates
    # so 2 pred made and then they are averaged
        for cur_date in test_dates:
            ref_date = '2002_101_11apr'
            images = load_image_pair(opt.root_dir, cur_date, ref_date)

            # ORIGINAL: pixel synthetic misregistration to MODIS input (commented out)
            shift = 15
            save_dir = '/Users/sharo2/Downloads/projects/swinmod/swinstfm/fusedimg'
            if shift > 0:
                shifted = np.zeros_like(images[0])
                shifted[:, shift:, shift:] = images[0][:, :-shift, :-shift]
                images[0] = shifted
                shifted = np.zeros_like(images[2])
                shifted[:, shift:, shift:] = images[2][:, :-shift, :-shift]
                images[2] = shifted

            # ADDED: rotational misregistration on MODIS inputs only
            # images[0] = apply_rotation_misregistration(images[0], rotation_angle)
            # images[2] = apply_rotation_misregistration(images[2], rotation_angle)

            output_image = np.zeros(images[1].shape)
            image_mask = np.ones(images[1].shape)
            for i in range(4):
                            negtive_mask = np.where(images[i] < 0)
                            inf_mask = np.where(images[i] > 10000.)
                            image_mask[negtive_mask] = 0
                            image_mask[inf_mask] = 0

            for i in range(len(h_index_list)):
                            for j in range(len(w_index_list)):
                                h_start = h_index_list[i]
                                w_start = w_index_list[j]

                                input_lr = images[0][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                                target_hr = images[1][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                                ref_lr = images[2][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                                ref_hr = images[3][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]

                                flip_num = 0
                                rotate_num0 = 0
                                rotate_num = 0
                                input_lr, im_mask = transform_image(input_lr, flip_num, rotate_num0, rotate_num)
                                target_hr, im_mask = transform_image(target_hr, flip_num, rotate_num0, rotate_num)
                                ref_lr, im_mask = transform_image(ref_lr, flip_num, rotate_num0, rotate_num)
                                ref_hr, im_mask = transform_image(ref_hr, flip_num, rotate_num0, rotate_num)

                                input_lr = input_lr.unsqueeze(0).to(device)
                                ref_lr = ref_lr.unsqueeze(0).to(device)
                                ref_hr = ref_hr.unsqueeze(0).to(device)

                                output = model(ref_lr, ref_hr, input_lr)
                                output = output.squeeze()

                                h_end = h_start + PATCH_SIZE
                                w_end = w_start + PATCH_SIZE
                                cur_h_start = 0
                                cur_h_end = PATCH_SIZE
                                cur_w_start = 0
                                cur_w_end = PATCH_SIZE

                                if i != 0:
                                    h_start = h_start + PATCH_SIZE // 4
                                    cur_h_start = PATCH_SIZE // 4

                                if i != len(h_index_list) - 1:
                                    h_end = h_end - PATCH_SIZE // 4
                                    cur_h_end = cur_h_end - PATCH_SIZE // 4

                                if j != 0:
                                    w_start = w_start + PATCH_SIZE // 4
                                    cur_w_start = PATCH_SIZE // 4

                                if j != len(w_index_list) - 1:
                                    w_end = w_end - PATCH_SIZE // 4
                                    cur_w_end = cur_w_end - PATCH_SIZE // 4

                                output_image[:, h_start: h_end, w_start: w_end] = \
                                    output[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()

            real_im = images[1] * image_mask
            real_predict = (output_image + 1) * 0.5 * image_mask

            cur_result['rmse'] = []
            cur_result['ssim'] = []
            cur_result['cc'] = []
            cur_result['uiqi'] = []
            cur_result['ergas'] = 0

            for i in range(6):
                                cur_result['rmse'].append(rmse(real_im[i], real_predict[i]))
                                cur_result['ssim'].append(ssim(real_im[i], real_predict[i], MAX=1.0)[0])
                                cur_result['uiqi'].append(uiqi(real_im[i], real_predict[i]))
                                cur_cc = np.sum((real_im[i] - np.mean(real_im[i])) * (real_predict[i] - np.mean(real_predict[i]))) / \
                                         np.sqrt((np.sum(np.square(real_im[i] - np.mean(real_im[i])))) * np.sum(
                                             np.square(real_predict[i] - np.mean(real_predict[i]))) + 1e-100)
                                cur_result['cc'].append(cur_cc)
                                cur_result['ergas'] += rmse(real_im[i], real_predict[i]) ** 2 / (np.mean(real_im[i]) ** 2 + 1e-100)

            cur_result['ergas'] = np.sqrt(cur_result['ergas'] / 6.) * 6

            cur_im = real_im * 10000.
            cur_predict = real_predict * 10000.

            cur_result['sam'] = sam(cur_im.transpose(1, 2, 0), cur_predict.transpose(1, 2, 0)) * 180 / np.pi
            print('[%s/%s] RMSE: %.4f SSIM: %.4f UIQI: %.4f CC: %.4f ERGAS: %.4f SAM: %.4f' % (
                                cur_date, ref_date, np.mean(np.array(cur_result['rmse'])),
                                np.mean(np.array(cur_result['ssim'])), np.mean(np.array(cur_result['uiqi'])),
                                np.mean(np.array(cur_result['cc'])), cur_result['ergas'], cur_result['sam']
                            ))

            final_im = (real_predict * 10000.).astype(np.int16)
            metadata = {
                                    'driver': 'GTiff',
                                    'width': final_im.shape[2],
                                    'height': final_im.shape[1],
                                    'count': final_im.shape[0],
                                    'dtype': np.int16
                                }
            im_name = os.path.join(save_dir, f'swin_CIA_{cur_date}_shift{shift}px.tif')  
            print(f'Saving: {im_name}')
            with rasterio.open(im_name, mode='w', **metadata) as dst:
                for i in range(final_im.shape[0]):
                    dst.write(final_im[i], i + 1)
            print(f'Saved successfully.')
            
            if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)

            # ORIGINAL save name (commented out)
            #im_name = os.path.join(save_dir, f'swin_CIA_{cur_date}.tif')

            # ADDED: angle tag in filename so each run doesn't overwrite
            #angle_tag = f'rot{"p" if rotation_angle >= 0 else "n"}{abs(rotation_angle):.1f}'.replace('.', '_')  # ADDED
                                 

            with rasterio.open(im_name, mode='w', **metadata) as dst:
                                    for i in range(final_im.shape[0]):
                                        dst.write(final_im[i], i + 1)


def train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE):

    model_G = SwinSTFM()
    G_dict = model_G.state_dict()
    model_CKPT = torch.load('/Users/sharo2/Downloads/projects/swinstfm/experiment_best_cia/epoch_best.pth', map_location=device)
    pretained_dict = {k: v for k, v in model_CKPT.items() if k in G_dict}

    G_dict.update(pretained_dict)
    model_G.load_state_dict(G_dict)
    model_G.to(device)
    test(opt, model_G, test_dates, IMAGE_SIZE, PATCH_SIZE)


def main():
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--image_size', default=[2040, 1720], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--num_epochs', default=60, type=int, help='train epoch number')
    parser.add_argument('--root_dir', default='/Users/sharo2/Downloads/projects/cia/CIA_npy', help='Datasets root directory')
    parser.add_argument('--train_dir', default='/Users/sharo2/Downloads/projects/cia/CIA_Train', help='Datasets train directory')

    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size
    NUM_EPOCHS = opt.num_epochs

    test_dates = ['2002_108_18apr', '2002_117_27apr', '2002_124_04may']

    train(opt, [], test_dates, IMAGE_SIZE, PATCH_SIZE)


if __name__ == '__main__':
    main()