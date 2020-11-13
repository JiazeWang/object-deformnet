import os
import time
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
from lib.align import estimateSimilarityTransform
from lib.utils import load_depth, get_bbox, compute_mAP, plot_mAP2


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='real_test', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--model', type=str, default='results/T5_2105_three_stage_real/model_50.pth', help='resume from saved model')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
opt = parser.parse_args()

mean_shapes = np.load('assets/mean_points_emb.npy')

assert opt.data in ['val', 'real_test']
if opt.data == 'val':
    result_dir = 'results/'+str(opt.model).split('/')[1]+'_val'
    file_path = 'CAMERA/val_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
else:
    result_dir = 'results/'+str(opt.model).split('/')[1]+'_real_test'
    file_path = 'Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])


def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions

    result_dir = "vis"
    #result_dir = 'results/eval_spd_real'
    pkl_path = os.path.join('results/final_transformers/', 'mAP_Acc.pkl')
    with open(pkl_path, 'rb') as f:
        nocs_results = cPickle.load(f)
    iou_aps = nocs_results['iou_aps'][-1, :]
    pose_aps = nocs_results['pose_aps'][-1, :, :]
    #iou_aps = np.concatenate((iou_aps, nocs_iou_aps[None, :]), axis=0)
    #pose_aps = np.concatenate((pose_aps, nocs_pose_aps[None, :, :]), axis=0)
    # plot
    plot_mAP2(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)


if __name__ == '__main__':
    #print('Detecting ...')
    #detect()
    print('Evaluating ...')
    evaluate()
