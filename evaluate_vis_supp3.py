import os
import time
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
from lib.align import estimateSimilarityTransform
from lib.utils import load_depth, get_bbox, compute_mAP, plot_mAP4


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
    """
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_dir = "results/final_transformers"
    #result_dir = 'results/eval_spd_real'
    #result_dir = "results/eval_T5_f_STAGE3_R_CAMERA_2_1_0.5/"
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    # load NOCS results
    #pkl_path = os.path.join('results/nocs_results', opt.data, 'mAP_Acc.pkl')
    """
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_dir = "vis_real_three"


    #pkl_path = os.path.join('results/eval_T5_f_STAGE3_R_CAMERA_2_1_0.5/', 'mAP_Acc.pkl')
    pkl_path = os.path.join('supp/real_ours/', 'mAP_Acc.pkl')
    with open(pkl_path, 'rb') as f:
        nocs_results = cPickle.load(f)


    pkl_path_new = os.path.join('supp/real_spd/', 'mAP_Acc.pkl')
    with open(pkl_path_new, 'rb') as f:
        nocs_results_new = cPickle.load(f)

    pkl_path_nocs = os.path.join('results/nocs_results', opt.data, 'mAP_Acc.pkl')
    with open(pkl_path_nocs, 'rb') as f:
        nocs_results_nocs = cPickle.load(f)

    nocs_iou_aps = nocs_results['iou_aps'][1, :]
    nocs_pose_aps = nocs_results['pose_aps'][1, :, :]
    nocs_iou_aps_new = nocs_results_new['iou_aps'][1, :]
    nocs_pose_aps_new = nocs_results_new['pose_aps'][1, :, :]
    nocs_iou_aps_nocs = nocs_results_nocs['iou_aps'][1, :]
    nocs_pose_aps_nocs = nocs_results_nocs['pose_aps'][1, :, :]
    iou_aps = np.concatenate((nocs_iou_aps[None, :], nocs_iou_aps_new[None, :], nocs_iou_aps_nocs[None, :]), axis=0)
    pose_aps = np.concatenate((nocs_pose_aps[None, :, :], nocs_pose_aps_new[None, :, :], nocs_pose_aps_nocs[None, :, :]), axis=0)
    plot_mAP4(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list, name = '1.png')

    nocs_iou_aps = nocs_results['iou_aps'][2, :]
    nocs_pose_aps = nocs_results['pose_aps'][2, :, :]
    nocs_iou_aps_new = nocs_results_new['iou_aps'][2, :]
    nocs_pose_aps_new = nocs_results_new['pose_aps'][2, :, :]
    nocs_iou_aps_nocs = nocs_results_nocs['iou_aps'][2, :]
    nocs_pose_aps_nocs = nocs_results_nocs['pose_aps'][2, :, :]
    iou_aps = np.concatenate((nocs_iou_aps[None, :], nocs_iou_aps_new[None, :], nocs_iou_aps_nocs[None, :]), axis=0)
    pose_aps = np.concatenate((nocs_pose_aps[None, :, :], nocs_pose_aps_new[None, :, :], nocs_pose_aps_nocs[None, :, :]), axis=0)
    plot_mAP4(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list, name = '2.png')

    nocs_iou_aps = nocs_results['iou_aps'][3, :]
    nocs_pose_aps = nocs_results['pose_aps'][3, :, :]
    nocs_iou_aps_new = nocs_results_new['iou_aps'][3, :]
    nocs_pose_aps_new = nocs_results_new['pose_aps'][3, :, :]
    nocs_iou_aps_nocs = nocs_results_nocs['iou_aps'][3, :]
    nocs_pose_aps_nocs = nocs_results_nocs['pose_aps'][3, :, :]
    iou_aps = np.concatenate((nocs_iou_aps[None, :], nocs_iou_aps_new[None, :], nocs_iou_aps_nocs[None, :]), axis=0)
    pose_aps = np.concatenate((nocs_pose_aps[None, :, :], nocs_pose_aps_new[None, :, :], nocs_pose_aps_nocs[None, :, :]), axis=0)
    plot_mAP4(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list, name = '3.png')

    nocs_iou_aps = nocs_results['iou_aps'][4, :]
    nocs_pose_aps = nocs_results['pose_aps'][4, :, :]
    nocs_iou_aps_new = nocs_results_new['iou_aps'][4, :]
    nocs_pose_aps_new = nocs_results_new['pose_aps'][4, :, :]
    nocs_iou_aps_nocs = nocs_results_nocs['iou_aps'][4, :]
    nocs_pose_aps_nocs = nocs_results_nocs['pose_aps'][4, :, :]
    iou_aps = np.concatenate((nocs_iou_aps[None, :], nocs_iou_aps_new[None, :], nocs_iou_aps_nocs[None, :]), axis=0)
    pose_aps = np.concatenate((nocs_pose_aps[None, :, :], nocs_pose_aps_new[None, :, :], nocs_pose_aps_nocs[None, :, :]), axis=0)
    plot_mAP4(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list, name = '4.png')

    nocs_iou_aps = nocs_results['iou_aps'][5, :]
    nocs_pose_aps = nocs_results['pose_aps'][5, :, :]
    nocs_iou_aps_new = nocs_results_new['iou_aps'][5, :]
    nocs_pose_aps_new = nocs_results_new['pose_aps'][5, :, :]
    nocs_iou_aps_nocs = nocs_results_nocs['iou_aps'][5, :]
    nocs_pose_aps_nocs = nocs_results_nocs['pose_aps'][5, :, :]
    iou_aps = np.concatenate((nocs_iou_aps[None, :], nocs_iou_aps_new[None, :], nocs_iou_aps_nocs[None, :]), axis=0)
    pose_aps = np.concatenate((nocs_pose_aps[None, :, :], nocs_pose_aps_new[None, :, :], nocs_pose_aps_nocs[None, :, :]), axis=0)
    plot_mAP4(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list, name = '5.png')

    nocs_iou_aps = nocs_results['iou_aps'][6, :]
    nocs_pose_aps = nocs_results['pose_aps'][6, :, :]
    nocs_iou_aps_new = nocs_results_new['iou_aps'][6, :]
    nocs_pose_aps_new = nocs_results_new['pose_aps'][6, :, :]
    nocs_iou_aps_nocs = nocs_results_nocs['iou_aps'][6, :]
    nocs_pose_aps_nocs = nocs_results_nocs['pose_aps'][6, :, :]
    iou_aps = np.concatenate((nocs_iou_aps[None, :], nocs_iou_aps_new[None, :], nocs_iou_aps_nocs[None, :]), axis=0)
    pose_aps = np.concatenate((nocs_pose_aps[None, :, :], nocs_pose_aps_new[None, :, :], nocs_pose_aps_nocs[None, :, :]), axis=0)
    plot_mAP4(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list, name = '6.png')


if __name__ == '__main__':
    print('Evaluating ...')
    evaluate()
