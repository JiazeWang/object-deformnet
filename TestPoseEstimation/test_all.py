import os
import time
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from lib.network_spd import DeformNet
#from lib.network_t5_eval import DeformNet
from lib.align import estimateSimilarityTransform
from lib.utils import load_depth, get_bbox, compute_mAP, plot_mAP
from lib.utils import draw_detections2

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='val', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='../data', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--model', type=str, default='results/camera_more/model_50.pth', help='resume from saved model')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
opt = parser.parse_args()

mean_shapes = np.load('assets/mean_points_emb.npy')

# CAMERA
# cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
# REAL
assert opt.data in ['val', 'real_test']
if opt.data == 'val':
    result_dir = 'results/eval_T3_STAGE3_R_CAMERA_2_1_0.5'
    file_path = 'CAMERA/val_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
else:
    result_dir = 'results/eval_T3_STAGE3_R_CAMERA_2_1_0.5'
    file_path = 'Real/test_list.txt'
    cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084

intrinsics = np.array([[cam_fx, 0.0, cam_cx], [0.0, cam_fy, cam_cy], [0.0, 0.0, 1.0]], dtype=np.float)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def load_depth2(img_path):
    depth = cv2.imread(img_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def single_detect(estimator, raw_rgb, depth, segmentation):
    '''
    input:
        1. model file
        2. RGB image file
        3. depth file
        4. mask r-cnn segmentation result
    '''
    raw_rgb = raw_rgb[:, :, ::-1]
    # number of instances
    num_insts = len(segmentation['class_ids'])
    f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
    f_size = np.zeros((num_insts, 3), dtype=float)
    valid_inst = []
    f_points, f_rgb, f_choose, f_catId, f_prior = [], [], [], [], []

    for i in range(num_insts):
        cat_id = segmentation['class_ids'][i] - 1

        prior = mean_shapes[cat_id]
        rmin, rmax, cmin, cmax = get_bbox(segmentation['rois'][i])
        mask = np.logical_and(segmentation['masks'][:, :, i], depth > 0)
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) < 32:
            f_sRT[i] = np.identity(4, dtype=float)
            f_size[i] = 2 * np.amax(np.abs(prior), axis=0)
            continue
        else:
            valid_inst.append(i)
        # process objects with valid depth observation
        if len(choose) > opt.n_pts:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:opt.n_pts] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, opt.n_pts-len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / norm_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)
        rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (opt.img_size, opt.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = norm_color(rgb)
        crop_w = rmax - rmin
        ratio = opt.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * opt.img_size + np.floor(col_idx * ratio)).astype(np.int64)

        # concatenate instances
        f_points.append(points)
        f_rgb.append(rgb)
        f_choose.append(choose)
        f_catId.append(cat_id)
        f_prior.append(prior)

    if len(valid_inst):
        f_points = torch.cuda.FloatTensor(f_points)
        f_rgb = torch.stack(f_rgb, dim=0).cuda()
        f_choose = torch.cuda.LongTensor(f_choose)
        f_catId = torch.cuda.LongTensor(f_catId)
        f_prior = torch.cuda.FloatTensor(f_prior)

        assign_mat, deltas = estimator(f_points, f_rgb, f_choose, f_catId, f_prior)

        inst_shape = f_prior + deltas
        assign_mat = F.softmax(assign_mat, dim=2)
        f_coords = torch.bmm(assign_mat, inst_shape)  # bs x n_pts x 3

        f_coords = f_coords.detach().cpu().numpy()
        f_points = f_points.cpu().numpy()
        f_choose = f_choose.cpu().numpy()
        f_insts = inst_shape.detach().cpu().numpy()

        for i in range(len(valid_inst)):
            inst_idx = valid_inst[i]
            choose = f_choose[i]
            _, choose = np.unique(choose, return_index=True)
            nocs_coords = f_coords[i, choose, :]
            f_size[inst_idx] = 2 * np.amax(np.abs(f_insts[i]), axis=0)
            points = f_points[i, choose, :]
            _, _, _, pred_sRT = estimateSimilarityTransform(nocs_coords, points)
            if pred_sRT is None:
                pred_sRT = np.identity(4, dtype=float)
            f_sRT[inst_idx] = pred_sRT
    return {'predict_RT': f_sRT, 'predict_Size': f_size, 'predict_Category': f_catId}

def detect():
    model_path = "lib/spd_camera.pth"
    estimator = DeformNet(opt.n_cat, opt.nv_prior)
    estimator.cuda()
    estimator.load_state_dict(torch.load(model_path))
    estimator = nn.DataParallel(estimator)
    #estimator.load_state_dict(torch.load(model_path))
    estimator.eval()
    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data_dir, file_path))]
    # frame by frame test
    t_inference = 0.0
    t_umeyama = 0.0
    inst_count = 0
    img_count = 0
    t_start = time.time()
    for path in tqdm(img_list):
        img_path = os.path.join(opt.data_dir, path)
        rgbimg_path = cv2.imread(img_path + '_color.png')
        depth_path = img_path + '_depth.png'
        img_path_parsing = img_path.split('/')
        segmentation_path = os.path.join('../results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        raw_rgb = rgbimg_path[:, :, :3]
        raw_depth = load_depth2(depth_path)

        with open(segmentation_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)

        results = single_detect(estimator, raw_rgb, raw_depth, mrcnn_result)
        gt = {}
        gt['gt_class_ids'] = gts['class_ids']
        gt['gt_RTs'] = gts['poses']
        gt['gt_scales'] = gts['size']
        name = path.split('/')
        savename = name[-3]+'_'+name[-2]+'_'+name[-1]
        #print(savename)
        visualize('./spd_camera_vis/', savename, '0', raw_rgb, intrinsics, results, gt)

    """
    rgbimg_path = "data/0001_color.png"
    depth_path = "data/0001_depth.png"
    segmentation_path = "data/results_test_scene_5_0001.pkl"
    # segmentation_path = "data/results_test_scene_4_0004.pkl"
    gt_path = "data/0001_label.pkl"
    with open(gt_path, 'rb') as f:
            gts = cPickle.load(f)

    raw_rgb = cv2.imread(rgbimg_path)[:, :, :3]
    # raw_rgb[:,:,1] = raw_rgb[:,:,0]
    # raw_rgb[:,:,2] = raw_rgb[:,:,0]
    raw_depth = load_depth2(depth_path)

    with open(segmentation_path, 'rb') as f:
        mrcnn_result = cPickle.load(f)

    results = single_detect(estimator, raw_rgb, raw_depth, mrcnn_result)
    gt = {}
    gt['gt_class_ids'] = gts['class_ids']
    gt['gt_RTs'] = gts['poses']
    gt['gt_scales'] = gts['size']

    visualize('./data', 'test', 'real3', raw_rgb, intrinsics, results, gt)
    """

def visualize(dir, name, id, rgb_img, intrinsics, estimation, gt):
    '''
    input:
        1. Output dir
        2. Output name
        3. Output ID
        4. RGB image for display
        5. Estimation results
        6. Ground truth
    '''
    draw_detections2(rgb_img, dir, name, id, intrinsics, estimation['predict_RT'], estimation['predict_Size'], \
        estimation['predict_Category'], gt['gt_RTs'], gt['gt_scales'],
        gt['gt_class_ids'], None, None, None, False, False)


if __name__ == '__main__':

    detect()
