import argparse
import cv2
import json
import math
import numpy as np
import torch

from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet, PoseEstimationWithMobileNetV3
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from utils.config import config
from tabulate import tabulate


def precision(tp, fp):
    if tp == 0 and fp == 0:
        return 0
    return float(tp)/ float(tp+fp)

def recall(tp, fn):
    if fn == 0 and tp == 0:
        return 0
    return float(tp)/float(tp+fn)

def f1(tp, fp, fn):
    if precision(tp, fp) == 0 or recall(tp, fn) == 0:
        return 0
    return 2 * ((precision(tp, fp) * recall(tp, fn)) / (precision(tp, fp) + recall(tp, fn)) )

def AP(prec, rec):
    return np.sum((rec[:-1] - rec[1:]) * prec[:-1])

def oks(y_true, y_pred, visibility, sigmas, scale=1.):
    # Compute the L2/Euclidean Distance
    distances = np.linalg.norm(y_pred - y_true, axis=-1)
    # Compute the exponential part of the equation
    exp_vector = np.exp(-(distances**2) / (2 * (scale**2) * (sigmas**2)))
    # The numerator expression
    numerator = np.dot(exp_vector, visibility.astype(bool).astype(int))
    # The denominator expression
    denominator = np.sum(visibility.astype(bool).astype(int))
    return numerator / denominator

def compute_tp_fn_tn(oks_value, tps, fps, inc_precision, inc_recall, thresholds, total_samples):
    for t in thresholds:
        if oks_value >= t:
            tps[t] += 1
        else:
            fps[t] += 1
        inc_precision[t].append(precision(tps[t], fps[t]))
        inc_recall[t].append(tps[t]/total_samples)
    return tps, fps, inc_precision, inc_recall


def run_eval(gt_file_path, dt_file_path, thresholds = config['thresholds']):
    tps = {t: 0 for t in thresholds}
    fps = {t: 0 for t in thresholds}
    inc_precision = {t: [] for t in thresholds}
    inc_recall = {t: [] for t in thresholds}
    fns = 0

    f = open(gt_file_path)
    coco_gt = json.load(f)['annotations']
    f.close()
    f = open(dt_file_path)
    coco_dt = json.load(f)
    f.close()
    image_ids = [image['image_id'] for image in coco_dt]
    coco_gt = [image for id in image_ids for image in coco_gt if image['image_id'] == id]
    for gt, dt in zip(coco_gt, coco_dt):
        gt_keypoints = gt['keypoints']
        dt_keypoints = dt['keypoints']
        dt_keypoints = [[dt_keypoints[i], dt_keypoints[i+1]] for i in range(0, len(dt_keypoints), 3)]
        visibility = []
        gt_kpts = []
        for i in range(0, len(gt_keypoints), 3):
            gt_kpts.append([gt_keypoints[i], gt_keypoints[i+1]])
            visibility.append([gt_keypoints[i+2]])
        
        if [0, 0] not in dt_keypoints:
            oks_value = oks(np.array(gt_kpts), np.array(dt_keypoints), np.array(visibility), config['sigmas'])
            tps, fps, inc_precision, inc_recall = compute_tp_fn_tn(oks_value, tps, fps, inc_precision, inc_recall, thresholds, len(coco_dt))
        else:
            fns += 1
    
    data = []
    columns = ['Threshold', 'Precision', 'Recall', 'F1-Score', 'Average Precision']
    for t in thresholds:
        prec = inc_precision[t]
        rec = inc_recall[t]
        prec.reverse()
        rec.reverse()
        local_data = [t, precision(tps[t], fps[t]), recall(tps[t], fns),
                      f1(tps[t], fps[t], fns), AP(np.array(prec), np.array(rec))]
        data.append(local_data)
    print(tabulate(data, headers=columns, tablefmt="fancy_grid"))

def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    to_coco_map = config['reorder_map']
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * config['keypoint_number'] * 3
        object_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in range(config['keypoint_number']):
            position_id += 1
            cx, cy, visibility = 0, 0, 0
            if pose_entries[n][keypoint_id] != -1.0:
                cx = int(all_keypoints[int(pose_entries[n][keypoint_id]), 0])
                cy = int(all_keypoints[int(pose_entries[n][keypoint_id]), 1])
                visibility = 2
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(object_score * max(0, (pose_entries[n][-1]))) 
    return coco_keypoints, scores


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, _, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]

    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def evaluate(labels, output_name, images_folder, net, visualize=False):
    net = net.cuda().eval()
    stride = config['stride']

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []
    upsample_ratio = 4
    height_size = config['input_size']
    for sample in dataset:
        file_name = sample['file_name']
        img = sample['img']

        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(config['keypoint_number']):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 24, 
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey()
            if key == 27:  # esc
                return

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    if coco_result:
      run_eval(labels, output_name)
    else:
      print("No detections")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNetV3()
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    evaluate(args.labels, args.output_name, args.images_folder, net, args.multiscale, args.visualize)
