import math
import numpy as np
from operator import itemgetter
from utils.config import config

def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    heatmap[heatmap < config['map_filter_thresh']] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i+1, len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                         (keypoints[i][1] - keypoints[j][1]) ** 2) < config['map_filter_thresh']:
                suppressed[j] = 1
        keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],
                                      total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def connections_nms(a_idx, b_idx, affinity_scores):
    # From all retrieved connections that share the same starting/ending keypoints leave only the top-scoring ones.
    order = affinity_scores.argsort()[::-1]
    affinity_scores = affinity_scores[order]
    a_idx = a_idx[order]
    b_idx = b_idx[order]
    idx = []
    has_kpt_a = set()
    has_kpt_b = set()
    for t, (i, j) in enumerate(zip(a_idx, b_idx)):
        if i not in has_kpt_a and j not in has_kpt_b:
            idx.append(t)
            has_kpt_a.add(i)
            has_kpt_b.add(j)
    idx = np.asarray(idx, dtype=np.int32)
    return a_idx[idx], b_idx[idx], affinity_scores[idx]

# pose entry size = number of keypoints + 2 (confidence and number of good points)
def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=config['keypoint_number'], min_paf_score=0.05):
    pose_entry_size += 2
    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    points_per_limb = config['points_per_limb']
    grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)
    all_keypoints_by_type = [np.array(keypoints, np.float32) for keypoints in all_keypoints_by_type]
    for part_id in range(len(config['body_parts_paf_ids'])):
        part_pafs = pafs[:, :, config['body_parts_paf_ids'][part_id]]
        kpts_a = all_keypoints_by_type[config['body_parts_kpt_ids'][part_id][0]]
        kpts_b = all_keypoints_by_type[config['body_parts_kpt_ids'][part_id][1]]
        n = len(kpts_a)
        m = len(kpts_b)
        if n == 0 or m == 0:
            continue

        # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
        a = kpts_a[:, :2]
        a = np.broadcast_to(a[None], (m, n, 2))
        b = kpts_b[:, :2]
        vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2) 

        # Sample points along every candidate limb vector.
        steps = (1 / (points_per_limb - 1) * vec_raw)
        points = steps * grid + a.reshape(-1, 1, 2)
        # points = grid + a.reshape(-1, 1, 2)
        points = points.round().astype(dtype=np.int32)
        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        # Compute affinity score between candidate limb vectors and part affinity field.
        field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
        vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
        vec = vec_raw / (vec_norm + 1e-6)
        affinity_scores = (field * vec).sum(-1).reshape(-1, points_per_limb)

        valid_affinity_scores = affinity_scores > min_paf_score
        valid_num = valid_affinity_scores.sum(1)
        affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6) #1e-6 is an epsilon
        success_ratio = valid_num / points_per_limb

        # Get a list of limbs according to the obtained affinity score.
        valid_limbs = np.where(np.logical_and(affinity_scores > config['affinity_score'], 
                                              success_ratio > config['success_ratio']))[0]
        if len(valid_limbs) == 0:
            continue
        b_idx, a_idx = np.divmod(valid_limbs, n)
        affinity_scores = affinity_scores[valid_limbs]
        
        # Suppress incompatible connections.
        a_idx, b_idx, affinity_scores = connections_nms(a_idx, b_idx, affinity_scores)
        connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                               kpts_b[b_idx, 3].astype(np.int32),
                               affinity_scores))

        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][config['body_parts_kpt_ids'][0][0]] = connections[i][0]
                pose_entries[i][config['body_parts_kpt_ids'][0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        else:
            kpt_a_id = config['body_parts_kpt_ids'][part_id][0]
            kpt_b_id = config['body_parts_kpt_ids'][part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 1 or (pose_entries[i][-2] / pose_entries[i][-1] < config['valid_pose_thresh']):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints
