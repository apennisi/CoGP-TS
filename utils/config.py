import numpy as np

config = {}
config['epochs'] = 1000
config['keypoint_number'] = 6
config['input_size'] = 384
config['stride'] = 16
config['sigma'] = 7 #for the gaussian distribution
config['upsample_ratio'] = 4
config['gaussian_sigma'] = 7
config['path_thickness'] = 1
config['drop_after_epoch'] = [100, 200, 260]
config['checkpoint_folder'] = 'checkpoints_bananone/'
config['num_refinement_steps'] = 1

#set the maximum number of point per connection
config['points_per_limb'] = 2

#this configuration has to be done according to reorder_map, if your reorder_map and to_coco_map are the same, you can use the same annotation of your annotations
config['body_parts_kpt_ids'] = [[0, 1], [0, 2], [0, 4], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5]]
#pafs represent the vectors connecting the keypoints, each vector is represented as a 2 way out tensor, one for x and one for y
config['body_parts_paf_ids'] = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
config['paf_number'] = 8
config['reorder_map'] = [0, 1, 2, 3, 4, 5] # the order of points is: [right, bottom, left, center, top], we convert it to [center, right, bottom, left, top]
config['to_coco_map'] = [0, 1, 2, 3, 4, 5]
config['thresholds'] = list(np.arange(.5, 1., .05))

# Augmentation
config['swap_right'] = [1]
config['swap_left'] = [2]
config['swap_up'] = [0]
config['swap_down'] = [5]

# Map filtering
config['map_filter_thresh'] = 0.1 # threshold below which the heatmap is set to 0
config['affinity_score'] = 0
config['success_ratio'] = 0.8
config['valid_pose_thresh'] = 0.2

# Pose
config['kpt_names'] = ['center', 'right',
                 'bottom', 'left', 'top', 'left_again']
# config['sigmas'] = np.array([.2, .2, .2, .2, .2],
#                       dtype=np.float32) / 10.0
config['sigmas'] = np.array([20, 20, 20, 20, 20, 20]) #per-keypoint constant that controls falloff

