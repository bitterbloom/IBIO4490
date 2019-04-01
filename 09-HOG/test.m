%test code
clear all
clc

addpath('./code/') 
[~,~,~] = mkdir('visualizations');
test_scn_path = fullfile('./images/test_scenes/test_jpg'); %CMU+MIT test scenes
w = load('w.mat');
w = w.W;
b = load('b.mat');
b = b.B;
feature_params = struct('template_size', 36, 'hog_cell_size', 6);
label_path = fullfile('./images/test_scenes/ground_truth_bboxes.txt');

[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)