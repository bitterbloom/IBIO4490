%Demo code
clear all
clc

[~,~,~] = mkdir('visualizations');
test_scn_path = fullfile('../images/demo'); %Demo pictures
w = load('w.mat');
w = w.W;
b = load('b.mat');
b = b.B;
feature_params = struct('template_size', 36, 'hog_cell_size', 6);

[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);
visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path)