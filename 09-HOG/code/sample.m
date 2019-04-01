close all
clear
run('vlfeat/toolbox/vl_setup')

[~,~,~] = mkdir('visualizations');

data_path = '../data/'; %change if you want to work with a network copy
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
% test_scn_path = fullfile(data_path,'extra_test_scenes'); %Bonus scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

% Example
image_files = dir( fullfile( train_path_pos, '*.jpg') );
im = single(imread(fullfile(train_path_pos,'/',image_files(1).name)));
cellSize = 6 ;
hog = vl_hog(im, cellSize, 'verbose') ; % Technical details http://www.vlfeat.org/api/hog.html
imhog = vl_hog('render', hog, 'verbose') ;
figure; imagesc(imhog) ; colormap gray ;
title('Image Sample - HOG');
figure; imshow(uint8(im));
title('Image Sample');
