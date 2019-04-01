% Starter code prepared by James Hays
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    bol(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

% initialize these as empty and incrementally expand them
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

% differrnt scales
scales = [1, 0.85, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.1, 0.07]; %Different scales to resize the image and obtain hog at the different scales
threshold = 0.9; %Threshold for the confidence to control the amount of faces detected. We only want the ones with a high enough confidence.

cell_size = feature_params.hog_cell_size; %Cell size of the hog cell internal subdivision
window_size = feature_params.template_size; %Window size where hog is going to subdivide into cell_size^2 cells and calculate HOG in each of those cells


for i = 1:length(test_scenes)
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    % initialize these as empty and incrementally expand them
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);
    
    % scaling
    for scale = scales
        
        % resize image
        rescaled_img = imresize(img, scale); %Rescale the original image by a factor scale to obtain the different scales of the image to calculate HOG at these different scales for faces of different sizes.
        [height, width] = size(rescaled_img); %Width and height of the rescaled image
        
        % get hog features of resized test image
        feats = vl_hog(rescaled_img, cell_size);
        
        % number of cells in test image
        num_cells = window_size / cell_size; %Since a window is of size cell_size^2 each dimension, X and Y is going to end up getting divided by a factor of num_cells
        num_window_x = floor(width / cell_size) - num_cells + 1; %The original x dimension size is divided by cell_size and since we are using windows, a "padding" must be left so that's why we subtract a cell_size and add 1
        num_window_y = floor(height / cell_size) - num_cells + 1; %The original y dimension size is divided by cell_size and since we are using windows, a "padding" must be left so that's why we subtract a cell_size and add 1
        
        % preallocate a matrix that store all features
        D = num_cells^2 * 31; %Each cell has 31 features and a window has num_cells^2 cells so the descriptor D for a window is going to be num_cells^2 * 31
        feats_in_windows = zeros(num_window_x * num_window_y, D); 
        
        % begin sliding window 
        for x = 1:num_window_x %Slide along the columns of cells
            for y = 1:num_window_y %Slide along the rows of cells
                windows = feats(y:(y+num_cells-1), x:(x+num_cells-1),:); %A sliding window of size num_cells over the HOG output
                temp_pos = (x-1)*num_window_y + y; %Allocation temp variable
                feats_in_windows(temp_pos,:) = reshape(windows, 1, D); %Since we obtain a window of size num_cells^2 with each cell having 31 dimensions, we have to flatten it for the future calculation using w
            end
        end
        
        % calculate the scores of all features and find the confidence is larger than threshold
        confidence = feats_in_windows * w + b; %Confidence for each window
        index = find(confidence > threshold); %Index where a confidence is higher than the defined threshold
        thresholded_confidences = confidence(index); %Only high enough confidences are taken into account for the bounding boxes
        
        % calculate the coordinates of bbox
        x = floor(index./num_window_y)+1; %X coordinate of the cell where the sliding window began
        y = mod(index, num_window_y)-1; %Y coordinate of the cell where the sliding window began
        top_left_x = (x * cell_size)/ scale; %The minimum x is the x multiplied by cell_size to transform back from cell units to pixel units and divided by the scale since we rescaled the original image to obtain this coordinate
        top_left_y = (y * cell_size + 1) / scale; %The minimum y is the y multiplied by cell cell_size to transform back from cell units to pixel units and divided by the scale since we rescaled the original image to obtain this coordinate
        down_right_x = top_left_x + (window_size / scale); %The top left x coordinate is then added with the window_size but this divided once again since the window is also affected by the scale of the image
        down_right_y = top_left_y + (window_size / scale); %The top left y coordinate is then added with the window_size but this divided once again since the window is also affected by the scale of the image
        scaled_bboxes = [top_left_x, top_left_y, down_right_x, down_right_y];
        
        % record the image id and index of window
        thresholded_image_ids = repmat({test_scenes(i).name}, size(index,1), 1);
       
        % record the current bbox coordinate and confidence
        cur_bboxes      = [cur_bboxes;      scaled_bboxes];
        cur_confidences = [cur_confidences; thresholded_confidences];
        cur_image_ids   = [cur_image_ids;   thresholded_image_ids];
    end
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end
end


