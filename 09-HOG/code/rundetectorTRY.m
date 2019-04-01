test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

window_size = feature_params.template_size;
cell_size = feature_params.hog_cell_size;
thresh = 0.8;
dimensions = (window_size/cell_size)^2 * 31;
for i = 1:length(test_scenes)
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);
    for scale = 1:-0.1:0.1
        rescaled_img = imresize(img,scale);
        [width,height] = size(rescaled_img);
        hog = vl_hog(rescaled_img,cell_size);
        for y = 1:floor(height/cell_size) - (window_size/cell_size) + 1
            for x = 1:floor(width/cell_size) - (window_size/cell_size) + 1
                crop_hog = hog(x:(x+cell_size-1), y:(y+cell_size-1), :);
                flat_hog = reshape(crop_hog,[1,dimensions]);
                conf = flat_hog*w + b;                
                if conf > thresh
                    x_min = ((x*cell_size)-cell_size+1)/scale;
                    y_min = ((y*cell_size)-cell_size+1)/scale;
                    x_width = x_min + window_size/scale -1;
                    y_height = y_min + window_size/scale -1;
                    cur_bboxes(end+1,:) = [x_min y_min x_width y_height];
                    cur_confidences(end+1,1) = conf;
                    cur_image_ids{end+1,1} = test_scenes(i).name;
                end
            end
        end
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




