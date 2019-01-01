clear,clc,
close all
%parameter settings
CENTROID_NUM = 64 ;%number of visual words used in vlad
CODE_LENGTH = 256 ;%binary code length
repo = '/home/hulishuang/workspace/vlad/pictures/';%directory of images which need to encode
filelist = dir([repo '*.jpg']);%suffix for your input images
centroid_path = 'clust_flickr60_k100.fvecs' ;%directory of centroids
output_path = 'image_code_256.csv';%output image name

%step1:extract sift discriptors
sift_descr = {};
image_name = {};
for i = 1:size(filelist, 1)
    I = imread([repo filelist(i).name]) ;
    I = single(rgb2gray(I)) ;
    [f,d] = vl_sift(I) ;
    sift_descr{i} = d;
    image_name{i} = filelist(i).name;
end

%step2:load visual dictionary
v = fvecs_read (centroid_path);
centroids = v(1:CENTROID_NUM,:);
centroids = centroids';
kdtree = vl_kdtreebuild(centroids);

%step3:comput vlad discriptors
enc = zeros(CENTROID_NUM*128, numel(sift_descr));

for k=1:numel(sift_descr)

    % Create assignment matrix
    nn = vl_kdtreequery(kdtree, centroids, single(sift_descr{k}));
    assignments = zeros(CENTROID_NUM, numel(nn), 'single');
    assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

    % Encode using VLAD
    enc(:, k) = vl_vlad(single(sift_descr{k}), centroids, assignments);
end

%step4:comput ITQ binary code
code=compressITQ(enc',CODE_LENGTH);
code_cell = num2cell(code);
output = [image_name' code_cell];
cell2csv(output_path,output,' ');