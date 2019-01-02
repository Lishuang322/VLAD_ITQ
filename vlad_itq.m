clear,clc,
close all
%parameter settings
CENTROID_NUM = 64 ;%number of visual words used in vlad
%CODE_LENGTH = 256 ;%binary code length
repo = '/data/data-hulishuang/img-dataset/oxford/';%directory of images which need to encode
filelist = dir([repo '*.jpg']);%suffix for your input images
%centroid_path = 'clust_flickr60_k100.fvecs' ;%directory of centroids
%output_path = 'paris_c64_l256.csv';%output image name

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
%v = fvecs_read (centroid_path);
%centroids = v(1:CENTROID_NUM,:);
%centroids = centroids';
%kdtree = vl_kdtreebuild(centroids);

%step2:cluster the SIFT features of all images using k-means clustering
all_descr = single([sift_descr{:}]);
centroids = vl_kmeans(all_descr, CENTROID_NUM);
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
code_128=compressITQ(enc',128);
code_256=compressITQ(enc',256);
code_512=compressITQ(enc',512);

code_cell_128 = num2cell(code_128);
code_cell_256 = num2cell(code_256);
code_cell_512 = num2cell(code_512);

output_128 = [image_name' code_cell_128];
output_256 = [image_name' code_cell_256];
output_512 = [image_name' code_cell_512];


%%%%%need modify
cell2csv('oxford_c64_l128.csv',output_128,' ');
cell2csv('oxford_c64_l256.csv',output_256,' ');
cell2csv('oxford_c64_l512.csv',output_512,' ');



