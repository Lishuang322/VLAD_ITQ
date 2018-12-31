%step1:extract sift discriptors
CENTROID_NUM = 64 ;
repo = '/home/hulishuang/workspace/vlad/pictures/';
filelist = dir([repo '*.jpg']);
sift_descr = {};

for i = 1:size(filelist, 1)
    I = imread([repo filelist(i).name]) ;
    I = single(rgb2gray(I)) ;
    [f,d] = vl_sift(I) ;
    sift_descr{i} = d;
end
%step2:load visual dictionary
v = fvecs_read ("clust_flickr60_k100.fvecs");
centroids = v(1:CENTROID_NUM,:);
centroids = centroids';
kdtree = vl_kdtreebuild(centroids);
%step3:comput vlad discriptors
%%%modify later
enc = zeros(CENTROID_NUM*128, numel(sift_descr));

for k=1:numel(sift_descr)

    % Create assignment matrix
    nn = vl_kdtreequery(kdtree, centroids, single(sift_descr{k}));
    %%%modify later
    assignments = zeros(CENTROID_NUM, numel(nn), 'single');
    assignments(sub2ind(size(assignments), nn, 1:numel(nn))) = 1;

    % Encode using VLAD
    enc(:, k) = vl_vlad(single(sift_descr{k}), centroids, assignments);
end