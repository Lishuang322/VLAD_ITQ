% Read a set of cluster centroids stored in the fvec format
%
% Syntax: 
%   v = fvecs_read (filename)
function v = fvecs_read (filename)

% open the file and count the number of descriptors
fid = fopen (filename, 'rb');
 
% Read the vector size
d = fread (fid, 1, 'int');

fseek (fid, 0, 1);
n = ftell (fid) / (1 * 4 + d * 4);
fseek (fid, 0, -1);
  
% first read the meta information associated with the descriptor
v = zeros (n, d, 'single');

% read the elements
for i = 1:n
  d = fread (fid, 1, 'int');
  v(i,:) = fread (fid, d, 'float');
end

fclose (fid);