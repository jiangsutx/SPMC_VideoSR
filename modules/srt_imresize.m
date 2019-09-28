function imgOut = srt_imresize(imgIn, newSize, method)
%srt_imresize - Open source version of imresize
%
%   e.g.,
%   method = 'bicubic'; % can be 'nearest','bilinear','lanczos2/3/4'
%   img1 = rand(480, 640, 3);
%   img2 = srt_imresize(img1, [240,320], method);
%   ----------------------------------------------------------------------
%   NOTE!!!!!!!
%      for multi image SR, MATLAB's impl of imresize has problem, I just
%      changed one sentense to: u = x/scale + 1 - 1/scale; in line#154
%
%   Created by Ziyang Ma 2013.05.19, 19:00

% Set parameter defaults.

if newSize(1)==size(imgIn,1) && newSize(2)==size(imgIn,2)
    imgOut = imgIn;
    return;
end

[~,kernel,kernel_width] = getMethodInfo(method);

if strcmp(method,'box')
    antialiasing = 0;
elseif strcmp(method,'nearest')
    antialiasing = 0;
else
    antialiasing = 1;
end

scale = [newSize(1), newSize(2)] ./ [size(imgIn,1), size(imgIn,2)];
output_size = newSize;

A = imgIn;

% Calculate interpolation weights and indices for each dimension.
weights = cell(1,2);
indices = cell(1,2);
for k = 1:2
    [weights{k}, indices{k}] = contributions(size(A, k), ...
        output_size(k), scale(k), kernel, ...
        kernel_width, antialiasing);
end

%B = resizeAlongDim(A, 1, weights{1}, indices{1});
%B = resizeAlongDim(B, 2, weights{2}, indices{2});
B = imresizemex_dim1(A, weights{1}, indices{1});
B = imresizemex_dim2(B, weights{2}, indices{2});

imgOut = B;

%=====================================================================
function [name,kernel,width] = getMethodInfo(method)

% Original implementation of getMethodInfo returned this information as
% a single struct array, which was somewhat more readable. Replaced
% with three separate arrays as a performance optimization. -SLE,
% 31-Oct-2006
names = {'nearest', 'bilinear', 'bicubic', 'box', ...
                    'triangle', 'cubic', 'lanczos2','lanczos3','lanczos4'};

kernels = {@box, @triangle, @cubic, @box, @triangle, @cubic, ...
           @lanczos2, @lanczos3, @lanczos4};

widths = [1.0 2.0 4.0 1.0 2.0 4.0 4.0 6.0 8.0];

for nn=1:size(names,2)
    if strcmp(names{nn},method)
        name = names{nn};
        kernel = kernels{nn};
        width = widths(nn);
        break;
    end
end

%---------------------------------------------------------------------

%=====================================================================
function f = cubic(x)
% See Keys, "Cubic Convolution Interpolation for Digital Image
% Processing," IEEE Transactions on Acoustics, Speech, and Signal
% Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.

absx = abs(x);
absx2 = absx.^2;
absx3 = absx.^3;

f = (1.5*absx3 - 2.5*absx2 + 1) .* (absx <= 1) + ...
                (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) .* ...
                ((1 < absx) & (absx <= 2));
%---------------------------------------------------------------------

%=====================================================================
function f = box(x)
f = (-0.5 <= x) & (x < 0.5);
%---------------------------------------------------------------------

%=====================================================================
function f = triangle(x)
f = (x+1) .* ((-1 <= x) & (x < 0)) + (1-x) .* ((0 <= x) & (x <= 1));
%---------------------------------------------------------------------

%=====================================================================
function f = lanczos2(x)
% See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990,
% pp. 156-157.

f = (sin(pi*x) .* sin(pi*x/2) + eps) ./ ((pi^2 * x.^2 / 2) + eps);
f = f .* (abs(x) < 2);
%---------------------------------------------------------------------

%=====================================================================
function f = lanczos3(x)
% See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990,
% pp. 157-158.

f = (sin(pi*x) .* sin(pi*x/3) + eps) ./ ((pi^2 * x.^2 / 3) + eps);
f = f .* (abs(x) < 3);
%---------------------------------------------------------------------

%=====================================================================
function f = lanczos4(x)
% See Graphics Gems, Andrew S. Glasser (ed), Morgan Kaufman, 1990,
% pp. 157-158.

f = (sin(pi*x) .* sin(pi*x/4) + eps) ./ ((pi^2 * x.^2 / 4) + eps);
f = f .* (abs(x) < 4);
%---------------------------------------------------------------------

%=====================================================================
function [weights, indices] = contributions(in_length, out_length, ...
                                            scale, kernel, ...
                                            kernel_width, antialiasing)


if (scale < 1) && (antialiasing)
    % Use a modified kernel to simultaneously interpolate and
    % antialias.
    h = @(x) scale * kernel(scale * x);
    kernel_width = kernel_width / scale;
else
    % No antialiasing; use unmodified kernel.
    h = kernel;
end

% Output-space coordinates.
x = (1:out_length)';

% Input-space coordinates. Calculate the inverse mapping such that 0.5
% in output space maps to 0.5 in input space, and 0.5+scale in output
% space maps to 1.5 in input space.
%u = x/scale + 0.5 * (1 - 1/scale);  % 0.5-->0.5,   1.5-->0.5+scale
u = x/scale + 1 - 1/scale;  % 1-->1,  2-->1+scale

% What is the left-most pixel that can be involved in the computation?
left = floor(u - kernel_width/2);

% What is the maximum number of pixels that can be involved in the
% computation?  Note: it's OK to use an extra pixel here; if the
% corresponding weights are all zero, it will be eliminated at the end
% of this function.
P = ceil(kernel_width) + 2;

% The indices of the input pixels involved in computing the k-th output
% pixel are in row k of the indices matrix.
indices = bsxfun(@plus, left, 0:P-1);

% The weights used to compute the k-th output pixel are in row k of the
% weights matrix.
weights = h(bsxfun(@minus, u, indices));

% Normalize the weights matrix so that each row sums to 1.
weights = bsxfun(@rdivide, weights, sum(weights, 2));

% Clamp out-of-range indices; has the effect of replicating end-points.
indices = min(max(1, indices), in_length);

% If a column in weights is all zero, get rid of it.
kill = find(~any(weights, 1));
if ~isempty(kill)
    weights(:,kill) = [];
    indices(:,kill) = [];
end

%---------------------------------------------------------------------

% %=====================================================================
% function out = resizeAlongDim(in, dim, weights, indices)
% % Resize along a specified dimension
% %
% % in           - input array to be resized
% % dim          - dimension along which to resize
% % weights      - weight matrix; row k is weights for k-th output pixel
% % indices      - indices matrix; row k is indices for k-th output pixel
% 
% out = imresizemex(in, weights', indices', dim);
% %---------------------------------------------------------------------

%=====================================================================
function out = imresizemex_dim1(in, weights, indices)

outh = size(weights,1);
outw = size(in,2);
out = zeros(outh, outw, size(in,3));

numW = size(weights,2);

for y=1:outh
    for w=1:numW
        out(y,:,:) = out(y,:,:) + weights(y,w)*in(indices(y,w),:,:);
    end
end
%---------------------------------------------------------------------

%=====================================================================
function out = imresizemex_dim2(in, weights, indices)

outh = size(in,1);
outw = size(weights,1);
out = zeros(outh, outw, size(in,3));

numW = size(weights,2);

for x=1:outw
    for w=1:numW
        out(:,x,:) = out(:,x,:) + weights(x,w)*in(:,indices(x,w),:);
    end
end
%---------------------------------------------------------------------
