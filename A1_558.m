% FUNDAMENTALS OF COMPUTER VISION           ASSIGNMENT 1
% COMP 558                                     FALL 2018 
% 
% ANNASTASIA STATHAKOS 

%% Q1. A)

Im = imread('A1.jpg');
Im = imresize(Im,0.44);
Im1 = rgb2gray(Im);

H1 = make2DGaussian(30,10);
blurred2 = imfilter(Im1,H1,'replicate');

figure(1);
imshow(blurred2)
title('Gaussian Filter');

%for i = 1:30
%     blurred2 = imfilter(blurred2,H1,'replicate');
%     imshow(blurred2);
% end

%% Q1. B)

C1 = myConv2(double(Im1),H1);
figure(2);
imshow(C1,[])
title('2 Dimensional Convolution');

%% Q2.

R_Chan = Im(:,:,1);
G_Chan = Im(:,:,2); % Green Channel
B_Chan = Im(:,:,3);

F = 50; % Filter Size
T = 30; % Threshold

% Compute Gradient Magnitude, vary sigma
G_Mag1 = gradientMag(G_Chan,F,0.5);
G_Mag2 = gradientMag(G_Chan,F,1);
G_Mag3 = gradientMag(G_Chan,F,2);

% Gradient Magnitude Threshold
G_Thresh = binaryImage(G_Chan, G_Mag1, T);
G_Thresh2 = binaryImage(G_Chan, G_Mag2, T);
G_Thresh3 = binaryImage(G_Chan, G_Mag3, T);

figure(3);
imshow(G_Thresh, [])
title('Intensity Gadients, sigma = 0.5');

figure(4);
imshow(G_Thresh2, [])
title('Intensity Gadients, sigma = 1');

figure(5);
imshow(G_Thresh3, [])
title('Intensity Gadients, sigma = 2');

%% Q3

RD = imread('A1Q32_551.jpg');
RD = rgb2gray(RD);

% Take the Laplacian of the Gaussian, vary sigma
myF = myLOGfilter(5,0.5);
filtered2 = conv2(double(RD),myF,'same');
myF2 = myLOGfilter(5,1);
filtered3 = conv2(double(RD),myF2,'same');
myF3 = myLOGfilter(5,2);
filtered4 = conv2(double(RD),myF3,'same');

% Zero Crossings into a binary image
binIm1 = zeroCross(filtered2);
binIm2 = zeroCross(filtered3);
binIm3 = zeroCross(filtered4);

figure(6);
imshow(binIm1)
title('Marr-Hildreth Zero Crossings, sigma = 0.5');

figure(7);
imshow(binIm2)
title('Marr-Hildreth Zero Crossings, sigma = 1');

figure(8);
imshow(binIm3)
title('Marr-Hildreth Zero Crossings, sigma = 2');

figure(9);
imshow(filtered3)
title('Laplacian of Gaussian');


%% Q4

% Take cropped sections of previous images
[m,n] = size(G_Chan);
[x,y] = size(RD);
crop_2 = G_Chan(floor(m/2):floor(m/2)+20,floor(n/2):floor(n/2)+20);
crop_3 = RD(floor(x/2)-20:floor(x/2),floor(x/2)-20:floor(x/2));
cropEdge_2 = G_Thresh(floor(m/2):floor(m/2)+20,floor(n/2):floor(n/2)+20);

figure(10);
imshow(crop_2) 

figure(11);
imshow(crop_3)

%% Functions for Q1. A)

% Gaussian Filter
%
% @params: double N (filter size), double sigma (std. dev)
% @return: kernel of sizeNxN
function g = make2DGaussian(N,sigma)
n = floor(N/2);
[x,y] = meshgrid(-n:n,-n:n);
g = gaussFilter(x,y,sigma);
s = sum(sum(g));
g = g./s;
end 

function val = gaussFilter(x,y,sigma)
exponent = (x.^2 + y.^2)/ (2*sigma^2);
a = (1/(sigma*sqrt(2*pi)));
val = a*(exp(-exponent));
end

%% Functions for Q1. B)

% Function to Convolve an image and a filter
%
% @params: unit8 2D matrix image (image), double 2D matrix filter (filter)
% @returns: double 2D matrix
function c = myConv2(image,filter)
F = rot90(filter,2); % Rotate filter by 180

[m,n]   = size(image);
[fm,fn] = size(F);
fm1 = floor(fm/2);
fn1 = floor(fn/2);

I  = padarray(image,[fm1,fn1]); % Pad image with zeros 
c  = zeros(m,n);
  for j = 1:m
      for k = 1:n
          If = I(j:j+fm-1,k:k+fn-1); % Look at filter sized sections of image
          c(j,k) = c(j,k) + sum(sum(double(If).*F));
      end
  end
end

%% Functions for Q2

% Function to Compute Gradient Magnitude
%
% @params: unit8 2D matrix C (image), double F (filter size),
%          double S (sigma)
% @return: double 2D matrix 
function gm = gradientMag(C, F, S)
% Apply Gaussian filter to image channel C
G_filter = fspecial('gaussian',F,S);
G_blurred = imfilter(C,G_filter,'replicate'); 

% Take Local Difference in y-direction
G_diffy = conv2(double(G_blurred),[-0.5;0;0.5],'same');

% Take Local Difference in x-direction
G_diffx = conv2(double(G_blurred),[-0.5,0,0.5],'same');

% Compute Gradient Orientation
G_Ori = atan2(G_diffy,G_diffx);

% Compute Gradient Magnitude
gm = sqrt((G_diffx.*G_diffx) + (G_diffy.*G_diffy));
end

% Function to Compute Binary Image
%
% @params: unit8 2D matrix C (image), double 2D matrix M (grad. mag.),
%          double T (threshold)
% @return: double 2D matrix
function t = binaryImage(C, M, T)
t = double(C);
t(M>T) = 255;
t(M<=T) = 0;
end

%% Functions for Q3

% Function to compute the Laplacian of the Gaussian
%
% @params: double N (filter size), double sigma (standard dev.)
% @return: double 2D matrix
function v = myLOGfilter(N,sigma)
n = round(linspace(-floor(N/2),floor(N/2),N));
[x,y] = meshgrid(n,n);

exponent = (x.^2 + y.^2)/(2*sigma^2);
gauss = exp(-exponent);

kern = gauss.*(x.^2+y.^2-(2*sigma^2)) / (2*pi*sigma^4*sum(gauss(:)));
v = kern - sum(kern(:))/(N^2);
end

function zx = zeroCross(f) 
[x,y] = size(f);

zx = zeros(x,y);
for i = 2:x-1
    for j = 2:y-1
        
        if ( f(i-1,j)*f(i+1,j)<0 || f(i,j-1)*f(i,j+1)<0)
            zx(i,j) = 255; 
        else 
            zx(i,j) = 0; 
        end
        
    end
end

end

