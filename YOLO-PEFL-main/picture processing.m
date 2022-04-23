clear all;
clc;
I=imread('b.jpg');
image=im2double(I);
image1=edge(image, 'roberts');
subplot(231);
imshow(image1);
title('roberts算子的处理结果')
image2=imfilter(image,fspecial('prewitt'));
subplot(232);
imshow(image2);
title('Prewitt算子的处理结果')
image3=imfilter(image,fspecial('sobel'));
subplot(233);
imshow(image3);
title('sobel算子的处理结果')
image1=edge(image, 'canny');;
subplot(234);
imshow(image1);
title('canny算子的处理结果')
imfilter(image,fspecial('Laplacian'));
subplot(235);
imshow(image1);
title('Laplacian算子的处理结果')








