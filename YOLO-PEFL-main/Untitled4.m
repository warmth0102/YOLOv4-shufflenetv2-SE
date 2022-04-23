%灰度化算法
%matlab实现

clear;
clc;
img = imread('F:\green\光条图片\其他\6.png');
%提取图像的信息：行、列以及通道数量
[m,n,color] = size(img);
%创建一个模板
gray_img2 = zeros(m,n);
for x = 1:m
    for y = 1:n
        %平均值法
        gray_img2(x,y) = (double(img(x,y,1))+double(img(x,y,2))+double(img(x,y,3)))/3;
    end
end
figure();
imshow(uint8(gray_img2));
picture=uint8(gray_img2);
x=size(picture,1);
y=size(picture,2);
texture=uint8(zeros(x,y));

for i=2:1:x-1
    for j=2:1:y-1
        neighbor=uint8(zeros(1,8));
       
        neighbor(1,1)=picture(i-1,j);
        neighbor(1,2)=picture(i-1,j+1);
        neighbor(1,3)=picture(i,j+1);
        neighbor(1,4)=picture(i+1,j+1);
        neighbor(1,5)=picture(i+1,j);
        neighbor(1,6)=picture(i+1,j-1);
        neighbor(1,7)=picture(i,j-1);
         neighbor(1,8)=picture(i-1,j-1);
        center=picture(i,j);
        temp=uint8(0);
        for k=1:1:8
             temp =temp+ (neighbor(1,k) >= center)* 2^(k-1);
        end
        texture(i,j)=temp;
       
    end
end
imshow(texture);


