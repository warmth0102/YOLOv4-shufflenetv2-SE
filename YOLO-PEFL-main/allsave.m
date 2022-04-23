%%Hougn变换
clear all；
clc;
clear all；
clc;  
ObjDir = 'F:\green\光条图片\';%将被改变的图像地址，称为目标地址
OtpDir = 'D:\201818074001wang\MATLAB\bin\新建文件夹\光条\result1\';%输出图像地址，称为输出地址
for i = 1:1:80%我的图像标号是001到400
    bgFile = [ObjDir,num2str(i,'%03d'),'.jpg'];%这句话读取目标地址里面的格式为jpg的图片
    %num2str是先把数字i转换成string然后补零直到八位
    %举个例子：i=13,num2str(i,'%08d)=013,类型是string
    bgFile = imread(bgFile);%把图片读成matlab认识的，类型为：图片
    img = rgb2gray(bgFile);%调整大小到高416，长416
    filename=[num2str(i,'%03d'),'.jpg'];%输出的图片名称是001.jpg
    path=fullfile(OtpDir,filename);%输出的路径
    imwrite(img,path,'jpg');%以jpg格式输出出去
end


 








                             













