clear all；
clc;  
ObjDir = 'D:\201818074001wang\MATLAB\bin\新建文件夹\resize\';%将被改变的图像地址，称为目标地址
OtpDir = 'D:\201818074001wang\MATLAB\bin\新建文件夹\resize\change\';%输出图像地址，称为输出地址
for i = 1:1:2%我的图像标号是00000001到00001340
    bgFile = [ObjDir,num2str(i,'%01d'),'.jpg'];%这句话读取目标地址里面的格式为png的图片
    %num2str是先把数字i转换成string然后补零直到八位
    %举个例子：i=13,num2str(i,'%08d)=00000013,类型是string
    bgFile = imread(bgFile);%把图片读成matlab认识的，类型为：图片
    img = imresize(bgFile,[416,416]);%调整大小到高360，长480
    filename=[num2str(i,'%01d'),'.jpg'];%输出的图片名称是00000001.png
    path=fullfile(OtpDir,filename);%输出的路径
    imwrite(img,path,'jpg');%以png格式输出出去
end

