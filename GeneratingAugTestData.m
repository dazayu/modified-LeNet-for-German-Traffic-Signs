load('TestOri.mat')
images1=images(1:2880,:,:,:);
load('TestIntChange.mat')
images2=images(2881:3600,:,:,:);
load('TestNois.mat')
images3=images(3601:4320,:,:,:);
load('TestRot.mat')
images4=images(4321:5040,:,:,:);
load('TestTrans.mat')
images5 = images(5041:5760,:,:,:);
load('TestBlur.mat')
images6 = images(5761:6480,:,:,:);
load('TestGammaContrast.mat')
images7 = images(6481:7209,:,:,:);

images = cat(1,images1,images2,images3,images4,images5,images6,images7);
images = uint8(images);
labels = labels(1:7209);
labels = uint8(labels);

save('AugTest.mat','images','labels')