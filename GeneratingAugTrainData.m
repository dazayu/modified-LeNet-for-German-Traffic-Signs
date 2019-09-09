% load('TrainOri.mat')
% images1 = zeros(12800,32,32,3);
% labels1 = zeros(12800,1);
% 
% for i = 1:10
%     images1((i-1)*1280+1:(i*1280),:,:,:) = images((i-1)*3200+1:(i-1)*3200+1280,:,:,:);
%     labels1((i-1)*1280+1:(i*1280)) = labels((i-1)*3200+1:(i-1)*3200+1280);
% end
% 
% load('TrainIntChange.mat')
% images2 = zeros(3200,32,32,3);
% labels2 = zeros(3200,1);
% 
% for i = 1:10
%     images2((i-1)*320+1:i*320,:,:,:) = images((i-1)*3200+1281:(i-1)*3200+1600,:,:,:);
%     labels2((i-1)*320+1:i*320) = labels((i-1)*3200+1281:(i-1)*3200+1600);
% end
% 
% 
% load('TrainNois.mat')
% images3 = zeros(3200,32,32,3);
% labels3 = zeros(3200,1);
% 
% for i = 1:10
%     images3((i-1)*320+1:i*320,:,:,:) = images((i-1)*3200+1601:(i-1)*3200+1920,:,:,:);
%     labels3((i-1)*320+1:i*320) = labels((i-1)*3200+1601:(i-1)*3200+1920);
% end
% % images3=uint8(images3);
% 
% load('TrainRot.mat')
% images4 = zeros(3200,32,32,3);
% labels4 = zeros(3200,1);
% 
% for i = 1:10
%     images4((i-1)*320+1:i*320,:,:,:) = images((i-1)*3200+1921:(i-1)*3200+2240,:,:,:);
%     labels4((i-1)*320+1:i*320) = labels((i-1)*3200+1921:(i-1)*3200+2240);
% end
% 
% load('TrainTrans.mat')
% images5 = zeros(3200,32,32,3);
% labels5 = zeros(3200,1);
% 
% for i = 1:10
%     images5((i-1)*320+1:i*320,:,:,:) = images((i-1)*3200+2241:(i-1)*3200+2560,:,:,:);
%     labels5((i-1)*320+1:i*320) = labels((i-1)*3200+2241:(i-1)*3200+2560);
% end
% 
% load('TrainGammaContrast.mat')
% images6 = zeros(3200,32,32,3);
% labels6 = zeros(3200,1);
% 
% for i = 1:10
%     images6((i-1)*320+1:i*320,:,:,:) = images((i-1)*3200+2561:(i-1)*3200+2880,:,:,:);
%     labels6((i-1)*320+1:i*320) = labels((i-1)*3200+2561:(i-1)*3200+2880);
% end
% 
% load('TrainBlur.mat')
% 
% images7 = zeros(3200,32,32,3);
% labels7 = zeros(3200,1);
% 
% for i = 1:10
%     images7((i-1)*320+1:i*320,:,:,:) = images((i-1)*3200+2881:i*3200,:,:,:);
%     labels7((i-1)*320+1:i*320) = labels((i-1)*3200+2881:(i-1)*3200+3200);
% end
% 
% 
% 
% images = cat(1,images1,images2,images3,images4,images5,images6,images7);
% labels = cat(1,labels1,labels2,labels3,labels4,labels5,labels6,labels7);
% images = uint8(images);
% labels = uint8(labels);
% 
% save('AugTrain.mat','images','labels')
% 

load('TrainOri.mat')
images1=images(1:12800,:,:,:);
load('TrainIntChange.mat')
images2=images(12801:16000,:,:,:);
load('TrainNois.mat')
images3=images(16001:19200,:,:,:);
load('TrainRot.mat')
images4=images(19201:22400,:,:,:);
load('TrainTrans.mat')
images5 = images(22401:25600,:,:,:);
load('TrainBlur.mat')
images6 = images(25601:28800,:,:,:);
load('TrainGammaContrast.mat')
images7 = images(28801:32000,:,:,:);

images = cat(1,images1,images2,images3,images4,images5,images6,images7);
images = uint8(images);
labels = labels(1:32000);
labels = uint8(labels);

save('AugTrain.mat','images','labels')