tempImages = zeros(39209,48,48,3);
tempLabels = zeros(39209,1);

for iter = 1:39209
    temp = cell2mat(images(:,iter));
    tempImages(iter,:,:,:) = imresize(temp,[48 48]);
    tempLabels(iter) = str2num(labels(iter,:));
    disp(iter);
end

index = randperm(39209);
images = tempImages(index,:,:,:);
images = uint8(images);
labels = tempLabels(index);
labels = uint8(labels);

save('GTSResized.mat','images','labels');