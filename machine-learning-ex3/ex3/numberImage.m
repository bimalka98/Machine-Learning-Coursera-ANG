load('ex3data1.mat')
indx = find(y == 9);
a = X(indx(150),:);
colormap(gray);
img = zeros(20,20);
for i = 1:20, img(i,:) = a(1,20*(i-1)+1:20*i); end
imagesc(img');
axis image off