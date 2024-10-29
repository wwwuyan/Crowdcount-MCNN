% 高斯滤波器大小
filter_size = [15, 8];

% 高斯函数标准差
sigma = 4;

% 创建高斯滤波器
H = fspecial('Gaussian', filter_size, sigma);

% 显示高斯滤波器
figure;
surf(H);
title('Gaussian Filter');

% 显示每个权重值
figure;
imagesc(H);
colormap('hot');
colorbar;
title('Gaussian Filter Weights');

% 输出滤波器权重
disp('Gaussian Filter Weights:');
disp(H);

