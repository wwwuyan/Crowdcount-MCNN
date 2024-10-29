function im_density = get_density_map_gaussian(im,points)

% 初始化密度图
im_density = zeros(size(im)); 
[h,w] = size(im_density);

% 如果没有给定点，直接返回空的密度图
if(length(points)==0)
    return;
end

% 如果只有一个点，将其位置标记在密度图上并返回
if(length(points(:,1))==1)
    x1 = max(1,min(w,round(points(1,1))));
    y1 = max(1,min(h,round(points(1,2))));
    im_density(y1,x1) = 255;
    return;
end
for j = 1:length(points) 	
    f_sz = 15;
    sigma = 4.0;
    % 定义高斯滤波器大小和标准差
    % 并通过fspecial函数创建一个二维高斯滤波器
    H = fspecial('Gaussian',[f_sz, f_sz],sigma);
    % 获取点的坐标
    x = min(w,max(1,abs(int32(floor(points(j,1)))))); 
    y = min(h,max(1,abs(int32(floor(points(j,2))))));
    % 如果坐标超出范围，就跳过该点
    if(x > w || y > h)
        continue;
    end
    % 计算高斯滤波器应用的区域                                               
    x1 = x - int32(floor(f_sz/2)); y1 = y - int32(floor(f_sz/2));
    x2 = x + int32(floor(f_sz/2)); y2 = y + int32(floor(f_sz/2));
    dfx1 = 0; dfy1 = 0; dfx2 = 0; dfy2 = 0;
    change_H = false;
    % 处理边界情况，调整滤波器应用的区域
    if(x1 < 1)
        dfx1 = abs(x1)+1;
        x1 = 1;
        change_H = true;
    end
    if(y1 < 1)
        dfy1 = abs(y1)+1;
        y1 = 1;
        change_H = true;
    end
    if(x2 > w)
        dfx2 = x2 - w;
        x2 = w;
        change_H = true;
    end
    if(y2 > h)
        dfy2 = y2 - h;
        y2 = h;
        change_H = true;
    end
    % 计算调整后的高斯滤波器范围，并重新构建高斯滤波器
    x1h = 1+dfx1; y1h = 1+dfy1; x2h = f_sz - dfx2; y2h = f_sz - dfy2;
    if (change_H == true)
        H =  fspecial('Gaussian',[double(y2h-y1h+1), double(x2h-x1h+1)],sigma);
    end
    % 在密度图上应用高斯滤波器，进行卷积操作，由于只有在点的坐标处有1，所以可以直接相加
    im_density(y1:y2,x1:x2) = im_density(y1:y2,x1:x2) +  H;
     
end

end