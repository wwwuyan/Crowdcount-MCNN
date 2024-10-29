import os

import cv2
import numpy as np

from src import network, utils
from src.crowd_count import CrowdCounter

#img_path =  './data/original/shanghaitech/part_B_final/test_data/images/IMG_2.jpg'
img_path='./my_3.png'
model_path = './final_models/mcnn_shtechB_110.h5'

trained_model = os.path.join(model_path)
net = CrowdCounter()
network.load_net(trained_model, net)
net.cuda()
net.eval()

img = cv2.imread(img_path,0)
img = img.astype(np.float32, copy=False)
ht = img.shape[0]
wd = img.shape[1]
ht_1 = (ht//4)*4
wd_1 = (wd//4)*4
img = cv2.resize(img,(wd_1,ht_1))
im_data = img.reshape((1,1,img.shape[0],img.shape[1]))

density_map = net(im_data, )
density_map = density_map.data.cpu().numpy()
et_count = np.sum(density_map)
print('预测人群数量: ', et_count)

origin_img = im_data[0][0]
density_map = 255*density_map/np.max(density_map)
density_map= density_map[0][0]

# if (density_map.shape[1] != origin_img.shape[1]):
#     origin_img = cv2.resize(origin_img, (density_map.shape[1], density_map.shape[0]))
#     #density_map = cv2.resize(density_map, (origin_img.shape[1], origin_img.shape[0]))

#result_img = np.hstack((origin_img, density_map))
#result_img = result_img.astype(np.uint8, copy=False)

result_img = density_map.astype(np.uint8, copy=False)


scale_factor=10
resized_img = cv2.resize(result_img, None, fx=scale_factor, fy=scale_factor)
cv2.imshow('density_map', result_img)
cv2.waitKey(0)