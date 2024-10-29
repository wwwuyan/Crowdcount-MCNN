import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from src import network
from src.crowd_count import CrowdCounter


class CounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("密度图转换及人群计数系统")
        self.root.geometry("570x370")  # 设置UI界面大小为800x400

        # 创建框架用于展示原图和灰度图
        self.original_frame = tk.Frame(root, width=256, height=192, borderwidth=2, relief="solid")
        self.original_frame.grid(row=0, column=0, padx=10, pady=10)

        self.density_frame = tk.Frame(root, width=256, height=192, borderwidth=2, relief="solid")
        self.density_frame.grid(row=0, column=1, padx=10, pady=10)

        # 上传图片按钮
        self.upload_button = tk.Button(root, text="上传图片", command=self.upload_image)
        self.upload_button.grid(row=1, column=0, columnspan=2, pady=10)

        # 密度图转换按钮
        self.convert_button = tk.Button(root, text="转换密度图", command=self.convert_to_density)
        self.convert_button.grid(row=2, column=0, columnspan=2, pady=10)

        # 显示人数
        self.size_label = tk.Label(root, text="人群计数: N/A")
        self.size_label.grid(row=3, column=0, columnspan=2, pady=10)

        # 初始化变量
        self.original_image = None
        self.density_image = None
        self.img_path=None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_path:
            # 加载原图并显示
            self.img_path=file_path
            self.original_image = Image.open(file_path)
            self.original_image = self.original_image.resize((256,192))
            #self.original_image.thumbnail((256, 192))
            self.display_image(self.original_frame, self.original_image)

    def convert_to_density(self):
        if self.original_image:
            # 转换为灰度图并显示
            #model_path = './final_models/mcnn_shtechB_110.h5'
            model_path = './final_models/mcnn_shtechA_660.h5'

            trained_model = os.path.join(model_path)
            net = CrowdCounter()
            network.load_net(trained_model, net)
            net.cuda()
            net.eval()

            # 读取图像路径，img类型是numpy数组
            img = cv2.imread(self.img_path,0)
            img = img.astype(np.float32, copy=False)
            ht = img.shape[0]
            wd = img.shape[1]
            ht_1 = (ht // 4) * 4
            wd_1 = (wd // 4) * 4
            img = cv2.resize(img, (wd_1, ht_1))
            im_data = img.reshape((1, 1, img.shape[0], img.shape[1]))

            density_map = net(im_data, )
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            print('预测人群数量: ', et_count)

            density_map = 255 * density_map / np.max(density_map)
            density_map = density_map[0][0]

            self.density_image = density_map.astype(np.uint8, copy=False)
            self.density_image = Image.fromarray(cv2.cvtColor(self.density_image, cv2.COLOR_BGR2RGB))
            self.density_image = self.density_image.resize((256,192))

            size_text = f"人群计数: {et_count//1}"
            self.size_label.config(text=size_text)
            self.display_image(self.density_frame,self.density_image)
    def display_image(self, frame, image):
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(frame, image=photo)
        label.image = photo  # 保持对图像对象的引用
        label.grid(row=0, column=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = CounterApp(root)
    root.mainloop()
