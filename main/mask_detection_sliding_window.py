import os.path

import cv2
import numpy as np
from classification.ClassficationNetwork import CNN
import torch

import torchvision




class MaskDetector:
    pyramid_count = 5
    pyramid_ratio = 3/4
    labels = {
        0:'background',
        1: 'incorrect',
        2: 'mask',
        3: 'nomask',
    }
    colors = {
        0: (255,255,255),
        1: (0,0,255),
        2: (0,255,0),
        3: (255,0,0),
    }

    windwos = [

            # {'size':[24, 24], 'stride': 4},
               {'size': [30, 30], 'stride': 5},
               {'size': [42, 42], 'stride': 7},
               {'size': [60, 60], 'stride': 10},
               {'size': [100, 120], 'stride': 13},
        ]
    input_size = (24, 24)
    def __init__(self, model_path):
        self.model_path = model_path
        model = CNN()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self.model = model

    @staticmethod
    def get_images_by_window(image, window):
        res = []
        rois = []
        h,w,_ = image.shape
        window_h = window['size'][0]
        window_w = window['size'][1]
        for i in range(0, h - window_h, window['stride']):
            for j in range(0, w-window_w, window['stride']):
                roi = (j,i,j+window_w,i+window_h )
                sub_img = img[roi[1]:roi[3],roi[0]:roi[2]]
                sub_img = cv2.resize(sub_img, MaskDetector.input_size)
                # cv2.imshow('image', sub_img)
                # cv2.waitKey(0)
                res.append(MaskDetector.process_image(sub_img))
                rois.append(roi)
        return res,rois

    @staticmethod
    def process_image(img):
        img = np.array(img) / 255
        img = img.transpose((2, 0, 1))
        return img

    @staticmethod
    def get_all_windows(image):
        all_images = []
        all_rois = []
        for window in MaskDetector.windwos:
            imgs,rois = MaskDetector.get_images_by_window(image,window)
            all_images.extend(imgs)
            all_rois.extend(rois)
        return all_images,all_rois

    def evaluate(self,imgs, rois):
        arr = np.array(imgs)
        input = torch.tensor(imgs, dtype=torch.float)
        rois = torch.tensor(rois, dtype=torch.float)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        rois = rois.to(device)
        outputs = self.model(input)
        print(outputs)
        print(torch.exp(outputs))
        scores, preds = torch.max(outputs, 1)
        scores = torch.exp(scores)
        print(preds)
        indices = (preds != 0 ).nonzero(as_tuple=False)
        indices = indices.squeeze()
        print(indices)
        filtered_socors = torch.index_select(scores, 0, indices)
        filtered_rois = torch.index_select(rois, 0, indices)
        filtered_preds= torch.index_select(preds, 0, indices)
        print(filtered_socors)
        print(filtered_rois)
        print(filtered_preds)
        # preds = preds.cpu().data.numpy()
        # filter_arr = print(preds == 2)
        # for img in arr[filter_arr]:
        #     cv2.imshow('image', img)
        #     cv2.waitKey(0)
        res_indices = torchvision.ops.batched_nms(filtered_rois, filtered_socors, filtered_preds, 0.2)
        print(res_indices)
        # final_socors = torch.index_select(filtered_socors, 0, res_indices)
        # final_rois = torch.index_select(filtered_rois, 0, res_indices)
        # final_preds= torch.index_select(filtered_preds, 0, res_indices)
        return filtered_socors,filtered_rois,filtered_preds,res_indices
if __name__ == '__main__':
    data_dir = "/home/duhuaiyu/Downloads/facemaskdata/images"
    file_path = os.path.join(data_dir, 'maksssksksss0.png')
    img = cv2.imread(file_path)
    imgs,rois = MaskDetector.get_all_windows(img)

    color = (0, 255, 0)
    print(len(imgs))
    imgHeight, imgWidth, _ = img.shape
    thick = int((imgHeight + imgWidth) // 900)
    maskDetector = MaskDetector('../classification/checkpoint24_acc90.pth')
    final_socors,final_rois,final_preds,res_indices = maskDetector.evaluate(imgs,rois)
    print(final_rois)
    newImage = img.copy()
    for i in res_indices:
        score = final_socors[i].item()
        roi = final_rois[i].cpu().data.numpy().astype(int)
        print(roi)
        pred = final_preds[i].item()
        cv2.rectangle(newImage, (roi[0],roi[1]), (roi[2], roi[3]), MaskDetector.colors[pred], thick)
        cv2.putText(newImage, MaskDetector.labels[pred]+":"+ '{:.2f}'.format(score),
                    (roi[0], roi[1] - 12), 0, 1e-3 * imgHeight, MaskDetector.colors[pred], thick // 3)
    cv2.namedWindow('finalImg', 0)

    cv2.resizeWindow("enhanced", 1024, 960);
    cv2.imshow('image', newImage)
    cv2.waitKey(0)