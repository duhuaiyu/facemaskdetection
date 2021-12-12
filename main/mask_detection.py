import os.path

import cv2


class MaskDetector:
    pyramid_count = 5
    pyramid_ratio = 3/4
    def __init__(self, model_path):
        self.model_path = model_path

    @staticmethod
    def get_pyramid_images(image):
        imgs = []
        imgs.append({'image': image, 'shrink_size': 1})
        temp_img = image
        for i in range(MaskDetector.pyramid_count):
            height, width, channels = temp_img.shape
            lower_img = cv2.pyrDown(temp_img)
            imgs.append({'image': lower_img, 'shrink_size': (1.0 / (i + 1))})
            temp_img = lower_img
        return imgs


if __name__ == '__main__':
    data_dir = "/home/duhuaiyu/Downloads/facemaskdata/images"
    file_path = os.path.join(data_dir, 'maksssksksss0.png')
    img = cv2.imread(file_path)
    imgs = MaskDetector.get_pyramid_images(img)
    for img in imgs:
        cv2.imshow('image', img['image'])
        cv2.waitKey(0)