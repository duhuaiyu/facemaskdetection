from bs4 import BeautifulSoup
import os
import shutil
import cv2
import random

source_dir = "D:/2021-2022WinterSemester/ISP/project/facemaskdetection/data/mask_data"
dest_dir = "D:/2021-2022WinterSemester/ISP/project/facemaskdetection/data/classification_data"
resize_dim=(24,24)
catagory = ["train","validation","test"]
dirs = ["nomask","mask","incorrect","background"]
random_background = 5
background_size = [40, 60, 80, 100, 120]
def remove_folder_contents(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def generate_box(obj):

    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # # Labels (In my case, I only one class: target class or background)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        # # Tensorise img_id
        # img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        return target
def create_rotation (img,num, dir,name):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    for i in range(num):
        degree = random.randint(-10,10)
        M = cv2.getRotationMatrix2D(center, degree, scale  =1.2)
        rotated = cv2.warpAffine(img, M, (w, h))
        save_path = os.path.join(dest_dir,dir ,name+"_"+str(i)+".png")
        cv2.imwrite(save_path,rotated)

if __name__ == '__main__':
    image_path = os.path.join(source_dir,"images")
    annotation_path = os.path.join(source_dir,"annotations")
    imgs = list(sorted(os.listdir(image_path)))
    labels = list(sorted(os.listdir(annotation_path)))

    # clear output folder
    for cate in catagory:
        for dir in dirs:
            remove_folder_contents(os.path.join(dest_dir,cate,dir))

    for img_idx, img in enumerate(imgs):
        head, tail = os.path.split(img)
        file_name = tail[:-4]
        label_path = os.path.join(annotation_path, file_name +".xml")
        img_path = os.path.join(image_path, file_name +".png")
        labels = generate_target(file_name,label_path)
        org_img = cv2.imread(img_path)
        print(labels)
        if img_idx < 600:
            dest_dir_sub = os.path.join(dest_dir,"train")
        elif img_idx < 750:
            dest_dir_sub = os.path.join(dest_dir,"validation")
        else:
            dest_dir_sub = os.path.join(dest_dir,"test")
        for idx, box in enumerate(labels["boxes"]):
            new_img = org_img[box[1]:box[3],box[0]:box[2]]
            resized = cv2.resize(new_img, resize_dim, interpolation = cv2.INTER_CUBIC)
            # cv2.imshow('image',resized)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            save_path = os.path.join(dest_dir_sub,dirs[labels["labels"][idx]])
            save_file = os.path.join(save_path,file_name+"_"+str(idx)+".png")
            cv2.imwrite(save_file,resized)
            if labels["labels"][idx] == 0:
                flip = cv2.flip(resized,1)
                save_file = os.path.join(save_path,file_name+"_flip_"+str(idx)+".png")
                cv2.imwrite(save_file,flip)
                create_rotation(resized,3,save_path,file_name+"_rotation");
                create_rotation(flip,3,save_path,file_name+"_flip_rotation");
            elif labels["labels"][idx] == 2:
                flip = cv2.flip(resized,1)
                save_file = os.path.join(save_path,file_name+"_flip_"+str(idx)+".png")
                cv2.imwrite(save_file,flip)
                create_rotation(resized,10,save_path,file_name+"_rotation");
                create_rotation(flip,10,save_path,file_name+"_flip_rotation");

        image_shape = org_img.shape
        for i in range(random_background):
            x = random.randint(0,image_shape[0]-background_size[i])
            y = random.randint(0,image_shape[1]-background_size[i])
            new_img = org_img[x:x+background_size[i],y:y+background_size[i]]
            resized = cv2.resize(new_img, resize_dim, interpolation = cv2.INTER_CUBIC)
            save_path = os.path.join(dest_dir_sub,dirs[3],file_name+"_"+str(i)+".png")
            cv2.imwrite(save_path,resized)
