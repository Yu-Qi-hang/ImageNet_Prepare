import numpy as np
import cv2
import os

#cifar10 官方给出的解压函数
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

loc_1 = '/data/home/yuqihang/dataset/MosaicKD/ImageNet_32x32/train/'
loc_2 = '/data/home/yuqihang/dataset/MosaicKD/ImageNet_32x32/val/'

#判断文件夹是否存在，不存在的话创建文件夹
if os.path.exists(loc_1) == False:
    os.mkdir(loc_1)
if os.path.exists(loc_2) == False:
    os.mkdir(loc_2)


def imagenet_img(file_dir):
    for i in range(1,11):
        data_name = file_dir + '/train1/train_data_batch_'+ str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')
        data_size = data_dict['data'].shape[0]

        for j in range(data_size):
            img = np.reshape(data_dict['data'][j],(3,32,32))
            img = np.transpose(img,(1,2,0))
            #通道顺序为RGB
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #要改成不同的形式的文件只需要将文件后缀修改即可
            class_folder = os.path.join(loc_1 , str(data_dict['labels'][j]))
            if os.path.exists(class_folder) == False:
                os.mkdir(class_folder)
                # print(class_folder + "is created")
            img_name = str((i)*1000000 + j) + '.jpg'
            img_name = os.path.join(class_folder , img_name)
            cv2.imwrite(img_name,img)

        print(data_name + ' is done')

    val_data_name = file_dir + '/val1/val_data'
    print(val_data_name + ' is processing')
    val_dict = unpickle(val_data_name)
    val_size = val_dict['data'].shape[0]

    for m in range(val_size):
        img = np.reshape(val_dict['data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        # 通道顺序为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 要改成不同的形式的文件只需要将文件后缀修改即可
        class_folder = os.path.join(loc_2 , str(val_dict['labels'][m]))
        if os.path.exists(class_folder) == False:
            os.mkdir(class_folder)
            print(class_folder + "is created")
        img_name = str(1000000 + m) + '.jpg'
        img_name = os.path.join(class_folder , img_name)
        cv2.imwrite(img_name, img)
    print(val_data_name + ' is done')
    print('Finish transforming to image')

if __name__ == '__main__':

    file_dir = '/data/home/yuqihang/dataset/MosaicKD/ImageNet_32x32'
    imagenet_img(file_dir)