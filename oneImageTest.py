
from keras import backend as K
from keras import callbacks
import numpy as np

import glob
from PIL import Image
import cv2
from U_Net_Down_stage_output import superPixNet

test = glob.glob('/home/liu/libiao/oneImageTest')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_train_batch(batch_size,times,image_names,dir_path):
    batch_images_path = []
    selected_imgs_indexes=[batch_size*times+i for i in range(batch_size)]
    for index in selected_imgs_indexes:
        current_img =os.path.join(dir_path,image_names[index])
        batch_images_path.append(current_img)
    images ,path_list = get_img_and_labels(batch_images_path)
    return images ,path_list

def get_img_and_labels(batch_images_path):
    x_batch = []
    for image in batch_images_path:
        # image = image.split('/')[-1]
        name = image.split('/')[-1]
        im = Image.open(image)
        im = im.crop((0, 0, 320, 320))

        # (h, w) = im.size
        # print(h, w)
        # im = im.resize((640, 400))
        # im = im.crop((0, 0, 320, 320))
        # im.save("/home/libiao/PycharmProjects/EdgeNet/output/image/%s" % name)
        im = np.array(im, dtype=np.float32)
        im = im[..., ::-1]  # RGB 2 BGR
        R = im[..., 0].mean()
        G = im[..., 1].mean()
        B = im[..., 2].mean()
        im[..., 0] -= R
        im[..., 1] -= G
        im[..., 2] -= B
        # im[..., 0] -= 138.008
        # im[..., 1] -= 127.406
        # im[..., 2] -= 118.982
        x_batch.append(im)
    x_batch = np.array(x_batch, np.float32)
    # print(batch_images_path)
    return x_batch, batch_images_path
if __name__ == "__main__":
    #environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    # os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    model=superPixNet(input_shape=(320,320,3))
    model.load_weights('/home/liu/libiao/Simple_net_output/checkpoints/8.20_new_out/checkpoint.61-0.0452-0.9976-0.9341-0.8720-0.9004.hdf5',by_name=True)

    for dir_path in test:
        dir_test=os.listdir(dir_path)
        dir = dir_path.split('/')[-1]
        save_dir_path = os.path.join('/home/liu/libiao/oneImageTestSave', dir)
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)

        images, path_list = get_train_batch(batch_size=1, times=0,image_names=dir_test,dir_path=dir_path)
        print(path_list)
        prediction = model.predict(images,batch_size=1)
        print(len(prediction))

        for i in range(len(prediction[0])):
            print(i)
            name = path_list[i].split('/')[-1]
            mask = np.zeros_like(images[i][:, :, :1])
            mask += (prediction[-1][i])*255
            if '.jpg' in name:
                name=name.replace('.jpg','.png')
            final_save_path=os.path.join(save_dir_path,name)
            # final_save_path = os.path.join(save_dir_path, name)
            cv2.imwrite(final_save_path, mask)



