import nibabel as nib
import numpy as np
import os
import h5py
import random
import cv2
import copy
from matplotlib import pyplot as plt
from PIL import Image
import time

def nii_to_h5(path_nii,path_save,ratio=0.8):
    data = []
    label = []
    ori = []
    list_site = os.listdir(path_nii)
    list_data = []
    ori_min = 10000
    ori_max = 0
    for dir_num, dir_site in enumerate(list_site):
        if dir_site[-3:] == 'csv':
            continue

        list_patients = os.listdir(path_nii+'/'+dir_site)
        for dir_patients in list_patients:
            for t0n in ['/t01/', '/t02/']:
                try:
                    location = path_nii+'/' + dir_site + '/' + dir_patients + t0n
                    location_all = os.listdir(location)
                    for i in range(len(location_all)):
                        location_all[i] = location+location_all[i]
                    list_data.append(location_all)
                except:
                    continue

    random.shuffle(list_data)
    for num, data_dir in enumerate(list_data):
        for i, deface in enumerate(data_dir):
            if deface.find('deface') != -1:
                ori = nib.load(deface)
                ori = ori.get_fdata()
                ori = np.array(ori)
                ori = ori.transpose((2, 1, 0))
                if ori_max < ori.max():
                    ori_max = ori.max()
                if ori_min > ori.min():
                    ori_min = ori.min()
                del list_data[num][i]
                break

        label_merge = np.zeros_like(ori)
        for i, dir_data in enumerate(list_data[num]):
            img = nib.load(dir_data)
            img = np.array(img.get_fdata())
            img = img.transpose((2, 1, 0))
            label_merge = label_merge + img
        
        print(str(num)+'/'+str(len(list_data)),'max=',str(ori.max()),'min=',str(ori.min()))
        if num == 0 or num == int(ratio * len(list_data)):
            data = copy.deepcopy(ori)
            label = copy.deepcopy(label_merge)
        else:
            data = np.concatenate((data, ori), axis=0)
            label = np.concatenate((label, label_merge), axis=0)

        if num == int(ratio * len(list_data))-1:
            print('saving train set...')
            data = np.array(data, dtype=float)
            label = np.array(label, dtype=bool)
            #'''
            file = h5py.File(path_save + '/train_' + str(ratio), 'w')
            file.create_dataset('data', data=data)
            file.create_dataset('label', data=label)
            file.close()
            data = []
            label = []
            print('Finished!')

        elif num == len(list_data)-1:
            print('saving test set...')
            data = np.array(data, dtype=float)
            label = np.array(label, dtype=bool)
            file = h5py.File(path_save + '/test_' + str(ratio), 'w')
            file.create_dataset('data', data=data)
            file.create_dataset('label', data=label)
            file.close()
            print('Finished!')
    return ori_max, ori_min
            #'''

def data_adjust(max, min, h5_path, ratio=0.8):

    file = h5py.File(h5_path + '/test_' + str(ratio))
    data = file['data']
    label = file['label']
    data = data - min
    data = data / max
    data = data*255

    file_adjust = h5py.File(h5_path + '/detection/test', 'w')
    file_adjust.create_dataset('data', data=data)
    file_adjust.create_dataset('label', data=label)
    file.close()
    file_adjust.close()

    file = h5py.File(h5_path + '/train_' + str(ratio))
    data = file['data']
    label = file['label']
    data = data - min
    data = data / max
    data = data*255

    file_adjust = h5py.File(h5_path + '/detection/train', 'w')
    file_adjust.create_dataset('data', data=data)
    file_adjust.create_dataset('label', data=label)
    file.close()
    file_adjust.close()

def load_h5(path_h5, shuffle=False, size=None, test_programme=None, only=False):
    h5 = h5py.File(path_h5)
    data = h5['data'][:]
    label = h5['label'][:]

    if test_programme is not None:
        data = data[:test_programme]
        label = label[:test_programme]

    data_only = []
    label_only = []
    if only is True:
        for i in range(len(data)):
            if label[i].max() == 1:
                data_only.append(data[i])
                label_only.append(label[i])
        del data, label
        data = data_only
        label = label_only

    data = np.uint8(np.multiply(data, 2.55))
    label = np.uint8(np.multiply(label, 255))

    if size is not None:
        data_resize = []
        label_resize = []
        for i in range(len(data)):
            data_resize_single = Image.fromarray(data[i]).crop((10, 40, 190, 220))
            data_resize_single = data_resize_single.resize(size, Image.ANTIALIAS)
            data_resize_single = np.asarray(data_resize_single)

            label_resize_single = Image.fromarray(label[i]).crop((10, 40, 190, 220))
            label_resize_single = label_resize_single.resize(size, Image.ANTIALIAS)
            label_resize_single = np.asarray(label_resize_single)

            data_resize.append(data_resize_single)
            label_resize.append(label_resize_single)

        data = np.array(data_resize, dtype=float)
        label = np.array(label_resize, dtype=int)

    data = data - data.min()
    data = data / data.max()
    label = label - label.min()
    label = label / label.max()

    if shuffle is True:
        orders = []
        data_output = np.zeros_like(data)
        label_output = np.zeros_like(label)

        for i in range(len(data)):
            orders.append(i)
        random.shuffle(orders)
        for i, order in enumerate(orders):
            data_output[i] = data[order]
            label_output[i] = label[order]
    else:
        data_output = data
        label_output = label
    # for i in range(500):
    #     plt.subplot(1,2,1)
    #     plt.imshow(data_output[i],cmap='gray')
    #     plt.subplot(1,2,2)
    #     plt.imshow(label_output[i],cmap='gray')
    #     plt.pause(0.1)
    #     print(data_output[i].max(),data_output[i].min(),label_output[i].max(),label_output[i].min())

    return data_output, label_output

def data_toxn(data, z):
    data_xn = np.zeros((data.shape[0], data.shape[1], data.shape[2], z))
    for patient in range(int(len(data) / 189)):
        for i in range(189):
            for j in range(z):
                if i + j - z // 2 >= 0 and i + j - z // 2 < 189:
                    data_xn[patient * 189 + i, :, :, j] = data[patient * 189 + i + j - z // 2]
                    print(i, i + j - z // 2)
                else:
                    data_xn[patient * 189 + i, :, :, j] = np.zeros_like(data[0])
    return data_xn


if __name__ == "__main__":

    start = time.time()
    path_nii = '/media/root/72572323-0458-4f59-9fd6-33d6839809a01/brain_stroke/ATLAS_R1.1'
    path_save = '/media/root/72572323-0458-4f59-9fd6-33d6839809a01/brain_stroke/h5'
    ratio = 0.8
    img_size = [192, 192]
    ori_max, ori_min = nii_to_h5(path_nii, path_save, ratio=ratio)
    data_adjust(ori_max, ori_min, path_save)

    print('using :{}'.format(time.time()-start))

    print('loading training-data...')
    time_start = time.time()
    original, label = load_h5(path_save + 'train_' + str(ratio), size=(img_size[1], img_size[0]),
                              test_programme = None)
    file = h5py.File(path_save+'/train', 'w')
    original = data_toxn(original, 4)
    file.create_dataset('data', data=original)
    original = original.transpose((0, 3, 1, 2))
    original = np.expand_dims(original, axis=-1)
    file.create_dataset('data_lstm', data=original)
    del original

    label_change = data_toxn(label, 4)
    file.create_dataset('label_change', data=label_change)
    del label_change

    label = np.expand_dims(label, axis=-1)
    file.create_dataset('label', data=label)
    del label
    file.close()

    print('training_data done!, using:', str(time.time() - time_start) + 's\n\nloading validation-data...')
    time_start = time.time()
    original_val, label_val = load_h5(path_save + 'test_' + str(ratio), size=(img_size[1], img_size[0]))
    file = h5py.File(path_save+'/train', 'w')
    original_val = data_toxn(original_val, 4)
    file.create_dataset('data_val', data=original_val)

    original_val = original_val.transpose((0, 3, 1, 2))
    original_val = np.expand_dims(original_val, axis=-1)
    file.create_dataset('data_val_lstm', data=original_val)
    del original_val

    label_val_change = data_toxn(label_val, 4)
    file.create_dataset('label_val_change', data=label_val_change)
    del label_val_change

    label_val = np.expand_dims(label_val, axis=-1)
    file.create_dataset('label_val', data=label_val)
    del label_val
    file.close()

    print('validation_data done!, using:', str(time.time() - time_start) + 's\n\n')