import numpy as np
import os
from model import *
from Statistics import *

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = .25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma + 1e6) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def EML(y_true, y_pred):
    gamma = 2
    smooth = 1.
    alpha = .25
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    dice_loss = (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)

    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    focal_loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1), axis=-1) - K.mean(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=-1)
    return focal_loss - K.log(dice_loss)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    path_h5_save = '/media/siat/770f90d2-a17b-4301-aa9f-ca2616ede7ec/brain_stroke/brain_stroke/h5/'
    output_path = '/media/siat/770f90d2-a17b-4301-aa9f-ca2616ede7ec/brain_stroke/brain_stroke/model_6.17_repeat/'
    dataset_name = '0.8'
    # load_weight = '/media/root/72572323-0458-4f59-9fd6-33d6839809a01/brain_stroke/model/D_Dice2.hdf5'
    load_weight = ''
    mode = 'train'  # use 'train' or 'detect'
    img_size = [192, 192]
    batch_size = 36
    lr = 1e-5
    gpu_used = 2
    test_programme = None  # 378#8316
    only = False

    model = D_Unet()
    h5_name = 'DUnet'
    output_path += h5_name+'/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.summary()
    model = multi_gpu_model(model, gpus=gpu_used)
    model.compile(optimizer=Adam(lr=lr), loss= EML, metrics=[dice_coef])

    if load_weight != '':
        print('loading：', load_weight)
        model.load_weights(load_weight, by_name=True)
    else:
        print('no loading weight!')

    if mode == 'train':
        h5 = h5py.File('/home/siat/data/train')
        original = h5['data']
        label = h5['label']
        # label = h5['label_change']
        h5 = h5py.File('/home/siat/data/test')
        original_val = h5['data_val']
        label_val = h5['label_val']
        # label_val = h5['label_val_change']


        num_train_steps = math.floor(len(original) / batch_size)
        num_val_steps = math.floor(len(original_val) / batch_size)

        print('training data:' + str(len(original)) + '  validation data:' + str(len(original_val)))

        # print('using:', str(time.time() - time_start) + 's\n')
        time_start = time.time()
        data_gen_args = dict(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, rotation_range=20,
                             horizontal_flip=True, featurewise_center=True, featurewise_std_normalization=True)
        data_gen_args_validation = dict(featurewise_center=True, featurewise_std_normalization=True)

        #data_gen_args = dict()
        #data_gen_args_validation = dict()

        train_datagen = ImageDataGenerator(**data_gen_args)
        train_datagen_label = ImageDataGenerator(**data_gen_args)
        validation_datagen = ImageDataGenerator(**data_gen_args_validation)
        validation_datagen_label = ImageDataGenerator(**data_gen_args_validation)

        image_generator = train_datagen.flow(original, batch_size=batch_size, seed=1)
        mask_generator = train_datagen_label.flow(label, batch_size=batch_size, seed=1)
        image_generator_val = validation_datagen.flow(original_val, batch_size=batch_size, seed=1)
        mask_generator_val = validation_datagen_label.flow(label_val, batch_size=batch_size, seed=1)

        train_generator = zip(image_generator, mask_generator)
        validation_generator = zip(image_generator_val, mask_generator_val)

        early_stopping = EarlyStopping(patience=150)  # 当训练patience个epoch,测试集的loss还没有下降时停止训练
        checkpointer = ModelCheckpoint(output_path + h5_name + '-{epoch:02d}-{val_dice_coef:.2f}.hdf5', verbose=2, save_best_only=False, period=10)
        History=model.fit_generator(train_generator, epochs=150, steps_per_epoch=num_train_steps,
                            shuffle=True, callbacks=[checkpointer], validation_data=validation_generator, validation_steps=num_val_steps, verbose=2)




    elif mode == 'detect':
        print('loading testing-data...')
        h5 = h5py.File('/media/root/72572323-0458-4f59-9fd6-33d6839809a01/brain_stroke/h5/x4/test')
        original = h5['data_val']
        label = h5['label_val']
        # label_val_change = h5['label_val_change']
        print('load data done!')

        model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[TP, TN, FP, FN, dice_coef])

        dice_list = []
        recall_list = []
        precision_list = []

        tp = 0
        fp = 0
        fn = 0
        for i in range(len(label) // 189):
            start = i * 189
            result = model.evaluate(original[start:start + 189], label[start:start + 189], verbose=2)
            dice_per = (2 * result[1] / (2 * result[1] + result[3] + result[4]))
            recall_per = result[1] / (result[1] + result[4])
            precision_per = result[1] / (float(result[1]) + float(result[3]))
            dice_list.append(dice_per)
            recall_list.append(recall_per)
            if np.isnan(precision_per):
                precision_per = 0
            precision_list.append(precision_per)
            tp = tp + result[1]
            fp = fp + result[3]
            fn = fn + result[4]

        dice_all = 2 * tp / (2 * tp + fp + fn)
        dice_list = sorted(dice_list)
        dice_mean = np.mean(dice_list)
        dice_std = np.std(dice_list)
        print('dice_media: ' + str(
            (dice_list[int(dice_list.__len__() / 2)] + dice_list[int(dice_list.__len__() / 2 - 1)]) / 2) +
              ' dice_all: ' + str(dice_all) + '\n'
                                              'dice_mean: ' + str(np.mean(dice_list)) + ' dice_std:' + str(
            np.std(dice_list, ddof=1)) + '\n'
                                         'recall_mean: ' + str(np.mean(recall_list)) + ' recall_std:' + str(
            np.std(recall_list, ddof=1)) + '\n'
                                           'precision_mean: ' + str(np.mean(precision_list)) + ' precision_std:' + str(
            np.std(precision_list, ddof=1)) + '\n')

        #np.save('/root/桌面/paper材料/box/' + h5_name, dice_list)
        # plt.boxplot(dice_list)
        # plt.show()

        #'''
        tim = time.time()
        predict = model.predict(original, verbose=1, batch_size=batch_size)
        print('predict patients: '+str(len(predict)/189)+'   using: '+str(time.time()-tim)+'s')
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0

        for i in range(len(predict)):
            if predict[i, :, :, 0].max() == 1 or label[i, :, :, 0].max() == 1:
                plt.subplot(1, 3, 1)
                plt.imshow(original[i, :, :, 0], cmap='gray')
                plt.subplot(1, 3, 2)
                plt.imshow(label[i, :, :, 0], cmap='gray')
                plt.subplot(1, 3, 3)
                plt.imshow(predict[i, :, :, 0], cmap='gray')
                plt.title(i)
                plt.pause(0.1)

                if i % 20==0:
                    plt.close()
                    plt.close()
                    plt.close()
