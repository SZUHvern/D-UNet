from keras import backend as K
import numpy as np

def TP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    return true_positives


def FP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f01 = K.round(K.clip(y_pred_f, 0, 1))
    tp_f01 = K.round(K.clip(y_true_f * y_pred_f, 0, 1))
    false_positives = K.sum(K.round(K.clip(y_pred_f01 - tp_f01, 0, 1)))
    return false_positives


def TN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f01 = K.round(K.clip(y_pred_f, 0, 1))
    all_one = K.ones_like(y_pred_f01)
    y_pred_f_1 = -1 * (y_pred_f01 - all_one)
    y_true_f_1 = -1 * (y_true_f - all_one)
    true_negatives = K.sum(K.round(K.clip(y_true_f_1 + y_pred_f_1, 0, 1)))
    return true_negatives


def FN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    tp_f01 = K.round(K.clip(y_true_f * y_pred_f, 0, 1))
    false_negatives = K.sum(K.round(K.clip(y_true_f - tp_f01, 0, 1)))
    return false_negatives


def recall(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fn = FN(y_true, y_pred)
    return tp / (tp + fn)


def precision(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fp = FP(y_true, y_pred)
    return tp / (tp + fp)


def patch_whole_dice(truth, predict):
    dice = []
    count_dice = 0
    for i in range(len(truth)):
        true_positive = truth[i] > 0
        predict_positive = predict[i] > 0
        match = np.equal(true_positive, predict_positive)
        match_count = np.count_nonzero(match)

        P1 = np.count_nonzero(predict[i])
        T1 = np.count_nonzero(truth[i])

        full_back = np.zeros(truth[i].shape)
        non_back = np.invert(np.equal(truth[i], full_back))
        TP = np.logical_and(match, non_back)
        TP_count = np.count_nonzero(TP)
        # print("m:", match_count, " P:", P1, " T:", T1, " TP:", TP_count)

        if (P1 + T1) == 0:
            dice.append(0)
        else:
            dice.append(2 * TP_count / (P1 + T1))
        if P1 != 0 or T1 != 0:
            count_dice += 1
    if count_dice == 0:
        count_dice = 1e6
    return dice  # , count_dice
    # return dice

def patch_whole_dice2(truth, predict):
    y_true_f = np.reshape(truth, (1, -1))
    y_pred_f = np.reshape(predict, (1, -1))
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + 1)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def EML(y_true, y_pred):
    gamma = 1.1
    alpha = 0.48
    smooth = 1.
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true*y_pred)
    dice_loss = (2.*intersection + smooth)/(K.sum(y_true*y_true)+K.sum(y_pred * y_pred)+smooth)
    y_pred = K.clip(y_pred, K.epsilon())
    pt_1 = tf.where(tf.equal(y_true, 1),y_pred,tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0),y_pred,tf.zeros_like(y_pred))
    focal_loss = -K.mean(alpha*K.pow(1. -pt_1, gamma)*K.log(pt_1),axis=-1)\
                   -K.mean(1-alpha)*K.pow(pt_0,gamma)*K.log(1. -pt_0),axis=-1)
    return focal_loss - K.log(dice_loss)
