from tensorflow import reduce_sum
from tensorflow.keras import backend as  K

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jaccard_loss(y_true,y_pred):
    return 1.0-jacard_coef(y_true, y_pred)

def diceCoef(y_true, y_pred, smooth=1e-6):   
    y_true_f = K.flatten(y_true)    
    y_pred_f = K.flatten(y_pred)    
    intersection = K.sum(y_true_f * y_pred_f)    
    loss =  (2 * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
    return loss

def diceCoefLoss(y_true, y_pred):
    return 1.0 - diceCoef(y_true, y_pred)

def bceDiceLoss(y_true, y_pred):
    loss = 0.5*tf.keras.losses.binary_crossentropy(y_true, y_pred) + diceCoef(y_true, y_pred)
    return loss

def total_loss(y_true,y_pred):
    loss = tversky_loss(y_true,y_pred) + 0.1 * diceCoefLoss (y_true, y_pred)
    return loss

def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha * false_neg + (1-alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
