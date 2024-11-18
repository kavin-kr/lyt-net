import tensorflow as tf
import cv2
import numpy as np

# Initialize VGG for perceptual loss calculation
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
feature_layer = 'block3_conv3'
loss_model = tf.keras.Model(inputs=vgg.input,outputs=vgg.get_layer(feature_layer).output)


def smooth_l1_loss(target, pred, beta=1.0):
    diff = tf.abs(target - pred)
    less_than_beta = tf.cast(tf.less(diff, beta), tf.float32)
    
    smooth_l1_loss = (less_than_beta * 0.5 * tf.square(diff) / beta +
                      (1 - less_than_beta) * (diff - 0.5 * beta))
    smooth_l1_loss = tf.reduce_mean(smooth_l1_loss)
    
    return smooth_l1_loss



def perc_loss(target, pred,feature_model):
    #vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    #vgg.trainable = False
    target = tf.keras.applications.vgg19.preprocess_input(target * 255.0)
    pred = tf.keras.applications.vgg19.preprocess_input(pred * 255.0)
    #feature_layer = 'block3_conv3'
    #feature_model = tf.keras.Model(inputs=vgg.input, 
                                  # outputs=vgg.get_layer(feature_layer).output)
    true_features = feature_model(target)
    pred_features = feature_model(pred)
    perc_loss = tf.reduce_mean(tf.square(true_features - pred_features))

    return perc_loss

# Define Histogram Loss
def hist_loss(target,pred):
    bins=256
    sigma=0.01
    bin_edges = tf.linspace(0.0, 1.0, bins)
    target_hist = tf.reduce_sum(tf.exp(-0.5 * ((target[..., tf.newaxis] - bin_edges) / sigma) ** 2), axis=0)
    pred_hist = tf.reduce_sum(tf.exp(-0.5 * ((pred[..., tf.newaxis] - bin_edges) / sigma) ** 2), axis=0)
    
    target_hist = target_hist/ tf.reduce_sum(target_hist)
    pred_hist   = pred_hist /  tf.reduce_sum(pred_hist)

    hist_loss = tf.reduce_mean(tf.abs(target_hist - pred_hist))

    return hist_loss

def psnr_loss(target,pred):
    psnr = tf.image.psnr(target, pred, max_val=1.0)
    psnr_loss = tf.reduce_mean(40 - psnr) #generally perfect images are 30 to 50db range
    return psnr_loss

def color_loss(target, pred):
    # Almost like global average pooling
    target_means = tf.reduce_mean(target, axis=[1, 2])
    pred_means = tf.reduce_mean(pred, axis=[1, 2]) 
    channel_diff = tf.abs(target_means - pred_means)    
    color_loss = tf.reduce_sum(channel_diff, axis=-1) #across channel mean
    color_loss = tf.reduce_mean(color_loss)    
    return color_loss

def ms_ssim_loss(target, pred, max_val=1.0, power_factors=[0.0448, 0.2856, 0.3001]):
    # compare scales at 3 diff scales with more weightage to lower scales
    # the power factors will be normalized internally but the function itself to add up to 1 
    ms_ssim = tf.image.ssim_multiscale(target, pred, max_val, power_factors=power_factors)
    ms_ssim_loss = 1 - tf.reduce_mean(ms_ssim)
    return ms_ssim_loss

def hybrid_loss(target, pred):
    
    # normalize to 0,1 as it is a requirement for most losses below
    target = (target + 1.0) / 2.0
    pred = (pred + 1.0) / 2.0

    alpha_1 = 0.06
    alpha_2 = 0.05
    alpha_3 = 0.5
    alpha_4 = 0.0083
    alpha_5 = 0.25

    L_s = smooth_l1_loss(target, pred)
    L_perc = perc_loss(target, pred,loss_model)
    L_hist = hist_loss(target, pred)
    L_psnr = psnr_loss(target, pred)
    L_color = color_loss(target, pred)
    L_ms_ssim = ms_ssim_loss(target, pred)    
    

    hybrid_loss = L_s + alpha_1*L_perc + alpha_2*L_hist + alpha_3*L_psnr + alpha_4*L_color+ alpha_5*L_ms_ssim
    hybrid_loss = tf.reduce_mean(hybrid_loss)
    return hybrid_loss
