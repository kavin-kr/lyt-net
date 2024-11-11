
import tensorflow as tf
from keras import layers,models,Input,Concatenate,Add

def rgb_to_yuv(rgb):
    yuv = tf.image.rgb_to_yuv(rgb)
    y, u, v = tf.split(yuv, 3, axis=-1)
    # y,u,v represent luminance, chromablue and chromared
    return y,u,v

def yuv_to_rgb(yuv):
    return tf.image.yuv_to_rgb(yuv)

def LYT(input_shape,num_kernels = 32):
    inputs = Input(shape=input_shape)
    y, u, v = layers.Lambda(rgb_to_yuv)(inputs)

    y_processed = luminance_process(y, num_kernels)
    u_cwd = CWD(u)
    v_cwd = CWD(v)
    concat1= layers.Concatenate()([u_cwd, v_cwd])
    uv_processed = layers.Conv2D(num_kernels, (1, 1), padding = 'same', activation='relu')(concat1)
    alpha = 0.5 # try 0.2?
    scaled_y = alpha * y_processed
    msef_input = uv_processed + scaled_y
    msef_output = MSEF(msef_input)
    concat2= layers.Concatenate()([uv_processed , y_processed, msef_output])
    conv1 = layers.Conv2D(num_kernels, (3, 3), activation='relu', padding='same')(concat2)
    # why tanh?
    lyt_output = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(conv1)
    lyt_output = layers.Lambda(yuv_to_rgb)(lyt_output)

    return lyt_output


def luminance_process(y, num_kernels):
    conv1 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu')(y)
    #Stride = 2 ....8?
    pooled = layers.MaxPooling2D((3, 3), strides=(2, 2))(conv1)
    mhsa_output = MHSA(pooled,num_kernels)
    upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(mhsa_output)
    skip_conn1 = upsampled +  conv1
    lum_processed = layers.Conv2D(num_kernels, (1, 1), padding = 'same', activation='relu')(skip_conn1)

    return lum_processed


def CWD(input_layer,num_kernels=16):
    # Denoise  - try diff number of kernels?
    conv1 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu')(input_layer)
    conv2 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu',strides=2)(conv1)
    conv3 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu',strides=2)(conv2)
    conv4 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu',strides=2)(conv3)
    mhsa_output = MHSA(conv4,num_kernels)
    # interpolation upsampling
    upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(mhsa_output)
    skip_conn1 = upsampled + conv4 + conv3 + conv2 + conv1

    # why tanh here?

    conv5 = layers.Conv2D(1, (3,3), padding='same', activation='tanh')(skip_conn1)
    skip_conn2 = conv5+input_layer
    conv6 = layers.Conv2D(1, (3,3), padding='same', activation='tanh')(skip_conn2)
    # done inside block
    cwd_output = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu',strides=2)(conv6) 

    return cwd_output

def MHSA(input_layer, embedding_size, num_heads):
    q_fc = layers.Dense(embedding_size)(input_layer)
    k_fc = layers.Dense(embedding_size)(input_layer)
    v_fc = layers.Dense(embedding_size)(input_layer)

    q_split = tf.concat(tf.split(q_fc, num_heads, axis=-1), axis=0)
    k_split = tf.concat(tf.split(k_fc, num_heads, axis=-1), axis=0)
    v_split = tf.concat(tf.split(v_fc, num_heads, axis=-1), axis=0)

    matmul_qk = tf.matmul(q_split, k_split, transpose_b=True)
    dk = tf.cast(tf.shape(k_split)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    scaled_attention = tf.matmul(attention_weights, v_split)

    concat_attention = tf.concat(tf.split(scaled_attention, num_heads, axis=0), axis=-1)

    output = layers.Dense(embedding_size)(concat_attention)

    return output

def MSEF(input_layer, num_kernels):
    norm_layer = layers.LayerNormalization()(input_layer)
    global_pool = layers.GlobalAveragePooling2D()(norm_layer)
    s_reduced = layers.Dense(num_kernels // 16, activation='relu')(global_pool)
    s_expanded = layers.Dense(num_kernels, activation='tanh')(s_reduced)
    s_expanded = layers.Reshape((1, 1, num_kernels))(s_expanded)
    dw_conv = layers.DepthwiseConv2D((3, 3), padding='same')(norm_layer)
    scaled_features = layers.Multiply()([dw_conv, s_expanded])
    output = layers.Add()([scaled_features, input_layer])

    return output