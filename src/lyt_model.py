
import tensorflow as tf
from keras import layers,models,Input

# Convert RGB to YUV color space
def rgb_to_yuv(rgb):
    yuv = tf.image.rgb_to_yuv(rgb)
    y, u, v = tf.split(yuv, 3, axis=-1)
    # y,u,v represent luminance, chromablue and chromared
    return y,u,v

# Convert YUV to RGB color space
def yuv_to_rgb(yuv):
    return tf.image.yuv_to_rgb(yuv)

# Define the LYT architecture
def LYT(input_shape,num_kernels = 32):
    inputs = Input(shape=input_shape)
    y, u, v = layers.Lambda(rgb_to_yuv)(inputs)

    y_processed = luminance_process(y, num_kernels)
    u_cwd = CWD(u) + u
    v_cwd = CWD(v) + v
    concat1= layers.Concatenate()([u_cwd, v_cwd])
    uv_processed = layers.Conv2D(num_kernels, (1, 1), padding = 'same', activation='relu')(concat1)
    alpha = 0.2 
    scaled_y = alpha * y_processed
    msef_input = uv_processed + scaled_y
    msef_output = MSEF(msef_input,num_kernels)
    concat2= layers.Concatenate()([uv_processed , y_processed, msef_output])
    conv1 = layers.Conv2D(num_kernels, (3, 3), activation='relu', padding='same')(concat2)
    
    lyt_output = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(conv1)
    lyt_output = layers.Lambda(yuv_to_rgb)(lyt_output)

    model = models.Model(inputs=inputs, outputs=lyt_output)

    return model

# Process the luminance (Y) channel
def luminance_process(y, num_kernels):
    conv1 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu')(y)
    
    pooled = layers.MaxPooling2D((3, 3), strides=(8, 8))(conv1)
    mhsa_output = MHSA(pooled,num_kernels)
    upsampled = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(mhsa_output)
    skip_conn1 = upsampled +  conv1
    lum_processed = layers.Conv2D(num_kernels, (1, 1), padding = 'same', activation='relu')(skip_conn1)

    return lum_processed

# Process the chrominance (U and V) channels through Denoiser block
def CWD(input_layer,num_kernels=16):
    
    conv1 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu')(input_layer)
    conv2 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu',strides=2)(conv1)
    conv3 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu',strides=2)(conv2)
    conv4 = layers.Conv2D(num_kernels, (3, 3), padding = 'same', activation='relu',strides=2)(conv3)
    mhsa_output = MHSA(conv4,num_kernels)

    # interpolation upsampling
    upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(mhsa_output)
    upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(upsampled+conv3)
    upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(upsampled+conv2)
    skip_conn1 = upsampled + conv1    

    conv5 = layers.Conv2D(1, (3,3), padding='same', activation='tanh')(skip_conn1)
    skip_conn2 = conv5+input_layer
    conv6 = layers.Conv2D(1, (3,3), padding='same', activation='tanh')(skip_conn2)
    
    cwd_output = layers.Conv2D(32, (3, 3), padding = 'same', activation='relu')(conv6) 

    return cwd_output

# Multi-Head Self-Attention (MHSA) block
def MHSA(input_layer, embedding_size, num_heads=4):
    q_fc = layers.Dense(embedding_size)(input_layer)
    k_fc = layers.Dense(embedding_size)(input_layer)
    v_fc = layers.Dense(embedding_size)(input_layer)

    q_split = layers.Lambda(lambda x: tf.concat(tf.split(x, num_heads, axis=-1), axis=0))(q_fc)
    k_split = layers.Lambda(lambda x: tf.concat(tf.split(x, num_heads, axis=-1), axis=0))(k_fc)
    v_split = layers.Lambda(lambda x: tf.concat(tf.split(x, num_heads, axis=-1), axis=0))(v_fc)

    matmul_qk = layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([q_split, k_split])
    dk = layers.Lambda(lambda x: tf.cast(tf.shape(x)[-1], tf.float32))(k_split)
    scaled_attention_logits = layers.Lambda(lambda x: x[0] / tf.math.sqrt(x[1]))([matmul_qk, dk])
    attention_weights = layers.Lambda(lambda x: tf.nn.softmax(x, axis=-1))(scaled_attention_logits)
    scaled_attention = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([attention_weights, v_split])

    concat_attention = layers.Lambda(lambda x: tf.concat(tf.split(x, num_heads, axis=0), axis=-1))(scaled_attention)
    output = layers.Dense(embedding_size)(concat_attention)

    return output

# Multi-Scale Enhancement Fusion (MSEF) block
def MSEF(input_layer, num_kernels):
    
    norm_layer = layers.LayerNormalization()(input_layer)
    dw_conv = layers.DepthwiseConv2D((3, 3), padding='same')(norm_layer)
    global_pool = layers.GlobalAveragePooling2D()(norm_layer)
    reduced_descriptor = layers.Dense(num_kernels // 16, activation='relu')(global_pool)
    expanded_descriptor = layers.Dense(num_kernels, activation='tanh')(reduced_descriptor)
    expanded_descriptor = layers.Reshape((1, 1, num_kernels))(expanded_descriptor)
    scaled_features = layers.Multiply()([dw_conv, expanded_descriptor])
    output = layers.Add()([scaled_features, input_layer])

    return output