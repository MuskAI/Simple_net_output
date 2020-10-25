from keras.layers import Conv2DTranspose,Conv2D, add,DepthwiseConv2D,Input, MaxPooling2D,BatchNormalization, Add,multiply,Multiply
from keras.layers import Dropout, Concatenate,Activation, AveragePooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.layers.core import  Lambda
from keras.initializers import glorot_uniform
import numpy as np
from keras.engine import Layer,InputSpec
from itertools import product
from subpixel_conv2d import SubpixelConv2D
from keras.utils import conv_utils
from keras.backend.common import normalize_data_format
# from DualAttention import CAM,PAM
from keras import initializers
class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
            input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
            input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
                                                   int(inputs.shape[2]*self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    # valid mean no padding / glorot_uniform equal to Xaiver initialization - Steve

    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)
    # Third component of main path (≈2 lines)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    ### END CODE HERE ###

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1', padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    # X = layers.add([X, X_shortcut])
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X
def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True,name='none'):
    if name=='none':
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
    else:
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal',name=name)(x)
        x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x


def aspp(x, input_shape, out_stride):
    # 膨胀率6 12 12
    b0 = Conv2D(128, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(128, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(128, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(128, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape = int(input_shape[0] / out_stride)
    out_shape1 = int(input_shape[1] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape1))(x)
    b4 = Conv2D(128, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape, out_shape1))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x

def superPixNet(input_shape=(320,320,3), classes=18,model_name='resNet50'):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    """
    采用u-net模型，编码器使用四个block的resnet 50 解码器采用 全局特征和局部特征model形成一个解码block，
    上采样使用sub-pixel，最后的输出结果是先计算出八张图 然后在八张图的基础上卷集得到无类别的双边缘图
    下一版本中需要加入可形变卷集进行实验
    """

    """
    8.12日网络，采用stage_out 其中对应的gt进行下采样，然后再上采样回到原来的尺寸
    中间层的skip，使用上一层的特征传递下来，mutil cat 1*1
    底层加一个Aspp模块增加感受野
    resblock的数量按照实际情况决定多少个，不需要太多
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = Conv2D(64, (7, 7), padding='same', strides=(1, 1), kernel_initializer='he_normal', activation='relu',
               name='stage_conv_1')(X_input)
    X = BatchNormalization(axis=3, name="stage_bn_conv1")(X)
    X = Activation("relu")(X)
    X = convolutional_block(X, f=3, filters=[32, 32, 64], stage=1, block="a", s=1)
    X = identity_block(X, f=3, filters=[32, 32, 64], stage=1, block="b")
    skip320_160 = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_normal', activation='relu',
               name='stage_1_skip_conv')(X)
    skip320_160 = BatchNormalization(axis=3, name="stage_bn_skip_conv1")(skip320_160)
    skip320_160 = Activation("relu")(skip320_160)
    X = Dropout(0.5)(X)
    skip_1 = Conv2d_BN(X, 32, 1, name='skip1')
    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 128], stage=2, block="a", s=2)
    X = identity_block(X, f=3, filters=[64, 64, 128], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 128], stage=2, block="c")
    skip160_80 = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_normal', activation='relu',
                        name='stage_2_skip_conv')(X)
    skip160_80 = BatchNormalization(axis=3, name="stage_2_skip_bn_1")(skip160_80)
    skip160_80 = Activation("relu")(skip160_80)
    X = Dropout(0.5)(X)
    skip160_160 = Conv2d_BN(X, 32, 1, name='stage_16_skip')

    # Stage 3 (≈4 lines) 80*80
    X = convolutional_block(X, f=3, filters=[128, 128, 128], stage=3, block="a", s=2)
    X = identity_block(X, f=3, filters=[128, 128, 128], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 128], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 128], stage=3, block="d")
    skip80_40 = Conv2D(64, (3, 3), padding='same', strides=(2, 2), kernel_initializer='he_normal', activation='relu',
                       name='stage_3_skip_conv')(X)
    skip80_40 = BatchNormalization(axis=3, name="stage_3_skip_bn_1")(skip80_40)
    skip80_40 = Activation("relu")(skip80_40)
    X = Dropout(0.5)(X)
    skip80_80 = Conv2d_BN(X, 32, 1, name='stage_8_skip')

    # Stage 4 (≈6 lines) 40*40
    X = convolutional_block(X, f=3, filters=[128, 128, 256], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[128, 128, 256], stage=4, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 256], stage=4, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 256], stage=4, block="d")
    X = identity_block(X, f=3, filters=[128, 128, 256], stage=4, block="e")
    X = identity_block(X, f=3, filters=[128, 128, 256], stage=4, block="f")
    X = Dropout(0.5)(X)
    skip40_40 = Conv2d_BN(X, 64, 1, name='stage_4_skip')

    # Stage 5 (≈3 lines) 20*20
    X = convolutional_block(X, f=3, filters=[128, 128, 256], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[128, 128, 256], stage=5, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 256], stage=5, block="c")
    X = Dropout(0.5)(X)
    # 输出依然是20*20
    X = aspp(X, input_shape, 16)
    X = Conv2d_BN(X, 256, 1, name='stage_5_aspp')

    # up_stage1 20---40 开始解码器部分  256---->64
    sub_pixel1 = SubpixelConv2D(upsampling_factor=2, name='sub_pixel_1')(X)
    sub_pixel1 = BatchNormalization(axis=3)(sub_pixel1)
    sub_pixel1 = Activation('relu')(sub_pixel1)

    multi_skip=Multiply()([skip40_40,skip80_40])
    skip = Concatenate(axis=-1)([skip40_40, skip80_40,multi_skip])
    skip = Conv2d_BN(skip, 64, 3, name='skip_conv_1')
    X_up = Concatenate(axis=3)([skip, sub_pixel1])
    X_up = Conv2d_BN(X_up, 128, 3, name='up_stage_conv_1')
    X_up = Conv2d_BN(X_up, 256, 3, name='up_stage_conv_2')
    X_out4_pre = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', activation='relu',
                    name='X_out4_pre')(X_up)
    X_out4=Conv2D(1, (1,1), padding='same', strides=(1,1), kernel_initializer='he_normal',activation='sigmoid',name='X_out4')(X_out4_pre)
    sub_pixel_out4 = SubpixelConv2D(upsampling_factor=8, name='sub_pixel_out4')(X_out4_pre)
    sub_pixel_out4 = Activation('sigmoid')(sub_pixel_out4)


    sub_pixel2 = SubpixelConv2D(upsampling_factor=2, name='sub_pixel-2')(X_up)
    sub_pixel2 = BatchNormalization(axis=3)(sub_pixel2)
    sub_pixel2 = Activation('relu')(sub_pixel2)

    multi_skip = Multiply()([skip80_80, skip160_80])
    skip = Concatenate(axis=-1)([skip80_80, skip160_80,multi_skip])
    skip = Conv2d_BN(skip, 64, 3, name='skip2_conv_1')
    X_up = Concatenate(axis=3)([skip, sub_pixel2])
    X_up = Conv2d_BN(X_up, 128, 3, name='up2_stage_conv_1')
    X_up = Conv2d_BN(X_up, 256, 3, name='up2_stage_conv_2')
    X_out8_pre = Conv2D(16, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', activation='relu',
                        name='X_out8_pre')(X_up)
    X_out8 = Conv2D(1, (1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal', activation='sigmoid',
                    name='X_out8')(X_out8_pre)
    sub_pixel_out8 = SubpixelConv2D(upsampling_factor=4, name='sub_pixel_out8')(X_out8_pre)
    sub_pixel_out8 = Activation('sigmoid')(sub_pixel_out8)

    # up_stage2 80---160 开始解码器部分  256---->64
    sub_pixel3 = SubpixelConv2D(upsampling_factor=2, name='sub_pixel-3')(X_up)
    sub_pixel3 = BatchNormalization(axis=3)(sub_pixel3)
    sub_pixel3 = Activation('relu')(sub_pixel3)

    multi_skip = Multiply()([skip160_160, skip320_160])
    skip = Concatenate(axis=-1)([skip160_160, skip320_160,multi_skip])
    skip = Conv2d_BN(skip, 64, 3, name='skip3_conv_1')
    X_up = Concatenate(axis=3)([skip, sub_pixel3])
    X_up = Conv2d_BN(X_up, 128, 3, name='up3_stage_conv_1')
    X_up = Conv2d_BN(X_up, 128, 3, name='up3_stage_conv_2')

    X_out16_pre = Conv2D(4, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', activation='relu',
                        name='X_out16_pre')(X_up)
    X_out16 = Conv2D(1, (1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal', activation='sigmoid',
                    name='X_out16')(X_out16_pre)
    sub_pixel_out16 = SubpixelConv2D(upsampling_factor=2, name='sub_pixel_out16')(X_out16_pre)
    sub_pixel_out16 = Activation('sigmoid')(sub_pixel_out16)

    # up_stage2 160---320 开始解码器部分  128---->32
    sub_pixel4 = SubpixelConv2D(upsampling_factor=2, name='sub_pixel-4')(X_up)
    sub_pixel4 = BatchNormalization(axis=3)(sub_pixel4)
    sub_pixel4 = Activation('relu')(sub_pixel4)
    X_up = Concatenate(axis=3)([skip_1, sub_pixel4])
    X_up = Conv2d_BN(X_up, 128, 3, name='up4_stage_conv_1')
    final = Conv2d_BN(X_up, 128, 3, name='up4_stage_conv_2')
    skip_fianl = Conv2D(3, (1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal', activation='relu',
                        name='up4_stage_conv_3')(final)
    skip_fianl = BatchNormalization(axis=3, name="up4_stage_conv_bn")(skip_fianl)
    skip_fianl = Activation("sigmoid")(skip_fianl)

    X1 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='new1_e_chanel_1_1')(final)
    X2 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='new1_e_chanel_10')(final)
    X3 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='new1_e_chanel_11')(final)
    X4 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='new1_e_chanel0_1')(final)
    X5 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='new1_e_chanel01')(final)
    X6 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='new1_e_chanel1_1')(final)
    X7 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='new1_e_chanel10')(final)
    X8 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', name='new1_e_chanel11')(final)

    X_all = Concatenate(axis=-1)([X1, X2, X3, X4, X5, X6, X7, X8, skip_fianl,sub_pixel_out4,sub_pixel_out8,sub_pixel_out16])
    X_fuse = Conv2D(16, (3, 3), activation='relu', padding='same', name='fuse1')(X_all)
    X_fuse = Conv2D(16, (3, 3), activation='relu', padding='same', name='fuse2')(X_fuse)
    X_fuse = Conv2D(8, (3, 3), activation='relu', padding='same', name='fuse3')(X_fuse)
    X_fuse = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='fuse')(X_fuse)
    model = Model(inputs=X_input, outputs=[X1, X2, X3, X4, X5, X6, X7, X8, X_out4,X_out8,X_out16,X_fuse],
                  name="Eight_ResNet50")
    model.summary()
    return model
# 处理双通道切分并且进行logical_or
def slices_or(x):
    x0=x[:,:,:,0:1]
    x0=tf.greater(x0,0.5)
    x1=x[:,:,:,1:2]
    x1=tf.greater(x1,0.5)
    x_final=tf.logical_or(x0,x1)
    x_final=tf.cast(x_final,tf.float32)
    return x_final

def slice(x,index):
    return x[:,:,:,index]
#计算W_gd
def get_distance(inputs):
    x,y,z=inputs
    D = tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1))
    # 求张量最小值
    k=tf.reduce_min(D)
    W_g=tf.keras.backend.exp(k-D)
    # W_g=tf.matmul(W_g,z)
    return W_g
#计算W_l
def get_distance_l(inputs):
    x=inputs
    W_l=1-x
    return W_l
def get_C(inputs):
    x,y=inputs
    print(y.shape)
    C = tf.multiply(x, y)
    return C
def get_W_g(inputs):
    x=inputs
    W_g=tf.tile(x,[1,1,1,32])
    print(W_g.shape)
    return W_g
def get_W_g2(inputs):
    x=inputs
    W_g=tf.tile(x,[1,1,1,128])
    print(W_g.shape)
    return W_g

def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = 1000*tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def cross_entropy_balanced1(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits

    # CRFLoss = CRF(y_true, y_pred)
    costSSIM = SSIM(y_true, y_pred)
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))


    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = 1000*tf.reduce_mean(cost * (1 - beta))

    cost +=10*costSSIM

    # check if image has no edge pixels return 0 else return complete error function
    # return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)+tf.where(tf.equal(count_pos, 0.0), 0.0, CRFLoss)
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
    # return CRFLoss
def CRF(y_true, y_pred):
    SIGMA = 1.0
    dtype = tf.float32
    # Input images batch
    img = y_pred
    img_shape = tf.shape(img)
    img_height = img_shape[1]
    img_width = img_shape[2]
    # Compute 3 x 3 block means
    mean_filter = tf.ones((3, 3), dtype) / 9
    mean_filter = mean_filter[:, :, tf.newaxis, tf.newaxis]
    img_mean = tf.nn.conv2d(img,
                            mean_filter,
                            [1, 1, 1, 1], 'SAME')
    # Remove 1px border
    img_clip = img
    # Difference between pixel intensity and its block mean
    x_diff = img_clip - img_mean
    # Compute neighboring pixel loss contributions
    contributions = []
    for i, j in product((-1, 0, 1), repeat=2):
        if i == j == 0: continue
        # Take "shifted" image
        displaced_img = img
        # Compute difference with mean of corresponding pixel block
        y_diff = displaced_img - img_mean
        # Weights formula
        weight = 1 + x_diff * y_diff / (SIGMA ** 2)
        # Contribution of this displaced image to the loss of each pixel
        contribution = weight * displaced_img
        contributions.append(contribution)
    contributions = tf.add_n(contributions)

    # contributions = y_true*contributions*(y_true-y_pred)

    # Compute loss value
    loss = tf.reduce_mean(y_true*tf.squared_difference(img_clip, contributions)*(y_true-y_pred))
    return  loss

def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')

def _tf_fspecial_gauss(size, sigma=1.5):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def SSIM(img1, img2, k1=0.01, k2=0.02, L=1, window_size=11):
    """
    The function is to calculate the ssim score
    """

    # img1 = tf.expand_dims(img1, 0)
    # img1 = tf.expand_dims(img1, -1)
    # img2 = tf.expand_dims(img2, 0)
    # img2 = tf.expand_dims(img2, -1)

    window = _tf_fspecial_gauss(window_size)


    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')


    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides = [1 ,1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

    c1 = (k1*L)**2
    c2 = (k2*L)**2

    ssim_map = 1-((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

    return tf.reduce_mean(ssim_map)

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def load_weights_from_hdf5_group_by_name(model, filepath):
    ''' Name-based weight loading '''

    import h5py

    f = h5py.File(filepath, mode='r')

    flattened_layers = model.layers
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '") expects ' +
                                str(len(symbolic_weights)) +
                                ' weight(s), but the saved weights' +
                                ' have ' + str(len(weight_values)) +
                                ' element(s).')
            # set values
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
                K.batch_set_value(weight_value_tuples)

if __name__ == "__main__":
    # model
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model = superPixNet()

