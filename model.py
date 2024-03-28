import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf



def SqueezeAndExcite(inputs, ratio=8):
    init = inputs
    filters = init.shape[-1]
    se_shape = (1,1,filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu',kernel_initializer='he_normal',use_bias=False)(se)
    se = Dense(filters, activation='sigmoid',kernel_initializer='he_normal',use_bias=False)(se)

    X = init * se
    return X




def ASPP(inputs):
    """Image Pooling"""
    print(inputs.shape)
    shape = inputs.shape
    y1 = AveragePooling2D(pool_size=(shape[1],shape[2]))(inputs)    ## coverted into 1X1
    print(y1.shape)

    y1 = Conv2D(256,1,padding="same",use_bias=False)(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1],shape[2]),interpolation='bilinear')(y1)
    print(y1.shape)

    """ 1X1 conv """
    y2 = Conv2D(256,1,padding="same",use_bias=False)(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    """ 3X3 conv rate=6 """

    y3 = Conv2D(256,3,padding="same",use_bias=False,dilation_rate=6)(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    """ 3X3 conv rate=12 """

    y4 = Conv2D(256,3,padding="same",use_bias=False,dilation_rate=12)(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    """ 3X3 conv rate=18 """

    y5 = Conv2D(256,3,padding="same",use_bias=False,dilation_rate=18)(inputs)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256,1,padding="same",use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y



def deeplabv3_plus(shape):
    """Input"""
    inputs=Input(shape)

    """Encoder"""
    encoder = ResNet50(weights="imagenet",include_top=False,input_tensor=inputs)

    image_features = encoder.get_layer("conv4_block6_out").output
    x_a = ASPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation='bilinear')(x_a)
    print(x_a.shape)

    x_b = encoder.get_layer("conv2_block2_out").output
    x_b = Conv2D(filters=48,kernel_size=1,padding='same',use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation("relu")(x_b)

    X = Concatenate()([x_a, x_b])
    X = SqueezeAndExcite(X)
    print(X.shape)

    X = Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    print(X.shape)

    X = Conv2D(filters=256,kernel_size=3,padding='same',use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = SqueezeAndExcite(X)


    X = UpSampling2D((4, 4), interpolation='bilinear')(X)
    X = Conv2D(1,1)(X)
    X = Activation("sigmoid")(X)
    print(X.shape)

    """model building"""
    model = Model(inputs, X)
    return model


    

if __name__ == "__main__":
    model = deeplabv3_plus((512,512,3))
    model.summary()

