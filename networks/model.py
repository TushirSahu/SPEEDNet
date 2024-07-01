from .layers import *
from tensorflow.keras.layers import Conv2D, Input, AveragePooling2D, MaxPooling2D, concatenate
from tensorflow.keras.models import Model
from .salient_attention import *


def build_model(start_neurons):
    """
    SPEEDNet Network
    """
    input_layer = Input((224, 224, 3),dtype='float32')

    conv1 = conv_block(input_layer,3,start_neurons * 2)
    conv1 = conv_block(conv1,3,start_neurons * 2)

    pool1 = MaxPooling2D((2, 2))(conv1)
    avg_pool1 = AveragePooling2D((2,2))(conv1)
    
    input2 = input_layer
        
    dwns1 = MaxPooling2D(2,2)(input2)
    attn1 = SalientAttentionBlock(conv1, dwns1, pool1, avg_pool1,start_neurons * 2,10)
    

    conv2 = conv_block(attn1,3,start_neurons * 4)
    conv2 = conv_block(conv2,3,start_neurons * 4)
    
    pool2 = MaxPooling2D((2, 2))(conv2)
    avg_pool2 = AveragePooling2D((2,2))(conv2)
    
    dwns2 = MaxPooling2D(4,4)(input2)
    attn2 = SalientAttentionBlock(conv2, dwns2, pool2,avg_pool2, start_neurons * 4,20)


    conv3 = conv_block(attn2,3,start_neurons * 8)
    conv3 = conv_block(conv3,3,start_neurons * 8)

    pool3 = MaxPooling2D((2, 2))(conv3)
    avg_pool3 = AveragePooling2D((2,2))(conv3)
    
    dwns3 = MaxPooling2D(8,8)(input2)
    attn3 = SalientAttentionBlock(conv3, dwns3, pool3,avg_pool3, start_neurons * 8,30)


    conv4 = conv_block(attn3,3,start_neurons * 16)
    conv4 = conv_block(conv4,3,start_neurons * 16)

    pool4 = MaxPooling2D((2, 2))(conv4)
    avg_pool4 = AveragePooling2D((2,2))(conv4)
    
    dwns4 = MaxPooling2D(16,16)(input2)
    attn4 = SalientAttentionBlock(conv4, dwns4, pool4,avg_pool4,start_neurons * 16,40)

    convm=bottleneck(attn4,start_neurons * 32)
    
    deconv4 = keras.layers.UpSampling2D((2, 2),data_format="channels_last")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = keras.layers.UpSampling2D((2, 2),data_format="channels_last")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = keras.layers.UpSampling2D((2, 2),data_format="channels_last")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = keras.layers.UpSampling2D((2, 2),data_format="channels_last")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons *2 ,(3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv1)
    
    
    output_layer = layers.BatchNormalization(axis=3)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(output_layer)

    model=Model(inputs =input_layer,outputs=output_layer)
    return model

