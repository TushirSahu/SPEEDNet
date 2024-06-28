import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers

class Involution(keras.layers.Layer):
    def __init__(
        self, channel, kernel_size, stride,dilation_rate,name,group_number=1,reduction_ratio=1):
        super().__init__(name=name)

      
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.dialtion_rate = dilation_rate
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

       
        height = height // self.stride
        width = width // self.stride

   
        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="SAME"
            )
            if self.stride > 1
            else tf.identity
        )
      
        self.kernel_gen = keras.Sequential(
            [
#                 DeformableConvLayer(filters=6, kernel_size=3, strides=1, padding='valid', dilation_rate=1, num_deformable_group=1)
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1,dilation_rate=self.dialtion_rate
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )
        self.c = keras.layers.Conv2D(self.channel,(1,1),padding='same')

    def call(self, x):
    
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        kernel = self.kernel_reshape(kernel)

        
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

     
        input_patches = self.input_patches_reshape(input_patches)

     
        output = tf.multiply(kernel, input_patches)
 
        output = tf.reduce_sum(output, axis=3)
        output = self.output_reshape(output)
            
        return output, kernel




# model code starts here
def conv_block(x, filter_size, size):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    return conv



def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv1 = layers.Conv2D(filters, (3,3), padding="same",dilation_rate=(2,2))(x)
#     conv1d = layers.Conv2D(filters, (1,1), padding="same",dilation_rate=(2,2))(conv1)
#     conv1 = add([conv1,conv1d])
    conv1 = layers.BatchNormalization(axis=3)(conv1)
    conv1 = layers.Activation("relu")(conv1)
    
    
    conv2 = layers.Conv2D(filters, (3,3), padding="same",dilation_rate=(4,4))(conv1)
#     conv2d = layers.Conv2D(filters, (1,1), padding="same",dilation_rate=(4,4))(conv2)
#     conv2 = add([conv2d,conv2])
    conv2 = layers.BatchNormalization(axis=3)(conv2)
    conv2 = layers.Activation("relu")(conv2)
    
#     conv = add([conv2,conv1])
    return conv2
