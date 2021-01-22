"""
MIT License

Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

from .common import YOLOConv2D

class SPP(Model):
    """
    Spatial Pyramid Pooling layer(SPP)
    """

    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = layers.MaxPooling2D((13, 13), strides=1, padding="same")
        self.pool2 = layers.MaxPooling2D((9, 9), strides=1, padding="same")
        self.pool3 = layers.MaxPooling2D((5, 5), strides=1, padding="same")
        self.concat = layers.Concatenate(axis=-1)

    def call(self, x):
        return self.concat([self.pool1(x), self.pool2(x), self.pool3(x), x])


class CSPDarknet53Tiny(Model):
    def __init__(
        self,
        activation: str = "mish",
        kernel_regularizer=None,
    ):
        super(CSPDarknet53Tiny, self).__init__(name="CSPDarknet53Tiny")
        self.conv0 = YOLOConv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv1 = YOLOConv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv2 = YOLOConv2D(
            filters=64,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv3 = YOLOConv2D(
            filters=32,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv4 = YOLOConv2D(
            filters=32,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat3_4 = layers.Concatenate(axis=-1)
        self.conv5 = YOLOConv2D(
            filters=64,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat2_5 = layers.Concatenate(axis=-1)
        self.maxpool5 = layers.MaxPool2D((2, 2), strides=2, padding="same")

        self.conv6 = YOLOConv2D(
            filters=128,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv7 = YOLOConv2D(
            filters=64,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv8 = YOLOConv2D(
            filters=64,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat7_8 = layers.Concatenate(axis=-1)
        self.conv9 = YOLOConv2D(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat6_9 = layers.Concatenate(axis=-1)
        self.maxpool9 = layers.MaxPool2D((2, 2), strides=2, padding="same")

        self.conv10 = YOLOConv2D(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv11 = YOLOConv2D(
            filters=128,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv12 = YOLOConv2D(
            filters=128,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat11_12 = layers.Concatenate(axis=-1)
        self.conv13 = YOLOConv2D(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat10_13 = layers.Concatenate(axis=-1)
        self.maxpool13 = layers.MaxPool2D((2, 2), strides=2, padding="same")
        self.conv14 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.spp = SPP()
        self.conv15 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv16 = YOLOConv2D(
            filters=1024,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv17 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, x): # shape=(None, 512, 512, 3)
        x1 = self.conv0(x)  #shape=(None, 256, 256, 32)
        x1 = self.conv1(x1) #shape=(None, 128, 128, 64)
        x1 = self.conv2(x1) # shape=(None, 128, 128, 64)
        _, x2 = tf.split(x1, 2, axis=-1) #shape=(None, 128, 128, 32)
        x2 = self.conv3(x2) #shape=(None, 128, 128, 32)
        x3 = self.conv4(x2) # shape=(None, 128, 128, 32)
        x3 = self.concat3_4([x3, x2]) #shape=(None, 128, 128, 64)
        x3 = self.conv5(x3) #shape=(None, 128, 128, 64)
        x3 = self.concat2_5([x1, x3]) #shape=(None, 128, 128, 128)
        x1 = self.maxpool5(x3) #shape=(None, 64, 64, 128)

        x1 = self.conv6(x1) #shape=(None, 64, 64, 128)
        _, x2 = tf.split(x1, 2, axis=-1) #shape=(None, 64, 64, 64)
        x2 = self.conv7(x2) #shape=(None, 64, 64, 64)
        x3 = self.conv8(x2) #shape=(None, 64, 64, 64)
        x3 = self.concat7_8([x3, x2]) #shape=(None, 64, 64, 128)
        x3 = self.conv9(x3) #shape=(None, 64, 64, 128)
        x3 = self.concat6_9([x1, x3]) #shape=(None, 64, 64, 256)
        
        x1 = self.maxpool9(x3) #shape=(None, 32, 32, 256)
        

        x1 = self.conv10(x1) #shape=(None, 32, 32, 256)
        _, x2 = tf.split(x1, 2, axis=-1) #shape=(None, 32, 32, 128)
        x2 = self.conv11(x2) #shape=(None, 32, 32, 128)
        
        x3 = self.conv12(x2) #(None, 32, 32, 128)
        
        x3 = self.concat11_12([x3, x2]) #(None, 32, 32, 256)
        
        route1 = self.conv13(x3) #(None, 32, 32, 256)
       
        x3 = self.concat10_13([x1, route1]) #(None, 32, 32, 512)
        
        x1 = self.maxpool13(x3) #(None, 16, 16, 512)

        x1 = self.conv14(x1) #(None, 16, 16, 512)
        
        x1 = self.spp(x1) #(None, 16, 16, 2048)

        x1 = self.conv15(x1) #(None, 16, 16, 512)

        x1 = self.conv16(x1) #shape=(None, 16, 16, 1024)
        
        route2 = self.conv17(x1) #shape=(None, 16, 16, 512)
    
        return route1, route2 #(None, 32, 32, 256),(None, 16, 16, 512)
