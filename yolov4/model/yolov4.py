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
from tensorflow.keras import Model

from .backbone import  CSPDarknet53Tiny
from .head import  YOLOv3HeadTiny
from .neck import  BiFPNTiny


class YOLOv4Tiny(Model):
    def __init__(
        self,
        anchors,
        num_classes: int,
        xyscales,
        activation: str = "leaky",
        kernel_regularizer=None,
    ):
        super(YOLOv4Tiny, self).__init__(name="YOLOv4Tiny")
        self.csp_darknet53_tiny = CSPDarknet53Tiny(
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.bifpn_tiny = BiFPNTiny(num_classes=num_classes,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.yolov3_head_tiny = YOLOv3HeadTiny(
            anchors=anchors, num_classes=num_classes, xysclaes=xyscales
        )

    def call(self, x):
        x = self.csp_darknet53_tiny(x)
        x = self.bifpn_tiny(x)
        x = self.yolov3_head_tiny(x)
        return x
