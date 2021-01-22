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
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, backend, layers, Model

class YOLOv3HeadTiny(Model):
    def __init__(self, anchors, num_classes, xysclaes):
        super(YOLOv3HeadTiny, self).__init__(name="YOLOv3HeadTiny")
        self.a_half = []
        self.anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
        self.grid_coord = []
        self.grid_size = None
        self.image_width = None
        self.num_classes = num_classes
        self.scales = xysclaes

    def build(self, input_shape):
        # input_shape == None, g_height, g_width,
        #                      (xywh + conf + num_classes) * (# of anchors)

        # g_width, g_height
        _size = [(shape[2], shape[1]) for shape in input_shape] #[(32, 32), (16, 16)]

        for i in range(2):
            xy_grid = np.meshgrid(
                np.arange(_size[i][0]), np.arange(_size[i][1])
            )
            xy_grid = np.stack(xy_grid, axis=-1)
            xy_grid = xy_grid[np.newaxis, ...]
            self.grid_coord.append(
                tf.convert_to_tensor(xy_grid, dtype=tf.float32)
            )

        self.grid_size = tf.convert_to_tensor(_size, dtype=tf.float32) # tf.Tensor([32. 32.],[16. 16.]], shape=(2, 2)
        
        self.image_width = tf.convert_to_tensor( _size[0][0] * 16.0, dtype=tf.float32) # tf.Tensor(512.0, shape=()

    def call(self, x):
        raw_m, raw_l = x #(None, 32, 32, 48), (None, 16, 16, 48)

        sig_m = activations.sigmoid(raw_m) #(None, 32, 32, 48)
        sig_l = activations.sigmoid(raw_l) #(None, 16, 16, 48)
        
        # Dim(batch, g_height, g_width, 5 + num_classes)
        sig_m = tf.split(sig_m, 3, axis=-1) #(None, 32, 32, 16) x3
        raw_m = tf.split(raw_m, 3, axis=-1) #(None, 32, 32, 16) x3
        sig_l = tf.split(sig_l, 3, axis=-1) #(None, 16, 16, 16) x3
        raw_l = tf.split(raw_l, 3, axis=-1) #(None, 16, 16, 16) x3
        
        for i in range(3):
            txty_m, a_m, conf_prob_m = tf.split(sig_m[i], (2, 2, -1), axis=-1) #(None, 32, 32, 2),(None, 32, 32, 12) 

            _, twth_m, _ = tf.split(raw_m[i], (2, 2, -1), axis=-1) #(None, 32, 32, 2)
            
            txty_m = (txty_m - 0.5) * self.scales[0] + 0.5 #(None, 32, 32, 2)
            
            bxby_m = (txty_m + self.grid_coord[0]) / self.grid_size[0] #(None, 32, 32, 2)
            
            bwbh_m = (self.anchors[0][i] / self.image_width) * backend.exp(twth_m) #(None, 32, 32, 2)
            

            sig_m[i] = tf.concat([bxby_m, bwbh_m, conf_prob_m], axis=-1) #(None, 32, 32, 16)
            
            
            txty_l, _, conf_prob_l = tf.split(sig_l[i], (2, 2, -1), axis=-1) #(None, 16, 16, 2), (None, 16, 16, 12)
            _, twth_l, _ = tf.split(raw_l[i], (2, 2, -1), axis=-1) #(None, 16, 16, 2)
            txty_l = (txty_l - 0.5) * self.scales[1] + 0.5 #(None, 16, 16, 2)
            
            bxby_l = (txty_l + self.grid_coord[1]) / self.grid_size[1] #(None, 16, 16, 2)
            
            bwbh_l = (self.anchors[1][i] / self.image_width) * backend.exp(twth_l) #(None, 16, 16, 2)
            
            sig_l[i] = tf.concat([bxby_l, bwbh_l, conf_prob_l], axis=-1) #(None, 16, 16, 16)
            

        # Dim(batch, g_height, g_width, 3 * (5 + num_classes))
        pred_m = tf.concat(sig_m, axis=-1) #(None, 32, 32, 48)
        
        pred_l = tf.concat(sig_l, axis=-1) #(None, 16, 16, 48)

        return pred_m, pred_l #(None, 32, 32, 48), (None, 16, 16, 48)
