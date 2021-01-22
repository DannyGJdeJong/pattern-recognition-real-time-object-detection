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
from os import makedirs, path
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, layers, optimizers

from . import dataset, train, weights
from .train import SaveWeightsCallback
from ..common.base_class import BaseClass
from ..model import yolov4

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Call tf.config.experimental.set_memory_growth(GPU0, True)")


class YOLOv4(BaseClass):
    def __init__(self, tiny: bool = False, tpu: bool = False):
        """
        Default configuration
        """
        super(YOLOv4, self).__init__(tiny=tiny, tpu=tpu)

        self.batch_size = 32
        self._has_weights = False
        self.input_size = 608
        self.model = None

    def make_model(
        self,
        activation0: str = "mish",
        activation1: str = "leaky",
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
    ):
        # pylint: disable=missing-function-docstring
        self._has_weights = False
        backend.clear_session()

        # height, width, channels
        inputs = layers.Input([self.input_size[1], self.input_size[0], 3])
        if self.tiny:
            self.model = yolov4.YOLOv4Tiny(
                anchors=self.anchors,
                num_classes=len(self.classes),
                xyscales=self.xyscales,
                activation=activation1,
                kernel_regularizer=kernel_regularizer,
            )
        else:
            self.model = yolov4.YOLOv4(
                anchors=self.anchors,
                num_classes=len(self.classes),
                xyscales=self.xyscales,
                activation0=activation0,
                activation1=activation1,
                kernel_regularizer=kernel_regularizer,
            )
        self.model(inputs)

    def load_weights(self, weights_path: str, weights_type: str = "tf"):
        """
        Usage:
            yolo.load_weights("yolov4.weights", weights_type="yolo")
            yolo.load_weights("checkpoints")
        """
        if weights_type == "yolo":
            weights.load_weights(self.model, weights_path, tiny=self.tiny)
        elif weights_type == "tf":
            self.model.load_weights(weights_path)

        self._has_weights = True

    def save_weights(self, weights_path: str, weights_type: str = "tf"):
        """
        Usage:
            yolo.save_weights("yolov4.weights", weights_type="yolo")
            yolo.save_weights("checkpoints")
        """
        if weights_type == "yolo":
            weights.save_weights(self.model, weights_path, tiny=self.tiny)
        elif weights_type == "tf":
            self.model.save_weights(weights_path)

    def save_as_tflite(
        self,
        tflite_path,
        quantization=None,
        data_set=None,
        num_calibration_steps: int = 100,
    ):
        """
        Save model and weights as tflite

        Usage:
            yolo.save_as_tflite("yolov4.tflite")
            yolo.save_as_tflite("yolov4-float16.tflite", "float16")
            yolo.save_as_tflite("yolov4-int.tflite", "int", data_set)
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        _supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        def representative_dataset_gen():
            count = 0
            while True:
                images, _ = next(data_set)
                for i in range(len(images)):
                    yield [tf.cast(images[i : i + 1, ...], tf.float32)]
                    count += 1
                    if count >= num_calibration_steps:
                        return

        if quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int":
            converter.representative_dataset = representative_dataset_gen
        elif quantization == "full_int8":
            converter.experimental_new_converter = False
            converter.representative_dataset = representative_dataset_gen
            _supported_ops += [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        elif quantization:
            raise ValueError(
                "YOLOv4: {} is not a valid option".format(quantization)
            )

        converter.target_spec.supported_ops = _supported_ops

        tflite_model = converter.convert()
        with tf.io.gfile.GFile(tflite_path, "wb") as fd:
            fd.write(tflite_model)

    #############
    # Inference #
    #############

    @tf.function
    def _predict(self, x):
        # s_pred, m_pred, l_pred
        # x_pred == Dim(1, output_size, output_size, anchors, (bbox))
        candidates = self.model(x, training=False)
        _candidates = []
        for candidate in candidates:
            grid_size = candidate.shape[1:3]
            _candidates.append(
                tf.reshape(
                    candidate[0], shape=(1, grid_size[0] * grid_size[1] * 3, -1)
                )
            )
        return tf.concat(_candidates, axis=1)

    def predict(
        self,
        frame: np.ndarray,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.25,
    ):
        """
        Predict one frame

        @param frame: Dim(height, width, channels)

        @return pred_bboxes == Dim(-1, (x, y, w, h, class_id, probability))
        """
        # image_data == Dim(1, input_size[1], input_size[0], channels)
        image_data = self.resize_image(frame)
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        candidates = self._predict(image_data)

        # Select 0
        pred_bboxes = self.candidates_to_pred_bboxes(
            candidates[0].numpy(),
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        pred_bboxes = self.fit_pred_bboxes_to_original(pred_bboxes, frame.shape)
        return pred_bboxes

    ############
    # Training #
    ############

    def load_dataset(
        self,
        dataset_path,
        dataset_type="converted_coco",
        label_smoothing=0.1,
        image_path_prefix=None,
        training=True,
    ):
        return dataset.Dataset(
            anchors=self.anchors,
            batch_size=self.batch_size,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            data_augmentation=training,
            input_size=self.input_size,
            label_smoothing=label_smoothing,
            num_classes=len(self.classes),
            image_path_prefix=image_path_prefix,
            strides=self.strides,
            xyscales=self.xyscales,
        )

    def compile(
        self,
        loss_iou_type: str = "ciou",
        loss_verbose=1,
        optimizer=optimizers.Adam(learning_rate=1e-4),
        **kwargs
    ):
        # TODO: steps_per_execution tensorflow2.4.0-rc4
        self.model.compile(
            optimizer=optimizer,
            loss=train.YOLOv4Loss(
                batch_size=self.batch_size,
                iou_type=loss_iou_type,
                verbose=loss_verbose,
            ),
            **kwargs
        )

    def fit(
        self,
        data_set,
        epochs,
        verbose=2,
        callbacks=None,
        validation_data=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_freq=1,
        **kwargs
    ):
        self.model.fit(
            data_set,
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            **kwargs
        )

    def save_dataset_for_mAP(
        self, mAP_path, data_set, num_sample=None, images_optional=False
    ):
        """
        gt: name left top right bottom
        dr: name confidence left top right bottom

        @param `mAP_path`
        @parma `data_set`
        @param `num_sample`: Number of images for mAP. If `None`, all images in
                `data_set` are used.
        @parma `images_optional`: If `True`, images are copied to the
                `mAP_path`.
        """
        input_path = path.join(mAP_path, "input")

        if path.exists(input_path):
            shutil.rmtree(input_path)
        makedirs(input_path)

        gt_dir_path = path.join(input_path, "ground-truth")
        dr_dir_path = path.join(input_path, "detection-results")
        makedirs(gt_dir_path)
        makedirs(dr_dir_path)

        if images_optional:
            img_dir_path = path.join(input_path, "images-optional")
            makedirs(img_dir_path)

        max_dataset_size = len(data_set)

        if num_sample is None:
            num_sample = max_dataset_size

        for i in range(num_sample):
            # image_path, [[x, y, w, h, class_id], ...]
            _dataset = data_set.dataset[i % max_dataset_size].copy()

            if images_optional:
                image_path = path.join(img_dir_path, "image_{}.jpg".format(i))
                shutil.copy(_dataset[0], image_path)

            image = cv2.imread(_dataset[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape

            _dataset[1] = _dataset[1] * np.array(
                [width, height, width, height, 1]
            )

            # ground-truth
            with open(
                path.join(gt_dir_path, "image_{}.txt".format(i)),
                "w",
            ) as fd:
                for xywhc in _dataset[1]:
                    # name left top right bottom
                    class_name = self.classes[int(xywhc[4])]
                    left = int(xywhc[0] - xywhc[2] / 2)
                    top = int(xywhc[1] - xywhc[3] / 2)
                    right = int(xywhc[0] + xywhc[2] / 2)
                    bottom = int(xywhc[1] + xywhc[3] / 2)
                    fd.write(
                        "{} {} {} {} {}\n".format(
                            class_name, left, top, right, bottom
                        )
                    )

            pred_bboxes = self.predict(image)
            pred_bboxes = pred_bboxes * np.array(
                [width, height, width, height, 1, 1]
            )

            # detection-results
            with open(
                path.join(dr_dir_path, "image_{}.txt".format(i)),
                "w",
            ) as fd:
                for xywhcp in pred_bboxes:
                    # name confidence left top right bottom
                    class_name = self.classes[int(xywhcp[4])]
                    probability = xywhcp[5]
                    left = int(xywhcp[0] - xywhcp[2] / 2)
                    top = int(xywhcp[1] - xywhcp[3] / 2)
                    right = int(xywhcp[0] + xywhcp[2] / 2)
                    bottom = int(xywhcp[1] + xywhcp[3] / 2)
                    fd.write(
                        "{} {} {} {} {} {}\n".format(
                            class_name, probability, left, top, right, bottom
                        )
                    )
