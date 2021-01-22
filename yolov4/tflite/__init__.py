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

try:
    import tflite_runtime.interpreter as tflite
    from tflite_runtime.interpreter import load_delegate
except ModuleNotFoundError:
    import tensorflow.lite as tflite

    load_delegate = tflite.experimental.load_delegate

from ..common.base_class import BaseClass


class YOLOv4(BaseClass):
    def __init__(self, tiny: bool = False, tpu: bool = False):
        """
        Default configuration
        """
        super(YOLOv4, self).__init__(tiny=tiny, tpu=tpu)
        self.grid_coord = []
        self.input_index = None
        self.interpreter = None
        self.output_index = None
        self.output_size = None

    def load_tflite(self, tflite_path: str) -> None:
        if self.tpu:
            self.interpreter = tflite.Interpreter(
                model_path=tflite_path,
                experimental_delegates=[load_delegate("libedgetpu.so.1")],
            )
        else:
            self.interpreter = tflite.Interpreter(model_path=tflite_path)

        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()[0]
        # width, height
        self.input_size = (input_details["shape"][2], input_details["shape"][1])
        self.input_index = input_details["index"]
        output_details = self.interpreter.get_output_details()
        self.output_index = [details["index"] for details in output_details]

    #############
    # Inference #
    #############

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
        if not self.tpu:
            image_data = image_data.astype(np.float32) / 255.0
        image_data = image_data[np.newaxis, ...]

        # s_pred, m_pred, l_pred
        # x_pred == Dim(1, g_height, g_width, anchors, (bbox))
        self.interpreter.set_tensor(self.input_index, image_data)
        self.interpreter.invoke()
        if not self.tpu:
            candidates = [
                self.interpreter.get_tensor(index)
                for index in self.output_index
            ]
        else:
            candidates = [
                self.interpreter.get_tensor(index).astype(np.float32) / 255.0
                for index in self.output_index
            ]
        _candidates = []
        for candidate in candidates:
            grid_size = candidate.shape[1:3]
            _candidates.append(
                np.reshape(
                    candidate[0], (1, grid_size[0] * grid_size[1] * 3, -1)
                )
            )
        candidates = np.concatenate(_candidates, axis=1)

        pred_bboxes = self.candidates_to_pred_bboxes(
            candidates[0],
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        pred_bboxes = self.fit_pred_bboxes_to_original(pred_bboxes, frame.shape)
        return pred_bboxes
