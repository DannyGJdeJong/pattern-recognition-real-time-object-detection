from tensorflow.keras import callbacks, optimizers
from yolov4.tf import SaveWeightsCallback, YOLOv4
import datetime
import time

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

yolo = YOLOv4(tiny=True)
yolo.classes = "./custom.names"
yolo.input_size = (512, 512)
yolo.batch_size = 16

yolo.make_model(activation1="mish")
# yolo.load_weights(
#     "./trained/yolov4-tiny-10.weights",
#     weights_type="yolo"
# )

train_data_set = yolo.load_dataset(
    "./custom_train.txt",
    image_path_prefix="./data",
    label_smoothing=0.05
)
val_data_set = yolo.load_dataset(
    "./custom_val.txt",
    image_path_prefix="./data",
    training=False
)

epochs = 20
lr = 1e-3

optimizer = optimizers.Adam(learning_rate=lr)
yolo.compile(optimizer=optimizer, loss_iou_type="ciou")

def lr_scheduler(epoch):
    if epoch < int(epochs * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1
    return lr * 0.01

log_dir = f"./logs/{now}"
_callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        profile_batch=(10,20)
    ),
    SaveWeightsCallback(
        yolo=yolo, dir_path=f"./trained/{now}",
        weights_type="yolo", epoch_per_save=10
    ),
]

yolo.fit(
    train_data_set,
    epochs=epochs,
    steps_per_epoch=100,
    callbacks=_callbacks,
    validation_data=val_data_set,
    validation_steps=50,
    validation_freq=5,
)
