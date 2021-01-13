from tensorflow.keras import callbacks, optimizers
import kerastuner as kt
from yolov4.tf import SaveWeightsCallback, YOLOv4
import datetime
import time

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


yolo = YOLOv4(tiny=True)
yolo.classes = "./custom.names"
yolo.input_size = (512, 512)
yolo.batch_size = 16

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

def model_builder(hp: kt.HyperParameters):
    hp_activation = hp.Choice("activation", values=["mish", "leaky", "relu"], default="leaky")

    hp_learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
    hp_loss_iou_type = hp.Choice("loss_iou_type", values=["iou", "giou", "ciou"], default="ciou")

    yolo.make_model(activation1=hp_activation)

    optimizer = optimizers.Adam(learning_rate=hp_learning_rate)
    yolo.compile(
        optimizer=optimizer,
        loss_iou_type=hp_loss_iou_type,
        loss_verbose=0,
    )

    return yolo.model


epochs = 20

log_dir = f"./logs/{now}"
_callbacks = [
    # callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    ),
    # SaveWeightsCallback(
    #     yolo=yolo, dir_path=f"./trained/{now}",
    #     weights_type="yolo", epoch_per_save=10
    # ),
    callbacks.EarlyStopping(monitor="loss", patience=1)
]

tuner = kt.Hyperband(
    model_builder,
    objective="loss",
    max_epochs=20,
    directory=log_dir
)

tuner.search(
    train_data_set,
    epochs=10,
    steps_per_epoch=200,
    callbacks=_callbacks,
    validation_data=val_data_set,
    validation_steps=50,
    validation_freq=1,
)

tuner.search_space_summary(extended=True)
