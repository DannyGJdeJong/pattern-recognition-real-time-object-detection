from tensorflow.keras import callbacks, optimizers
import kerastuner as kt
from yolov4.tf import SaveWeightsCallback, YOLOv4
import datetime
import time

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

batch_size = 32
input_size = (512,512)
train_size = 12577
val_size = 662

yolo = YOLOv4(tiny=True)
yolo.classes = "./custom.names"
yolo.input_size = input_size
yolo.batch_size = batch_size

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


class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)



epochs = 400
schedule = SGDRScheduler(min_lr=1e-5,
                        max_lr=1e-2,
                        steps_per_epoch= int(np.ceil(epochs/batch_size)),)

log_dir = f"./logs/{now}"
_callbacks = [
    schedule,
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    ),
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
    steps_per_epoch=int(np.ceil(train_size/batch_size)),
    callbacks=_callbacks,
    validation_data=val_data_set,
    validation_steps=int(np.ceil(val_size/batch_size)),
    validation_freq=1,
)

tuner.search_space_summary(extended=True)
