from yolov4.tf import YOLOv4

yolo = YOLOv4(tiny=True)

yolo.classes = "./custom.names"

yolo.make_model()
yolo.load_weights("./trained/yolov4-tiny-final.weights", weights_type="yolo")

dataset = yolo.load_dataset(
    "./custom_val.txt",
    training=False,
    image_path_prefix="./data",
)

yolo.save_dataset_for_mAP("./mAP", dataset)
