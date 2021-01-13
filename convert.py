from collections import defaultdict
import random

INSTANCES_PATH = "_annotations.csv"
NAMES_PATH = "custom.names"
TRAIN_OUTPUT_FILE_PATH = "custom_train.txt"
VAL_OUTPUT_FILE_PATH = "custom_val.txt"

validation_ratio = 0.05

files = defaultdict(lambda: [])
classes = defaultdict(lambda: len(classes))

with open(INSTANCES_PATH) as f:
    # Skip header
    f.readline()

    for line in f:
        if len(line) > 5:
            filename, width, height, class_name, xmin, ymin, xmax, ymax = line.split(',')
            width = int(width)
            height = int(height)
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)
            class_num = classes[class_name]
            center_x = (xmin + xmax) / 2 / width
            center_y = (ymin + ymax) / 2 / height
            width = (xmax - xmin) / width
            height = (ymax - ymin) / height
            files[filename].append([class_num, center_x, center_y, width, height])

with open(NAMES_PATH, "w") as f:
    for class_name in classes.keys():
        f.write(class_name + "\n")

filenames = list(files.keys())
random.shuffle(filenames)

cutoff = round(len(filenames) * validation_ratio)

val_filenames = filenames[:cutoff]
train_filenames = filenames[cutoff:]

with open(VAL_OUTPUT_FILE_PATH, "w") as f:
    for filename in val_filenames:
        f.write(filename)
        for detected_object in files[filename]:
            f.write(" " + ','.join(map(str, detected_object)))
        f.write('\n')

with open(TRAIN_OUTPUT_FILE_PATH, "w") as f:
    for filename in train_filenames:
        f.write(filename)
        for detected_object in files[filename]:
            f.write(" " + ','.join(map(str, detected_object)))
        f.write('\n')
