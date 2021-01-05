from collections import defaultdict

INSTANCES_PATH = "_annotations.csv"
NAMES_PATH = "custom.names"
OUTPUT_FILE_PATH = "custom_train.txt"

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

with open(OUTPUT_FILE_PATH, "w") as f:
    for filename in files.keys():
        f.write(filename)
        for detected_object in files[filename]:
            f.write(" " + ','.join(map(str, detected_object)))
        f.write('\n')
