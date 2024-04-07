import numpy as np

def one_hot_encode(label, label_value):
    sematic_map = []
    print(label.dtype)
    print(label_value.dtype)

    for pixel in label_value:
        equality = np.equal(label, pixel)
        class_map = np.all(equality, axis=-1)
        sematic_map.append(class_map)
    sematic_map = np.stack(sematic_map, axis=-1)

    return sematic_map

