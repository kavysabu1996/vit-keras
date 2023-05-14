from vit_keras import vit, utils
import numpy as np
from pathlib import Path

image_size = 384
classes = utils.get_imagenet_classes()
model = vit.vit_b16(
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)

model.summary()
model.save("vit_b16_withoutserialization.h5")
for variable in model.variables:
    path_list = variable.name.replace(":0","").split("/")
    dir_path = "weights/"+ ("/").join(path_list[:-1])
    file_name = path_list[-1]
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    np.save(dir_path+"/"+file_name+".npy",variable.numpy())