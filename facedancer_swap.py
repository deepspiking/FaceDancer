from networks.layers import AdaIN, AdaptiveAttention
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# To hide "WARNING:root:The given value for groups will be overwritten."
import logging
logging.getLogger().setLevel(logging.ERROR)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# To hide very long tensorflow log like:
# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 256, 256, 3  0           []                               
#
# Can be added directly to networks/layers.py
import tensorflow as tf
#tf.keras.utils.disable_interactive_logging()

# Add compile=False to hide
# "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually."

#gpu = tf.config.experimental.list_physical_devices('GPU')
#if gpu:
#    try:
#        for i in gpu:
#            tf.config.experimental.set_memory_growth(i, True)
#    except RuntimeError as e:
#        print(e)

def bgra2bgr(img):
	if len(img.shape) > 2 and img.shape[2] == 4:
		#convert the image from RGBA2RGB
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
	return img

face_dancer_model_path="../models/FaceDancer/FaceDancer_config_B.h5"

target_img="assets/target_paper.png"

model = load_model(face_dancer_model_path, compile=False, custom_objects={"AdaIN": AdaIN, "AdaptiveAttention": AdaptiveAttention, "InstanceNormalization": InstanceNormalization})

# target and source images need to be properly cropeed and aligned
target = np.asarray(Image.open(target_img).resize((256, 256)))

target = bgra2bgr(target)

source_z_np = np.load('tmp_z.npy')
source_z = tf.convert_to_tensor(source_z_np, dtype=tf.float32)

face_swap = model([np.expand_dims((target - 127.5) / 127.5, axis=0), source_z]).numpy()
face_swap = (face_swap[0] + 1) / 2
face_swap = np.clip(face_swap * 255, 0, 255).astype('uint8')

cv2.imwrite("./swapped_face.png", cv2.cvtColor(face_swap, cv2.COLOR_BGR2RGB))
