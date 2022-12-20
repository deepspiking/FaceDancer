from networks.layers import AdaIN, AdaptiveAttention
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# To hide "WARNING:root:The given value for groups will be overwritten."
import logging
logging.getLogger().setLevel(logging.ERROR)

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

gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
    try:
        for i in gpu:
            tf.config.experimental.set_memory_growth(i, True)
    except RuntimeError as e:
        print(e)

def bgra2bgr(img):
	if len(img.shape) > 2 and img.shape[2] == 4:
		#convert the image from RGBA2RGB
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
	return img


arcface_model_path="../models/ArcFace/ArcFace-Res50.h5"

#source_img="assets/video_swap_ex/suhwan.png"
source_img="assets/src_paper.png"

arcface = load_model(arcface_model_path, compile=False)

# target and source images need to be properly cropeed and aligned
source = np.asarray(Image.open(source_img).resize((112, 112)))

source = bgra2bgr(source)


source_z = arcface(np.expand_dims(source / 255.0, axis=0))
source_z_np = source_z.numpy()

np.save('tmp_z.npy',source_z_np)
