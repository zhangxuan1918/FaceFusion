from tensorflow.python import keras
from tensorflow.python.keras import Model

face_vgg2 = keras.models.load_model('/opt/data/face_fuse/face_vgg_v2/weights.h5')

x = face_vgg2.layers[-2].output
model = Model(inputs=face_vgg2.input, outputs=x)
print(model.summary())


# model.save_weights('/opt/project/project_code/data/pretrained_model/20190718/weights.h5')