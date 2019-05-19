from project_code.training.networks_3dmm import Resnet50, Face3DMM

face_model = Face3DMM()
face_model.build(input_shape=(None, 224, 224, 3))
# print(res50.summary())

# print(res50.trainable_variables[0])
for var in face_model.trainable_variables:
    print(var.name, var.shape)