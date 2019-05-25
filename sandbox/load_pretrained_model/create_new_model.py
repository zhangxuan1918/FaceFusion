from project_code.models.networks_3dmm import Face3DMM

face_model = Face3DMM()
face_model.build(input_shape=(None, 224, 224, 3))
print(face_model.summary())

for var in face_model.trainable_variables:
    print(var.name, var.shape)