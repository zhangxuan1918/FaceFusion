from project_code.models.resnet50 import Resnet50

resnet50 = Resnet50()
resnet50.build(input_shape=(None, 224, 224, 3))
print(resnet50.summary())

# for var in face_model.trainable_variables:
#     print(var.name, var.shape)