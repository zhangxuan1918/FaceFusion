from project_code.models.resnet50 import Resnet50

resnet50 = Resnet50()
resnet50.build(input_shape=(None, 224, 224, 3))
print(resnet50.summary())

# print(resnet50.compute_output_shape(input_shape=(None, 224, 224, 3)))
