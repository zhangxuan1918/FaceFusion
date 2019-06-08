from tensorflow.python import keras

pretrained_model = keras.models.load_model('G:\PycharmProjects\FaceFusion\project_code\data\\face_vgg_v2\weights.h5')
print(pretrained_model.summary())

for layer in pretrained_model.layers:
    print(layer.name)
    if len(layer.weights) > 0:
        for weights in layer.weights:
            print('\t', weights.name, weights.value().shape)
