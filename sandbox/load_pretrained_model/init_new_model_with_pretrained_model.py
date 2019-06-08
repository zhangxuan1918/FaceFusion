from tensorflow.python import keras

from project_code.models.networks_resnet50 import ConvBlock, IdentityBlock
from project_code.models.networks_3dmm import Face3DMM

face_model = Face3DMM()
face_model.build(input_shape=(None, 224, 224, 3))
pretrained_model = keras.models.load_model('G:/PycharmProjects/FaceFusion/project_code/data/face_vgg_v2/weights.h5')

for fm_layer in face_model.layers:
    if not isinstance(fm_layer, ConvBlock) and not isinstance(fm_layer, IdentityBlock):
        print(fm_layer.name)

        if len(fm_layer.weights) > 0:
            try:
                pm_layer = pretrained_model.get_layer(name=fm_layer.name)
            except:
                continue
            pm_layer_weights = {}
            for weights in pm_layer.weights:
                weights_name = weights.name
                pm_layer_weights[weights_name] = weights.value()

            for weights in fm_layer.weights:
                weights_name = weights.name
                print('\t', weights_name, weights.value().shape, pm_layer_weights[weights_name].shape)
                assert weights.value().shape == pm_layer_weights[weights_name].shape
            fm_layer.set_weights(pm_layer.get_weights())
    else:
        for fm_sublayer in fm_layer.layers:
            print(fm_sublayer.name)
            if len(fm_sublayer.weights) > 0:
                try:
                    pm_layer = pretrained_model.get_layer(name=fm_sublayer.name)
                except:
                    continue
                pm_layer_weights = {}
                for weights in pm_layer.weights:
                    weights_name = weights.name
                    pm_layer_weights[weights_name] = weights.value()

                for weights in fm_sublayer.weights:
                    weights_name = weights.name[len(fm_layer.name) + 1:]
                    print('\t', weights_name, weights.value().shape, pm_layer_weights[weights_name].shape)
                    assert weights.value().shape == pm_layer_weights[weights_name].shape

                fm_sublayer.set_weights(pm_layer.get_weights())

print(face_model.summary())
face_model.save_weights('G:/PycharmProjects/FaceFusion/project_code/data/pretrained_model/20190608/face_model_pretrained_face_vgg2.ckpt')