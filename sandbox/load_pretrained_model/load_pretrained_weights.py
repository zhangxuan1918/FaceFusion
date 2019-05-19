import h5py


def get_weights_from_hdf5(filepath):
    """ Loads the weights from a saved Keras model into numpy arrays.
        The weights are saved using Keras 2.0 so we don't need all the
        conversion functionality for handling old weights.
    """

    with h5py.File(filepath, mode='r') as f:
        layer_names = list(f['model_weights'].keys())
        layer_weights = []
        for k, l_name in enumerate(layer_names):
            g = f[l_name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name][:] for weight_name in weight_names]
            if len(weight_values):
                layer_weights.append([l_name, weight_names, weight_values])
        return layer_weights

layer_weights = get_weights_from_hdf5(filepath='G:\PycharmProjects\FaceFusion\project_code\data\face_vgg_v2\weights.h5')

for l_name, weight_name, weight_value in layer_weights:
    print(l_name, weight_name, weight_value.shape)