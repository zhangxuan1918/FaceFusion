import tensorflow as tf

from project_code.models.resnet18 import Resnet18
from project_code.models.resnet50 import Resnet50


def export_model_pb_file(backbone, resolution, output_size, checkpoint_dir, model_export_dir):
    if backbone == 'resnet50':
        model = Resnet50(image_size=resolution, num_output=output_size)
    elif backbone == 'resnet18':
        model = Resnet18(image_size=resolution, num_output=output_size)
    else:
        raise ValueError('`backbone` not supported')

    checkpoint = tf.train.Checkpoint(model=model)

    # Restores the model from latest checkpoint.
    latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    assert latest_checkpoint_file
    print('Checkpoint file %s found and restoring from checkpoint' % latest_checkpoint_file)
    checkpoint.restore(latest_checkpoint_file).assert_existing_objects_matched()

    model.save(model_export_dir, include_optimizer=False, save_format='tf')


if __name__ == '__main__':
    backbone = 'resnet18'
    resolution = 224
    output_size = 290
    checkpoint_dir = '/opt/data/face-fuse/model/20200525/supervised/'
    model_export_dir = '/opt/data/face-fuse/model/20200525/supervised-exported/'
    export_model_pb_file(backbone, resolution, output_size, checkpoint_dir, model_export_dir)