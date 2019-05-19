from project_code.training.networks_3dmm import ConvBlock

conv_block = ConvBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block=1, strides=(1, 1),
                       trainable=True)
conv_block.build(input_shape=(None, 214, 214, 3))
print(conv_block.summary())
