import tensorflow as tf
from tensorflow.python.eager import context
print(tf.version.GIT_VERSION, tf.version.VERSION)


@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

context.enable_run_metadata()

simple_nn_layer(x, y)

run_metadata = context.export_run_metadata()
context.disable_run_metadata()

print("Step stats: ", run_metadata.step_stats)
