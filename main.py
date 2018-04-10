import os.path
from shutil import rmtree
import sys
import time
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    
    tensor_names = [
        'image_input:0',
        'keep_prob:0',
        'layer3_out:0',
        'layer4_out:0',
        'layer7_out:0'
    ]
    return (graph.get_tensor_by_name(n) for n in tensor_names)
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    encoder7_conv1x1 = tf.layers.conv2d(
        vgg_layer7_out, num_classes,
        1, strides=(1, 1), padding='same',
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
        name='encoder7_conv1x1')

    upsampled_for_decoder4 = tf.layers.conv2d_transpose(
        encoder7_conv1x1, num_classes,
        4, strides=(2, 2), padding='same', # upsample spatial dimensions by 2x
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

    encoder4_conv1x1 = tf.layers.conv2d(
        vgg_layer4_out, num_classes,
        1, strides=(1, 1), padding='same',
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
        )

    decoder4 = tf.add(upsampled_for_decoder4, encoder4_conv1x1, name='decoder4')

    upsampled_for_decoder3 = tf.layers.conv2d_transpose(
        decoder4, num_classes,
        4, strides=(2, 2), padding='same', # upsample spatial dimensions by 2x
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

    encoder3_conv1x1 = tf.layers.conv2d(
        vgg_layer3_out, num_classes,
        1, strides=(1, 1), padding='same',
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
        )

    decoder3 = tf.add(upsampled_for_decoder3, encoder3_conv1x1, name='decoder3')

    decoder1 = tf.layers.conv2d_transpose(
        decoder3, num_classes,
        16, strides=(8, 8), padding='same', # upsample spatial dimensions by 8x
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

    return decoder1
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, loss)
    """
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')

    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))

    reguralization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    loss = cross_entropy_loss + 0.001 * sum(reguralization_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    return logits, train_op, loss, accuracy
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn,
             train_op, loss, accuracy,
             input_image, correct_label,
             keep_prob, learning_rate,
             more_tensors=None, saver=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())

    learning_rates = [2e-4, 1e-4, 3e-5]
    learning_rate_cutoff_losses = [0.12, 0.09]

    saver_dir_parent = f"saved_sessions/{time.strftime('%Y-%m-%dT%H-%M-%S')}"
    saver_dir_best = ''
    best_loss = sys.float_info.max

    for epoch_ind in range(epochs):
        epoch_num = epoch_ind + 1

        loss_value = sys.float_info.max

        for train_images, label_images in get_batches_fn(batch_size):
            _, loss_value, accuracy_value = sess.run([
                    train_op,
                    loss,
                    accuracy
                ], {
                    input_image: train_images,
                    correct_label: label_images,
                    keep_prob: 0.5,
                    learning_rate: learning_rates[0]
                })

        if len(learning_rate_cutoff_losses) > 0 and loss_value < learning_rate_cutoff_losses[0]:
            learning_rate_cutoff_losses = learning_rate_cutoff_losses[1:]
            learning_rates = learning_rates[1:]

        if saver is not None and loss_value < best_loss:
            saver_dir = f"{saver_dir_parent}/epoch{epoch_num}"
            saver_path = f"{saver_dir}/model"
            os.makedirs(saver_dir)
            saver.save(sess, saver_path)

            if saver_dir_best != '':
                rmtree(saver_dir_best)
            saver_dir_best = saver_dir
            best_loss = loss_value
        print(f"after {epoch_num} epochs, loss was {loss_value}, accuracy {accuracy_value}")
tests.test_train_nn(train_nn)


def display_tensor_shapes(
    sess,
    tensor_input, tensor_keep_prob, tensor_layer3, tensor_layer4, tensor_layer7,
    tensor_output,
    tensor_correct_label, tensor_learning_rate,
    tensor_logits,
    get_batches_fn):
    """for debugging"""

    graph = tf.get_default_graph()

    printed_tensors = [
        tensor_input, tensor_layer3, tensor_layer4, tensor_layer7,
    ] + [
        graph.get_tensor_by_name(tensor_name) for tensor_name in [
            'encoder7_conv1x1/BiasAdd:0',
            'decoder4:0',
            'decoder3:0'
        ]
    ] + [
        tensor_output,
        tensor_correct_label,
        tensor_logits
    ]

    tensors_printer = tf.Print(
        tensor_output,
        [tf.shape(tensor) for tensor in printed_tensors],
        summarize=99)

    temp_gen = get_batches_fn(5)
    temp_train_images, temp_label_images = next(temp_gen)
    print(temp_train_images.shape, temp_label_images.shape)

    sess.run(tf.global_variables_initializer())

    sess.run(
        tensors_printer, {
            tensor_input: temp_train_images,
            tensor_correct_label: temp_label_images,
            tensor_keep_prob: 99,  
            tensor_learning_rate: 99
        })


def run():
    num_classes = 3
    image_shape = (160, 576)
    data_dir = './data'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    # (Did not do this.)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # (Did not do this.)

        # Build NN using load_vgg, layers, and optimize function
        tensor_input, tensor_keep_prob, tensor_layer3, tensor_layer4, tensor_layer7 = load_vgg(sess, vgg_path)
        tensor_output = layers(tensor_layer3, tensor_layer4, tensor_layer7, num_classes)

        tensor_correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        tensor_learning_rate = tf.placeholder(tf.float32)
        
        tensor_logits, train_op, tensor_loss, tensor_accuracy = \
            optimize(tensor_output, tensor_correct_label, tensor_learning_rate, num_classes)

        # display_tensor_shapes(
        #     sess,
        #     tensor_input, tensor_keep_prob, tensor_layer3, tensor_layer4, tensor_layer7,
        #     tensor_output,
        #     tensor_correct_label, tensor_learning_rate,
        #     tensor_logits,
        #     get_batches_fn)

        saver = tf.train.Saver()
        tf.add_to_collection('image_input', tensor_input)
        tf.add_to_collection('keep_prob', tensor_keep_prob)
        tf.add_to_collection('logits', tensor_logits)
        
        # Train NN using the train_nn function
        train_nn(
            sess, 10, 10, get_batches_fn,
            train_op, tensor_loss, tensor_accuracy,
            tensor_input, tensor_correct_label,
            tensor_keep_prob, tensor_learning_rate,
            saver=saver)


if __name__ == '__main__':
    run()
