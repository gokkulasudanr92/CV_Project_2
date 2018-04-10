# CNN Face Detection Neural Network

import numpy as np
import tensorflow as tf
import cv2
import os

tf.logging.set_verbosity(tf.logging.INFO)
FACES_TRAINING_SRC_LOCATION = "G:/Project/training/faces/"
FACES_TEST_SRC_LOCATION = "G:/Project/test/faces/"
NON_FACES_TRAINING_SRC_LOCATION = "G:/Project/training/non_faces/"
NON_FACES_TEST_SRC_LOCATION = "G:/Project/test/non_faces/"

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Input images are 60x60 pixels, and have 3 color channel
    input_layer = tf.reshape(features["x"], [-1, 60, 60, 3])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 60, 60, 3]
    # Output Tensor Shape: [batch_size, 60, 60, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 60, 60, 32]
    # Output Tensor Shape: [batch_size, 30, 30, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 30, 30, 32]
    # Output Tensor Shape: [batch_size, 30, 30, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 30, 30, 64]
    # Output Tensor Shape: [batch_size, 15, 15, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 15, 15, 64]
    # Output Tensor Shape: [batch_size, 15 * 15 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 15 * 15 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 15 * 15 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    ## One step is to run without regularization
    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    # logits = tf.layers.dense(inputs=dropout, units=2)
    logits = tf.layers.dense(inputs=dense, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + 1e1

    # Calculate the accuracy between the true labels, and our predictions
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('loss', loss)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    eval_metric_ops = {"accuracy": eval_accuracy}
    tf.summary.scalar('eval_accuracy', eval_accuracy)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def extract_data_images_from_folder(faces_folder_location, non_faces_folder_location, start_point, number_of_data_points):
    """ Load images and convert them as np array of data size X 60 * 60 * 3 """
    count = 0
    start = 0
    images_list = []
    for filename in os.listdir(faces_folder_location):
        if start < start_point:
            start += 1
            continue
        
        if count >= number_of_data_points:
            count += 1
            break
        img = cv2.imread(os.path.join(faces_folder_location, filename), flags = cv2.IMREAD_COLOR)
        if img is not None:
            img_reshaped = img.reshape((1, 10800))
            images_list.append(img_reshaped)
        count += 1

    count = 0
    start = 0
    for filename in os.listdir(non_faces_folder_location):
        if start < start_point:
            start += 1
            continue
         
        if count >= number_of_data_points:
            break
        img = cv2.imread(os.path.join(non_faces_folder_location, filename), flags = cv2.IMREAD_COLOR)
        if img is not None:
            img_reshaped = img.reshape((1, 10800))
            images_list.append(img_reshaped)
        count += 1

    data = np.vstack(images_list)
    data_marix = data.astype(np.float32)
    faces_labels = [1] * number_of_data_points
    non_faces_labels = [0] * number_of_data_points

    temp_labels = faces_labels
    temp_labels.extend(non_faces_labels)
    data_labels = np.asarray(temp_labels, dtype=np.int32)

    return data_marix, data_labels

def normalize_data_to_negative_one_to_positive_one(data_matrix):
    """ The input argument is a data matrix of np array """
    normalized_data_matrix = data_matrix
    mean_value = np.mean(normalized_data_matrix, axis=0)
    normalized_data_matrix -= mean_value
    normalized_data_matrix /= np.std(normalized_data_matrix, axis=0)

    return normalized_data_matrix

def main(unused_argv):
    # Load training and test data
    train_data, train_labels = extract_data_images_from_folder(FACES_TRAINING_SRC_LOCATION, NON_FACES_TRAINING_SRC_LOCATION, 0, 10000)
    normalized_train_data = normalize_data_to_negative_one_to_positive_one(train_data)

    eval_data, eval_labels = extract_data_images_from_folder(FACES_TEST_SRC_LOCATION, NON_FACES_TEST_SRC_LOCATION, 0, 1000)
    normalize_eval_data = normalize_data_to_negative_one_to_positive_one(eval_data)

    # Create CNN Estimator
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="../temp/final/classifier_cnn_model_SGD_1")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"accuracy": "Accuracy"}
    # _TENSORS_TO_LOG = dict((x, x) for x in ['accuracy',
    #                                     'loss'])
    # logging_hook = tf.train.LoggingTensorHook(tensors=_TENSORS_TO_LOG, every_n_iter=100)

    # Train the model with babysitting
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": normalized_train_data},
        y=train_labels,
        batch_size=24,
        num_epochs=1,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=2001,
        hooks=None)

    exit(0) 

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": normalize_eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    exit(0)

if __name__ == "__main__":
  tf.app.run()