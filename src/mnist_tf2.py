import tensorflow as tf
import argparse
import os
import numpy as np
import json
from typing import Tuple

def model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tf.keras.models.Sequential:
    """Generate a simple model.

    Args:
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Testing data.
        y_test (np.ndarray): Testing labels.

    Returns:
        tf.keras.models.Sequential: Trained model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test)

    return model

def _load_training_data(base_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST training data.

    Args:
        base_dir (str): Base directory containing training data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Training data and labels.
    """
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return x_train, y_train

def _load_testing_data(base_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST testing data.

    Args:
        base_dir (str): Base directory containing testing data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Testing data and labels.
    """
    x_test = np.load(os.path.join(base_dir, 'eval_data.npy'))
    y_test = np.load(os.path.join(base_dir, 'eval_labels.npy'))
    return x_test, y_test

def _parse_args() -> Tuple[argparse.Namespace, list]:
    """Parse command-line arguments.

    Returns:
        Tuple[argparse.Namespace, list]: Parsed arguments and unknown arguments.
    """
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is an S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    mnist_classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # Save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format
        # To export the model as h5 format use model.save('my_model.h5')
        mnist_classifier.save(os.path.join(args.sm_model_dir, '000000001'))
