import os
import csv
import random

import cv2
import discord_notify as dn
import numpy as np
import tensorflow as tf

from auth import WEBHOOK_URL
from callbacks import TrainingNotifier
from constants import *
from model_type import ModelType

IMG_CHANNELS = 3
MODEL_TYPE = ModelType.XCEPTION
PLOT_LABELS = False
LOAD_ALL_IMAGES = False

if PLOT_LABELS:
    import matplotlib.pyplot as plt

notifier = dn.Notifier(
    WEBHOOK_URL
)


def get_base_image_name(image_name):
    """
    Returns the image_name not including an identifier sequence
    (generated from image augmentation)
    """
    # Index where suffix sequence starts
    suffix_start = image_name.find(IMAGE_IDENTIFIER_SEQUENCE)

    base_image_name = image_name
    if suffix_start != -1:
        # Index where extension sequence starts (.jpg, .jpeg, or .png)
        extension_start = image_name.lower().find(".jp")
        if extension_start == -1:
            extension_start = image_name.lower().find(".png")
        if extension_start == -1:
            raise Exception("Invalid extension in image name: " + image_name)

        # imagename(--===--)(4).jpg becomes imagename.jpg
        base_image_name = image_name[:suffix_start]  # imagename
        extension = image_name[extension_start:]  # .jpg
        base_image_name += extension  # imagename.jpg

    return base_image_name


def get_processed_image(image_path):
    """
    Returns the image at image_path, processed to only contain one
    channel
    """
    image = cv2.imread(image_path)
    if MODEL_TYPE is ModelType.XCEPTION:
        image = tf.keras.applications.xception.preprocess_input(image)
    elif MODEL_TYPE is ModelType.INCEPTION_V3:
        image = tf.keras.applications.inception_v3.preprocess_input(image)
    elif MODEL_TYPE is ModelType.CUSTOM:
        image = image.astype(np.float)
        image /= 255
    else:
        raise Exception("Invalid MODEL_TYPE provided")
    return image


def load_images(image_name_to_label):
    """
    Returns (images, labels), where each image is from a file name
    present in the image_name_to_label dictionary
    """
    images = []
    labels = []

    image_names = os.listdir(DEFAULT_IMG_PATH_EDITED)

    # Remove directories
    image_names.remove("COVID-19")
    image_names.remove("Normal")
    image_names.remove("ViralPneumonia")

    # Load images from specific image directories (COVID-19, normal, viral pneumonia)
    def load_directory(directory):
        notifier.send("        Loading from directory: " + directory + "...")
        directory_path = DEFAULT_IMG_PATH_EDITED + os.sep + directory
        directory_image_names = os.listdir(directory_path)
        for i, image_name in enumerate(directory_image_names):
            base_image_name = get_base_image_name(image_name)
            query_name = directory + "/" + base_image_name
            query_name = query_name.lower().replace(" ", "")
            if query_name in image_name_to_label:
                print(f"            {i / len(directory_image_names) * 100}% - [{image_name}]")
                image_path = directory_path + os.sep + image_name
                image = get_processed_image(image_path)
                images.append(image)
                labels.append(image_name_to_label[query_name])
    load_directory("COVID-19")
    load_directory("Normal")
    load_directory("ViralPneumonia")

    # Load images from default directory
    if LOAD_ALL_IMAGES:
        notifier.send("        Loading from directory: default...")
        for i, image_name in enumerate(image_names):
            base_image_name = get_base_image_name(image_name)
            if base_image_name in image_name_to_label:
                print(f"            {i / len(image_names) * 100}% - [{image_name}]")
                image_path = DEFAULT_IMG_PATH_EDITED + os.sep + image_name
                image = get_processed_image(image_path)
                images.append(image)
                labels.append(image_name_to_label[base_image_name])

    return images, labels


def shuffle_data_pair(list1, list2):
    """
    Returns (list1, list2), both shuffled in the same manner
    """
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    list1, list2 = zip(*combined)
    return list(list1), list(list2)


def split_data(images, labels):
    """
    Split data into training (80%), validation (10%), and testing (10%)
    datasets

    Returns (images_train, images_validate, images_test, labels_train,
    labels_validate, labels_test)

    Assumes that num_covid_points <= num_normal_points and num_virus_points
    """
    images, labels = shuffle_data_pair(images, labels)

    num_covid_points = sum(map(lambda label: label == 0, labels))

    # Calculate split
    num_test = int(num_covid_points * 0.1)
    num_covid_train = num_covid_points - num_test * 2
    num_other_train = int(num_covid_train * 1.1)

    # (train, validate, test) points added
    num_points_added = [
        [0, 0, 0],  # COVID-19
        [0, 0, 0],  # Viral pneumonia
        [0, 0, 0]   # Normal
    ]

    # Datasets
    images_train = []
    labels_train = []
    images_validate = []
    labels_validate = []
    images_test = []
    labels_test = []

    # Add images and labels to datasets
    notifier.send("        Adding images and labels to dataset...")
    for i, label in enumerate(labels):
        print(f"            Point: {i} / {len(labels)}")
        completed_labels = [False, False, False]  # Enough of label added
        if all(completed_labels):
            break
        for j in range(3):  # 0: COVID-19, 1: Viral pneumonia, 2: Normal
            if completed_labels[j]:
                continue
            if label == j:
                # Add training data
                can_add_training = False
                if j == 0:  # COVID-19
                    if num_points_added[j][0] < num_covid_train:
                        can_add_training = True
                        num_points_added[j][0] += 1
                elif num_points_added[j][0] < num_other_train:  # Not COVID-19
                    can_add_training = True
                    num_points_added[j][0] += 1
                if can_add_training:
                    images_train.append(images[i])
                    labels_train.append(labels[i])
                    break

                # Add validation data
                if num_points_added[j][1] < num_test:
                    num_points_added[j][1] += 1
                    images_validate.append(images[i])
                    labels_validate.append(labels[i])
                    break

                # Add testing data
                if num_points_added[j][2] < num_test:
                    num_points_added[j][2] += 1
                    images_test.append(images[i])
                    labels_test.append(labels[i])
                    break

                # Point couldn't be added anywhere: label is complete
                completed_labels[j] = True
                break

    # Shuffle all data
    notifier.send("        Shuffling data...")
    images_train, labels_train = shuffle_data_pair(
        images_train, labels_train
    )
    images_validate, labels_validate = shuffle_data_pair(
        images_validate, labels_validate
    )
    images_test, labels_test = shuffle_data_pair(
        images_test, labels_test
    )

    if PLOT_LABELS:
        # Plot data frequencies
        plt.hist(labels, bins=3)
        plt.title("Labels")

        plt.hist(labels_train, bins=3)
        plt.title("Train Labels")

        plt.hist(labels_validate, bins=3)
        plt.title("Validate Labels")

        plt.hist(labels_test, bins=3)
        plt.title("Test Labels")

        plt.show()

    # Make labels categorical
    notifier.send("        Making labels categorical: train...")
    labels_train = tf.keras.utils.to_categorical(labels_train)
    notifier.send("        Making labels categorical: validate...")
    labels_validate = tf.keras.utils.to_categorical(labels_validate)
    notifier.send("        Making labels categorical: test...")
    labels_test = tf.keras.utils.to_categorical(labels_test)

    notifier.send("        Converting data to NumPy arrays...")
    return \
        np.array(images_train), np.array(images_validate), np.array(images_test), \
        np.array(labels_train), np.array(labels_validate), np.array(labels_test)


def load_data():
    """
    Returns (images_train, images_validate, images_test, labels_train,
    labels_validate, labels_test), where labels is a list of labels indicating
    COVID-19 (0), viral pneumonia (1), or normal (2)
    """
    # Dictionary mapping image names to labels
    image_name_to_label = dict()

    # Store labels associated with image names
    notifier.send("    Reading metadata...")
    with open("data/metadata.csv") as file:  # Original dataset
        # Use images for normal, virus (unknown type), COVID-19, SARS
        metadata_contents = csv.DictReader(file)
        for row in metadata_contents:
            if row["Label"].lower() == "normal":
                label = 2
            elif row["Label_2_Virus_category"].lower() == "covid-19":
                label = 0
            elif row["Label_1_Virus_category"].lower() == "virus":
                label = 1
            else:
                continue
            image_name_to_label[row["X_ray_image_name"]] = label
    with open("data/metadata2.csv") as file:  # GitHub dataset
        # Use COVID-19, SARS
        metadata_contents = csv.DictReader(file)
        for row in metadata_contents:
            if row["filename"] in image_name_to_label:  # Image already added
                continue
            if "covid-19" in row["finding"].lower():
                label = 0
            elif row["finding"].lower() == "sars":
                label = 1
            else:
                continue
            image_name_to_label[row["filename"]] = label
    with open("data/metadata_COVID-19.csv") as file:  # Additional COVID-19 images
        metadata_contents = csv.DictReader(file)
        for row in metadata_contents:
            name = "COVID-19/" + row["FILE NAME"] + "." + row["FORMAT"]
            image_name_to_label[name.lower().replace(" ", "")] = 0
    with open("data/metadata_ViralPneumonia.csv") as file:  # Additional virus images
        metadata_contents = csv.DictReader(file)
        for row in metadata_contents:
            name = "ViralPneumonia/" + row["FILE NAME"].replace("-", "(") + ")." + row["FORMAT"]
            image_name_to_label[name.lower().replace(" ", "")] = 1
    with open("data/metadata_Normal.csv") as file:  # Additional normal images
        metadata_contents = csv.DictReader(file)
        for row in metadata_contents:
            name = "Normal/" + row["FILE NAME"].replace("-", "(") + ")." + row["FORMAT"]
            image_name_to_label[name.lower().replace(" ", "")] = 2

    notifier.send("    Loading images...")
    images, labels = load_images(image_name_to_label)

    notifier.send("    Splitting data...")
    return split_data(images, labels)


def generate_model():
    """
    Returns a compiled convolutional neural network
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(LABEL_COUNT, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def generate_model_xception():
    """
    Returns a compiled convolutional neural network using transfer
    learning with Xception
    """
    base_model = tf.keras.applications.Xception(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    )
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(LABEL_COUNT, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def generate_model_inception_v3():
    """
    Returns a compiled convolutional neural network using transfer
    learning with Inception v3
    """
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    )
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(LABEL_COUNT, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    # Configure GPU memory usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            notifier.send(error)

    notifier.send("Loading data...")
    images_train, images_validate, images_test, \
        labels_train, labels_validate, labels_test = load_data()

    notifier.send("Generating model...")
    if MODEL_TYPE is ModelType.CUSTOM:
        model = generate_model()
    elif MODEL_TYPE is ModelType.XCEPTION:
        model = generate_model_xception()
    elif MODEL_TYPE is ModelType.INCEPTION_V3:
        model = generate_model_inception_v3()
    else:
        raise Exception("Invalid MODEL_TYPE provided")

    notifier.send("Fitting model...")
    model.fit(
        images_train,
        labels_train,
        batch_size=16,
        epochs=5,
        validation_data=(images_validate, labels_validate),
        callbacks=[TrainingNotifier(notifier)]
    )

    notifier.send("Testing model...")
    log_dict = model.evaluate(
        images_test,
        labels_test,
        batch_size=16,
        return_dict=True
    )
    notifier.send(
        f"Finished testing model with accuracy: {log_dict['accuracy']}",
        print_message=False
    )

    notifier.send("Saving model...")
    model_file_name = "models" + os.sep
    if MODEL_TYPE is ModelType.CUSTOM:
        model_file_name += "model_custom.h5"
    elif MODEL_TYPE is ModelType.XCEPTION:
        model_file_name += "model_xception.h5"
    elif MODEL_TYPE is ModelType.INCEPTION_V3:
        model_file_name += "model_inception_v3.h5"
    else:
        raise Exception("Invalid MODEL_TYPE provided")
    model.save(model_file_name)


if __name__ == "__main__":
    main()
