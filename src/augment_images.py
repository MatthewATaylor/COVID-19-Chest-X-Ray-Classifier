import os

import cv2
import numpy as np

from constants import IMG_SIZE, DEFAULT_IMG_PATH_EDITED, IMAGE_IDENTIFIER_SEQUENCE

IMG_CHANNELS = 1

DEFAULT_IMG_PATH_UNEDITED = "data" + os.sep + "images" + os.sep + "unedited"
COVID_IMG_PATH_UNEDITED = DEFAULT_IMG_PATH_UNEDITED + os.sep + "COVID-19"
VIRUS_IMG_PATH_UNEDITED = DEFAULT_IMG_PATH_UNEDITED + os.sep + "ViralPneumonia"
NORMAL_IMG_PATH_UNEDITED = DEFAULT_IMG_PATH_UNEDITED + os.sep + "Normal"

COVID_IMG_PATH_EDITED = DEFAULT_IMG_PATH_EDITED + os.sep + "COVID-19"
VIRUS_IMG_PATH_EDITED = DEFAULT_IMG_PATH_EDITED + os.sep + "ViralPneumonia"
NORMAL_IMG_PATH_EDITED = DEFAULT_IMG_PATH_EDITED + os.sep + "Normal"


def formatted_image(image_path):
    """
    Returns the image at the image_path, modified for better model
    training
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.reshape(image, (IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    return image


def gen_unique_image_path(standard_image_path, image_num):
    """
    Generates a unique image path based on the standard_image_path
    by adding the IMAGE_IDENTIFIER_SEQUENCE followed by a number
    """
    # Index where extension sequence starts (.jpg, .jpeg, or .png)
    extension_start = standard_image_path.lower().find(".jp")
    if extension_start == -1:
        extension_start = standard_image_path.lower().find(".png")
    if extension_start == -1:
        raise Exception("Invalid extension in image path: " + standard_image_path)

    # Split path into base and extension
    path_base = standard_image_path[:extension_start]
    path_extension = standard_image_path[extension_start:]

    # Insert unique identifier
    unique_identifier = IMAGE_IDENTIFIER_SEQUENCE + "(" + str(image_num) + ")"
    return path_base + unique_identifier + path_extension


def rotate_image(image, angle):
    """
    Returns the image rotated by the given angle about the image's
    center
    """
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return rotated_image


def image_noise_gaussian(image):
    """
    Adds Gaussian noise to the provided image
    """
    float_img = image.astype(np.float)
    gauss = np.random.normal(0.0, 4.0, (IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    gauss = gauss.reshape(IMG_SIZE, IMG_SIZE, IMG_CHANNELS).astype(np.float)
    result = float_img + gauss
    result = np.clip(result, 0, 255)
    result = result.astype(np.uint8)
    return result


def save_augmented_images(standard_image, standard_image_path):
    """
    Saves multiple augmented forms of the standard_image located at
    the standard_image_path, each with a unique name
    """
    # Rotated images
    for i in range(8):
        cv2.imwrite(
            gen_unique_image_path(standard_image_path, i),
            rotate_image(standard_image, i * 45 + 45)
        )

    # Flipped images
    cv2.imwrite(
        gen_unique_image_path(standard_image_path, 8),
        cv2.flip(standard_image, 0)  # Flip around x axis
    )
    cv2.imwrite(
        gen_unique_image_path(standard_image_path, 9),
        cv2.flip(standard_image, 1)  # Flip around y axis
    )

    # Blurred image
    cv2.imwrite(
        gen_unique_image_path(standard_image_path, 10),
        cv2.GaussianBlur(standard_image, (5, 5), 0)
    )

    # Noise image
    cv2.imwrite(
        gen_unique_image_path(standard_image_path, 11),
        image_noise_gaussian(standard_image)
    )


def get_image_names():
    """
    Returns (image_names, covid_image_names, normal_image_names,
    virus_image_names), where each is a list of image names
    """
    image_names = os.listdir(DEFAULT_IMG_PATH_UNEDITED)

    # Remove directories
    image_names.remove("COVID-19")
    image_names.remove("Normal")
    image_names.remove("ViralPneumonia")

    covid_image_names = os.listdir(COVID_IMG_PATH_UNEDITED)
    normal_image_names = os.listdir(NORMAL_IMG_PATH_UNEDITED)
    virus_image_names = os.listdir(VIRUS_IMG_PATH_UNEDITED)

    return image_names, covid_image_names, normal_image_names, virus_image_names


def main():
    image_names, covid_image_names, normal_image_names, virus_image_names = get_image_names()

    for i, image_name in enumerate(image_names):
        print(f"    Default category: {i / len(image_names) * 100}%")
        edited_image_path = DEFAULT_IMG_PATH_EDITED + os.sep + image_name
        image = formatted_image(DEFAULT_IMG_PATH_UNEDITED + os.sep + image_name)
        cv2.imwrite(edited_image_path, image)
        save_augmented_images(image, edited_image_path)

    for i, image_name in enumerate(covid_image_names):
        print(f"    COVID-19 category: {i / len(covid_image_names) * 100}%")
        edited_image_path = COVID_IMG_PATH_EDITED + os.sep + image_name
        image = formatted_image(COVID_IMG_PATH_UNEDITED + os.sep + image_name)
        cv2.imwrite(edited_image_path, image)
        save_augmented_images(image, edited_image_path)

    for i, image_name in enumerate(normal_image_names):
        print(f"    Normal category: {i / len(normal_image_names) * 100}%")
        edited_image_path = NORMAL_IMG_PATH_EDITED + os.sep + image_name
        image = formatted_image(NORMAL_IMG_PATH_UNEDITED + os.sep + image_name)
        cv2.imwrite(edited_image_path, image)
        save_augmented_images(image, edited_image_path)

    for i, image_name in enumerate(virus_image_names):
        print(f"    Viral pneumonia category: {i / len(virus_image_names) * 100}%")
        edited_image_path = VIRUS_IMG_PATH_EDITED + os.sep + image_name
        image = formatted_image(VIRUS_IMG_PATH_UNEDITED + os.sep + image_name)
        cv2.imwrite(edited_image_path, image)
        save_augmented_images(image, edited_image_path)


if __name__ == "__main__":
    main()
