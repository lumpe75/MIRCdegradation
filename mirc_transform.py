
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def apply_color_flatten(group):
    """
    args:
        group: the class/group of images
    Apply color flatten operation to a class of images by segmenting the image with a k-means clustering algorithm and
    setting the center color of a cluster for the full cluster
    """
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        pixels = picture.reshape((-1, 3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        k = 5
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        labels = labels.reshape(picture.shape[:2])
        segmented_image = np.zeros_like(picture)
        for c in range(k):
            segmented_image[labels == c] = centers[c]

        # plt.imshow(segmented_image), plt.show()
        cv2.imwrite("02_colorflatten_mircs\\" + group + "\\patch_id" + str(i) + ".png", segmented_image)


def apply_color_inverse(group):
    """
    args:
        group: the class/group of images
    Apply color inverse operation to a class of images by flipping all rgb values of an image
    """
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        inverted_image = cv2.bitwise_not(picture)
        cv2.imwrite("04_inverse_mircs\\" + group + "\\patch_id" + str(i) + ".png", inverted_image)


def apply_horizontal_flip(group):
    """
    args:
        group: the class/group of images
    Apply the horizontal flip operation (left <=> right) to a class of images
    """
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        flipped_image = cv2.flip(picture, 0)
        cv2.imwrite("01_flipvertical_mircs\\" + group + "\\patch_id" + str(i) + ".png", flipped_image)


def apply_noise_full_random(group):
    """
    args:
        group: the class/group of images
    Apply the noise operation to a class of images by adding gaussian noise.
    """
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    mean = 0
    variance = 100
    sigma = variance ** 0.5
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        gaussian_noise = np.random.normal(mean, sigma, picture.shape).astype('uint8')
        noisy_picture = cv2.add(picture, gaussian_noise)
        plt.imshow(noisy_picture), plt.show()


def apply_noise(group):
    """
    args:
        group: the class/group of images
    Apply the altered noise operation to a class of images by switching random pixels in the image with a chance of 0.5.
    """
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        height, width, channels = picture.shape
        shuffled_image = picture.copy()
        for y in range(height):
            for x in range(width):
                if np.random.rand() < 0.5:  # edited "noise amount" here
                    new_x, new_y = np.random.randint(0, width), np.random.randint(0, height)
                    shuffled_image[y, x], shuffled_image[new_y, new_x] = shuffled_image[new_y, new_x], shuffled_image[
                        y, x]
        cv2.imwrite("05_noise_mircs\\" + group + "\\patch_id" + str(i) + ".png", shuffled_image)


def apply_grid_shuffle(group):
    """
    args:
        group: the class/group of images
    Apply the grid shuffle operation to a class of images by separating the image into a 4x4 grid and switching
    the patches around
    """
    def split_into_squares(pic, number_of_cuts):
        """
        args:
            pic: the image to be split
            number_of_cuts: the dimension of the resulting quadratic grid
        """
        height, width = pic.shape[:2]
        square_size = height // number_of_cuts
        patches = []
        for noc in range(number_of_cuts):
            for nod in range(number_of_cuts):
                square = pic[noc * square_size: (noc + 1) * square_size, nod * square_size: (nod + 1) * square_size]
                patches.append(square)
        return patches

    def reconstruct_image(patches, number_of_cuts):
        """
        args:
            patches: a list of images
            number_of_cuts: the dimension of the resulting quadratic grid
        """
        square_size = patches[0].shape[0]
        image_size = square_size * number_of_cuts
        reconstructed_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        for noc in range(number_of_cuts):
            for nod in range(number_of_cuts):
                reconstructed_image[noc * square_size: (noc + 1) * square_size, nod * square_size: (nod + 1) * square_size] = \
                    patches[noc * number_of_cuts + nod]

        return reconstructed_image

    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        picture = cv2.resize(picture, (400, 400))
        min_dim = min(picture.shape[:2])
        image = picture[:min_dim, :min_dim]
        num_squares_per_row = 4
        squares = split_into_squares(image, num_squares_per_row)
        random.shuffle(squares)
        shuffled_image = reconstruct_image(squares, num_squares_per_row)
        cv2.imwrite("03_gridshuffle_mircs\\" + group + "\\patch_id" + str(i) + ".png", shuffled_image)


def print_texture_change(group):
    """
    args:
        group: the class/group of images
    Apply a texture change operation to the full image of a class, serving as basis for the MIRC texture change
    """
    full_color = cv2.imread("_coloredFull400px\\" + group + "_edited400px.png", cv2.IMREAD_UNCHANGED)
    cutout = cv2.imread("_changingTextureFull\\empty\\" + group + "_empty.png", cv2.IMREAD_UNCHANGED)
    texture = cv2.imread("_changingTextureFull\\texture\\" + group + "_texture.jpg")
    texture = cv2.resize(texture, (400, 400), interpolation=cv2.INTER_LINEAR)

    alpha_channel = cutout[:, :, 3]
    mask = np.where(alpha_channel != 0, 255, 0).astype(np.uint8)
    result = np.where(mask[:, :, np.newaxis] != 0, texture, full_color)

    plt.imshow(result), plt.show()
    cv2.imwrite("_changingTextureFull\\full\\" + group + "_full.png", result)


def apply_texture_change(group):
    """
    args:
        group: the class/group of images
    Cut the full texture-changed image down to its MIRC size per image
    """
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    texture_mirc = cv2.imread("_changingTextureFull\\full\\" + group + "_full.png")
    for i in index_list:
        tl = df.loc[df["id"] == i, "top left"].iloc[0]
        br = df.loc[df["id"] == i, "bottom right"].iloc[0]
        tl = tl.replace("[", "").replace("]", "").replace(",", " ").split()
        br = br.replace("[", "").replace("]", "").replace(",", " ").split()
        h1 = int(float(tl[0]))
        h2 = int(float(br[0]))
        w1 = int(float(tl[1]))
        w2 = int(float(br[1]))
        cut_mirc = texture_mirc[w1:w2, h1:h2]
        cv2.imwrite("06_texture_mircs\\" + group + "\\patch_id" + str(i) + ".png", cut_mirc)
