
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def apply_color_flatten(group):
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        # Reshape the image into a 2D array of pixels
        pixels = picture.reshape((-1, 3))

        # Convert the pixel values to floating-point
        pixels = np.float32(pixels)

        # Define the criteria for k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Define the number of clusters (k)
        k = 5

        # Perform k-means clustering
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert the centers of the clusters to integers
        centers = np.uint8(centers)

        # Reshape the labels to the original image shape
        labels = labels.reshape(picture.shape[:2])

        # Initialize an empty segmented image
        segmented_image = np.zeros_like(picture)

        # Assign each pixel to its corresponding cluster center
        for c in range(k):
            segmented_image[labels == c] = centers[c]

        # plt.imshow(segmented_image), plt.show()
        cv2.imwrite("02_colorflatten_mircs\\" + group + "\\patch_id" + str(i) + ".png", segmented_image)


def apply_color_inverse(group):
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        #picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        inverted_image = cv2.bitwise_not(picture)
        # inverted_image = cv2.resize(inverted_image, (50, 50), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite("04_inverse_mircs\\" + group + "\\patch_id" + str(i) + ".png", inverted_image)


def apply_horizontal_flip(group):
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        flipped_image = cv2.flip(picture, 0)
        # flipped_image = cv2.resize(inverted_image, (50, 50), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite("01_flipvertical_mircs\\" + group + "\\patch_id" + str(i) + ".png", flipped_image)


def apply_noise2(group):
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
        # noisy_picture = cv2.resize(inverted_image, (50, 50), interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite("05_noise_mircs\\"+group + "\\patch_id" + str(i) + ".png", flipped_image)
        plt.imshow(noisy_picture), plt.show()
        break


def apply_noise(group):
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        height, width, channels = picture.shape
        shuffled_image = picture.copy()
        # shuffled_image = cv2.cvtColor(shuffled_image, cv2.COLOR_BGR2RGB)
        for y in range(height):
            for x in range(width):
                # Check if shuffling should occur (50% chance)
                if np.random.rand() < 0.5:
                    # Randomly choose another pixel
                    new_x, new_y = np.random.randint(0, width), np.random.randint(0, height)

                    # Swap the pixels
                    shuffled_image[y, x], shuffled_image[new_y, new_x] = shuffled_image[new_y, new_x], shuffled_image[
                        y, x]

        # noisy_picture = cv2.resize(inverted_image, (50, 50), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite("05_noise_mircs\\" + group + "\\patch_id" + str(i) + ".png", shuffled_image)
        # plt.imshow(shuffled_image), plt.show()


def apply_gridshuffle(group):
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    for i in index_list:
        picture = cv2.imread("color_mircs\\" + group + "\\patch_id" + str(i) + ".png")
        picture = cv2.resize(picture,(400, 400))
        min_dim = min(picture.shape[:2])
        image = picture[:min_dim, :min_dim]

        # Split the image into 16 squares
        num_squares_per_row = 4
        squares = split_into_squares(image, num_squares_per_row)

        # Shuffle the squares
        shuffled_squares = shuffle_squares(squares)

        # Reconstruct the shuffled image
        shuffled_image = reconstruct_image(shuffled_squares, num_squares_per_row)
        cv2.imwrite("03_gridshuffle_mircs\\" + group + "\\patch_id" + str(i) + ".png", shuffled_image)


def split_into_squares(image, num_squares_per_row):
    height, width = image.shape[:2]
    square_size = height // num_squares_per_row

    squares = []
    for i in range(num_squares_per_row):
        for j in range(num_squares_per_row):
            square = image[i * square_size: (i + 1) * square_size, j * square_size: (j + 1) * square_size]
            squares.append(square)
    return squares


def shuffle_squares(squares):
    shuffled_squares = squares.copy()
    random.shuffle(shuffled_squares)
    return shuffled_squares


def reconstruct_image(squares, num_squares_per_row):
    square_size = squares[0].shape[0]
    image_size = square_size * num_squares_per_row
    reconstructed_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    for i in range(num_squares_per_row):
        for j in range(num_squares_per_row):
            reconstructed_image[i * square_size: (i + 1) * square_size, j * square_size: (j + 1) * square_size] = \
            squares[i * num_squares_per_row + j]

    return reconstructed_image

def print_texture_change(group):
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
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    df = df[df["class"] == group]
    index_list = df["id"]
    texture_mirc = cv2.imread("_changingTextureFull\\full\\" + group + "_full.png")
    for i in index_list:
        tl = df.loc[df["id"] == i, "top left"].iloc[0]
        br = df.loc[df["id"] == i, "bottom right"].iloc[0]
        # Clean up the string
        tl = tl.replace("[", "").replace("]", "").replace(",", " ").split()
        br = br.replace("[", "").replace("]", "").replace(",", " ").split()

        h1 = int(float(tl[0]))
        h2 = int(float(br[0]))
        w1 = int(float(tl[1]))
        w2 = int(float(br[1]))
        cut_mirc = texture_mirc[w1:w2, h1:h2]
        #plt.imshow(cut_mirc), plt.show()
        cv2.imwrite("06_texture_mircs\\" + group + "\\patch_id" + str(i) + ".png", cut_mirc)
