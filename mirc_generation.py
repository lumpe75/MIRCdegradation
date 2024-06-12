import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_classes():
    """
    Returns a list of class names
    """
    return ["bike", "eagle", "car", "glasses", "horse", "plane", "ship", "suit"]


def get_list(name):
    """
    args:
        name: the name of the specific class
    returns a list of lists, each inner list contains [id, original size, scaling factor, Boolean if feature matching
    was successfully]
    """
    if name == "bike":
        return [[2, 14.0, 1.654095703541293, True],
                [7, 11.666667, 1.8, True],
                [20, 25.0, 1.112726883393597, True],
                [21, 22.0, 1.6, True],
                [46, 12.0, 2.3348699408644147, True],
                [58, 18.9, 1.1319844124912524, True],
                [71, 8.2, 1.4, True],
                [74, 18.0, 1.5, True],
                [76, 25.0, 1.43560185807153, True],
                [89, 20.0, 1.1070999054285668, True],
                [109, 11.6, 1.5, True],
                [116, 24.666667, 1.325455376026936, True],
                [121, 16.0, 1.0, True],
                [147, 22.857143, 1.146690526842785, True],
                [185, 13.333333, 1.5998658745293997, True],
                [194, 11.885714, 1.4, True],
                [202, 17.4, 1.7226555313330123, True],
                [213, 9.111111, 1.4, True],
                [216, 23.68, 1.2728789909904252, True],
                [220, 21.0, 1.0, True],
                [230, 13.0, 1.0576645760082006, True]]

    if name == "eagle":
        return [[4, 16.0, 1.0, True],
                [15, 20.0, 1.0997410681702628, True],
                [23, 20.0, 1.229310569313639, True],
                [28, 30.0, 1.05, True],
                [38, 13.0, 1.0, True],
                [53, 13.0, 1.0, True],
                [83, 20.0, 1.9, True],
                [97, 20.0, 1.1545676804018832, True],
                [105, 26.0, 1.0, False],
                [116, 18.285714, 1.8, True],
                [122, 18.9, 1.85, True],
                [135, 26.0, 1.0442013735971034, True],
                [154, 16.571429, 1.754100332167455, True]]

    if name == "car":
        return [[22, 18.9, 1.1385249674845332, True],
                [32, 17.777778, 1.2801377975047252, True],
                [60, 14.666667, 1.5420962950589192, True]]

    if name == "fly":
        return [[27, 16.0, 1.4, False],
                [40, 17.142857, 1.4, False],
                [49, 16.0, 1.6208870702983653, True],
                [61, 15.428571, 1.8281264580710388, True],
                [67, 13.714286, 1.8, False],
                [68, 12.571429, 1.6, False],
                [75, 11.7, 1.4, False],
                [98, 18.0, 1.268761644186037, True],
                [114, 17.311111, 1.2, False],
                [125, 18.222222, 1.0, True],
                [136, 18.9, 1.0, True],
                [138, 16.0, 1.0, True],
                [151, 13.0, 1.0, True],
                [157, 14.577778, 2.5, False],
                [181, 14.0, 1.0, True],
                [199, 14.0, 1.4, False],
                [210, 13.0, 1.9, False],
                [219, 11.0, 1.6, False],
                [239, 11.76, 1.6, False],
                [245, 16.2, 1.3, False],
                [249, 12.0, 2.4, True]]

    if name == "glasses":
        return [[24, 11.666667, 1.8, True],
                [44, 15.0, 1.6, True],
                [60, 17.777778, 1.8, False],
                [66, 10.8, 2.0, False],
                [68, 10.285714, 2.0, False],
                [74, 6.0, 2.0, False],
                [104, 6.0, 2.0, False],
                [111, 7.2, 2.0, False],
                [114, 10.0, 1.8, False],
                [132, 14.628571, 1.4, True],
                [149, 12.0, 1.5, True],
                [174, 20.0, 1.0, True],
                [187, 5.942857, 4.0, False],
                [194, 12.428571, 1.8734451922723512, True],
                [205, 14.4, 1.5, True],
                [209, 13.0, 2.0137827565917026, True],
                [220, 8.4, 2.6, True],
                [227, 21.0, 1.0, True],
                [263, 9.24, 2.0, False],
                [268, 12.0, 2.0, False]]

    if name == "horse":
        return [[1, 20.0, 0.9889897838739502, True],
                [13, 11.666667, 1.7, False],
                [30, 28.8, 1.074431449792284, True],
                [35, 18.0, 1.1577358183470121, True],
                [59, 8.0, 1.5, True],
                [94, 13.0, 1.0, True],
                [107, 16.0, 1.2549726223696513, True],
                [130, 7.25, 1.6, True],
                [137, 16.0, 1.0, False],
                [147, 11.0, 2.0, False],
                [152, 9.0, 2.0, False],
                [238, 10.64, 2.0, False],
                [242, 8.0, 1.95, True],
                [259, 12.6, 2.0, True],
                [267, 14.857143, 2.131052670257795, True],
                [273, 10.971429, 1.0, False],
                [281, 14.5, 1.6858134308, True],
                [288, 16.444444, 1.1955540841, True],
                [295, 16.444444, 1.4532055162608313, True],
                [299, 10.0, 1.0, True],
                [316, 18.9, 1.2114765469279483, True],
                [327, 15.3, 1.161155775000713, True],
                [335, 8.14, 1.0, False]]

    if name == "plane":
        return [[7, 21.0, 1.0, True],
                [15, 16.0, 1.0, True],
                [22, 21.0, 0.9917635573036866, True],
                [26, 20.0, 1.0324738464798617, True],
                [50, 18.9, 1.1929378969394697, True],
                [56, 14.8, 1.3841661956708586, True],
                [85, 16.0, 1.0, True],
                [89, 18.285714, 1.397930817426827, True],
                [102, 14.5, 1.7848205601242366, True],
                [113, 14.4, 1.3665767923315586, True],
                [117, 24.0, 1.0, True],
                [136, 14.666667, 1.5297855706003165, True],
                [142, 15.64, 1.8290995317170349, True],
                [152, 16.2, 1.605094182211366, True],
                [154, 16.0, 1.4486038288039653, True],
                [164, 16.0, 1.5319260746317962, True],
                [185, 14.0, 1.4537094615989783, True],
                [204, 14.628571, 1.6180225038986176, True],
                [213, 21.0, 0.9917915771717571, True]]

    if name == "ship":
        return [[2, 25.0, 1.063664656823361, True],
                [19, 13.125, 1.4289011452945233, True],
                [21, 18.0, 1.5197982003579376, True],
                [28, 17.0, 1.8953912321643567, True],
                [29, 26.0, 1.0, True],
                [41, 18.0, 1.64809936108778, True],
                [59, 13.0, 1.0, True],
                [95, 21.0, 1.0, True],
                [111, 21.0, 1.0, True],
                [120, 24.0, 1.3813157529047528, True],
                [134, 20.0, 1.3850235433404505, True],
                [153, 16.571429, 1.669776061685432, True],
                [164, 11.885714, 1.55, True],
                [171, 13.0, 1.2227345995567511, True],
                [183, 13.0, 1.2, True]]

    if name == "suit":
        return [[1, 20.0, 1.2, True],
                [2, 20.0, 1.2, False],
                [3, 14.0, 2.4299053778752335, True],
                [54, 21.0, 1.0, True],
                [66, 18.9, 1.2, True],
                [95, 10.4, 4.023763319216937, True],
                [101, 15.64, 2.0, True],
                [122, 15.488889, 1.2292423833640622, True],
                [128, 18.9, 1.0, True],
                [130, 17.0, 1.0, True]]

    return []


def loop_over_class(name):
    """
    args:
        name: the name of the class
    returns a comparison of feature matches for the given class, used to verify if feature matching was
    successful or not
    """
    plt.close('all')
    full_list = get_list(name)

    mirc_length = 400  # 50 x 4 x 2
    mirc_points = (mirc_length, mirc_length)
    mirc = cv.imread("_coloredFull\\" + name + "_edited.png")
    mirc = cv.cvtColor(mirc, cv.COLOR_BGR2RGB)
    mirc = cv.resize(mirc, mirc_points, interpolation=cv.INTER_LINEAR)
    scalar_list = []
    images_color = []
    images_bw = []
    for k in full_list:
        submirc = cv.imread("bw_mircs\\" + name + "\\patch_id" + str(k[0]) + ".png")
        old_shape = (int(submirc.shape[0]), int(submirc.shape[0]))
        submirc_length = int(k[1] * k[2]) * 8  # 4 x 2
        submirc_points = (submirc_length, submirc_length)
        submirc = cv.resize(submirc, submirc_points, interpolation=cv.INTER_LINEAR)
        colored_mirc, scaler, tl, br = paint_and_match_submirc(mirc, submirc, k[0], k[3])
        colored_mirc = cv.cvtColor(colored_mirc, cv.COLOR_RGB2BGR)
        scalar_list.append([k[0], scaler])

        cv.imwrite("color_mircs\\" + name + "\\patch_id" + str(k[0]) + ".png", colored_mirc)
        images_color.append(cv.resize(colored_mirc, old_shape, interpolation=cv.INTER_LINEAR))
        images_bw.append(cv.resize(submirc, old_shape, interpolation=cv.INTER_LINEAR))

    draw_table(images_color, images_bw, name)


def draw_table(images_list1, images_list2, table_name):
    """
    args:
        image_list1: list of images on the left side of the table
        images_list2: list of images on the right side of the table
        table_name: name of the saved file
    Helper function for 'loop_over_class' to visualize the result
    """
    common_size = (100, 100)
    colored_photos_resized = [cv.resize(img, common_size) for img in images_list1]
    bw_photos_resized = [cv.resize(img, common_size) for img in images_list2]

    triplets = []
    max_height = max(img.shape[0] for img in colored_photos_resized)
    for bw_img, colored_img in zip(bw_photos_resized, colored_photos_resized):
        integer_width = int(bw_img.shape[1] * 0.1)

        bw_img_with_space = np.zeros((max_height, bw_img.shape[1] + integer_width, bw_img.shape[2]), dtype=np.uint8)
        bw_img_with_space[:bw_img.shape[0], :bw_img.shape[1]] = bw_img

        combined_img = np.concatenate((bw_img_with_space, cv.cvtColor(colored_img, cv.COLOR_RGB2BGR)), axis=1)
        triplets.append(combined_img)

    combined_image = np.concatenate(triplets, axis=0)
    cv.imwrite("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\coloring\\pythonProject3\\comparisons\\" + table_name + ".jpg",
               cv.cvtColor(combined_image, cv.COLOR_RGB2BGR))
    plt.imshow(combined_image), plt.show()


def build_data_assistent(name):
    """
    Assistant function for 'loop_over_data', generating the values per class
    """
    full_list = get_list(name)
    mirc_length = 400  # 50 x 4 x 2
    mirc_points = (mirc_length, mirc_length)
    mirc = cv.imread("_coloredFull\\" + name + "_edited.png")
    mirc = cv.cvtColor(mirc, cv.COLOR_BGR2RGB)
    mirc = cv.resize(mirc, mirc_points, interpolation=cv.INTER_LINEAR)
    top_left = []
    bottom_right = []
    cl = []
    number = []
    size = []
    for k in full_list:
        if k[3]:
            submirc = cv.imread("bw_mircs\\" + name + "\\patch_id" + str(k[0]) + ".png")
            submirc_length = int(k[1] * k[2]) * 8  # 4 x 2
            submirc_points = (submirc_length, submirc_length)
            submirc = cv.resize(submirc, submirc_points, interpolation=cv.INTER_LINEAR)
            colored_mirc, scaler, tl, br = paint_and_match_submirc(mirc, submirc, k[0], k[3])
            top_left.append(tl)
            bottom_right.append(br)
            cl.append(name)
            number.append(k[0])
            size.append(submirc_points)
    return top_left, bottom_right, cl, number, size


def loop_over_data():
    """
    creates a csv file containing the conducted cropping on the original images
    """
    class_list = get_classes()
    top_left = []
    bottom_right = []
    cl = []
    number = []
    size = []
    for c in class_list:
        tl, br, cla, numb, siz = build_data_assistent(c)
        top_left = top_left + tl
        bottom_right = bottom_right + br
        cl = cl + cla
        number = number + numb
        size = size + siz
    df = pd.DataFrame({'class': cl, 'id': number, 'size': size, 'top left': top_left, 'bottom right': bottom_right})
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",", index=False)


def paint_and_match_submirc(mirc, submirc, picture_id, correct):
    """
    args:
        mirc: the original image
        submirc: the reduced version in black and white
        picture_id: the name of the submirc
        correct: a boolean stating if the matching was successful in previous runs

    Feature matches a MIRC to a subMIRC to color in the subMIRC accordingly if possible.
    """

    def compute_angle(source_points, dest_points):
        """
        args:
            source_points: a list of starting points of lines
            dest_points: a list of destination points of lines
        Returns a list of degrees between lines to compare, if they are parallel
        """
        degree = []
        for src_pt, dst_pt in zip(source_points, dest_points):
            vector = np.array(dst_pt) - np.array(src_pt)
            angle = np.arctan2(vector[1], vector[0])
            angle_degrees = np.degrees(angle) % 360
            degree.append(angle_degrees)
        return degree

    def ratio_cd_to_ab(point_a, point_b, point_c, point_d):
        """
        Calculates the ratio between the distance of point_a and point_b and the distance of point_c and point_d.
        """
        distance_ab = math.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)
        distance_cd = math.sqrt((point_d[0] - point_c[0]) ** 2 + (point_d[1] - point_c[1]) ** 2)
        if distance_ab == 0:
            return 1.0
        return distance_cd / distance_ab

    small_gray = cv.cvtColor(submirc, cv.COLOR_BGR2GRAY)
    large_gray = cv.cvtColor(mirc, cv.COLOR_BGR2GRAY)

    # Apply feature matching algorithm
    sift = cv.SIFT_create()
    kp1, descriptors1 = sift.detectAndCompute(small_gray, None)
    kp2, descriptors2 = sift.detectAndCompute(large_gray, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good = []
    factor = 0.75
    while len(good) == 0:
        for m, n in matches:
            if m.distance < factor * n.distance:
                good.append([m])
        factor += 0.05
        if factor > 1.0:
            print("=====No match found=====")
            break
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if not correct:  # visualize non-correct feature matches
        img3 = cv.drawMatchesKnn(small_gray, kp1, large_gray, kp2, good, None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3), plt.title(str(id)), plt.show()

    scale_value = 1.0
    nr_of_matches = src_pts.shape[0]
    if nr_of_matches > 1:
        mean_length = nr_of_matches // 2
        if mean_length % 2 == 0:
            mean1, mean2 = mean_length, mean_length - 1
        else:
            mean1, mean2 = int(mean_length + 0.5), int(mean_length - 0.5)
        scale_value = ratio_cd_to_ab(src_pts[mean1].flatten(), src_pts[mean2].flatten(), dst_pts[mean1].flatten(),
                                     dst_pts[mean2].flatten())

        src_list = [list(item) for item in np.squeeze(src_pts, axis=1)]
        dst_list = [list(item) for item in np.squeeze(dst_pts, axis=1)]
        angles = sorted(compute_angle(src_list, dst_list))
        print(str(picture_id) + " with " + str(angles))

    c = np.zeros((2, 3), dtype=np.float32)
    c[:, :2] = np.eye(2)
    c[:, 2] = np.mean(dst_pts - src_pts, axis=0).flatten()

    h = submirc.shape[0]
    w = submirc.shape[1]
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv.transform(corners, c)
    top_left = transformed_corners[0].flatten()
    bottom_right = transformed_corners[2].flatten()

    cropped_image = mirc[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]

    return cropped_image, scale_value, top_left, bottom_right


def compare_values():
    """
    Creates a csv-file comparing the differences between stated and actual pixel values
    """
    all_mircs = []
    for c in get_classes():
        idl, pxl, scf = get_list(c)
        for index, content in enumerate(idl):
            print("bw_mircs\\" + c + "\\patch_id" + str(content) + ".png")
            submirc = cv.imread("bw_mircs\\" + c + "\\patch_id" + str(content) + ".png")
            real_shape = submirc.shape[0]
            all_mircs.append([real_shape, pxl[index], c + str(content)])
    df = pd.DataFrame(sorted(all_mircs, key=lambda x: (x[0], x[1])))
    df = df.rename(columns={0: "real pixel", 1: "stated pixel", 2: "file name"})
    df['matches'] = df['real pixel'] == df['stated pixel']
    df['difference'] = df['real pixel'] - df['stated pixel']
    print("Number of mismatches: " + str((~df['matches']).sum()) + " / " + str(len(df)))
    df.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\value_overview.csv", sep=",", index=False)


def draw_overview_table(group, picture_id):
    """
    args:
        group: a group/class of images
        picture_id: the id of an individual image
    Draws a comparison table which visualizes all the different versions of a MIRC
    """
    # the list of different versions existing
    folders = ["color_mircs",
               "bw_mircs",
               "01_flipvertical_mircs",
               "02_colorflatten_mircs",
               "03_gridshuffle_mircs",
               "04_inverse_mircs",
               "05_noise_mircs",
               "06_texture_mircs"]
    left = []
    right = []
    for i, f in enumerate(folders):
        path = f + "\\" + group + "\\patch_id" + str(picture_id) + ".png"
        image = cv.imread(path)
        image = cv.resize(image, (200, 200))
        left.append(image) if i % 2 == 0 else right.append(image)
    doubles = []
    for i, _ in enumerate(left):
        doubles.append(np.concatenate((left[i], right[i]), axis=1))
    result = np.concatenate(doubles, axis=0)
    cv.imwrite("comparisons\\" + group + "\\patch_id" + str(picture_id) + ".png", result)


def create_all_tables():
    """
    Wrapper for 'draw_overview_table' over all classes
    """
    df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    cl = get_classes()
    for group in cl:
        df2 = df[df["class"] == group]
        index_list = df2["id"]
        for i in index_list:
            draw_overview_table(group, i)


def paint_padding(pic, top_left, bottom_right):
    """
    args:
        pic: the full MIRC
        top_left: the top left corner of the image
        bottom_right: the bottom right corner of the image
    Applies grey padding for an individual image
    """
    parent_size = 400
    grey_background = np.full((parent_size, parent_size, 3), (128, 128, 128), dtype=np.uint8)
    top_left = top_left.strip("[]").split()
    bottom_right = bottom_right.strip("[]").split()
    pic = cv.flip(pic, 0)
    # for flip vertical, the padding needs also to be flipped
    # grey_background[int(float(top_left[1])):int(float(bottom_right[1])), int(float(top_left[0])):int(float(bottom_right[0]))] = pic
    grey_background[int(float(top_left[1])):int(float(bottom_right[1])),
                    int(float(top_left[0])):int(float(bottom_right[0]))] = pic
    grey_background = cv.flip(grey_background, 0)
    return grey_background


def apply_padding():
    """
    Applies a grey padding for each MIRC
    """
    frame = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\mirc_cropping.csv", sep=",")
    # for flip vertical, the padding needs also to be flipped
    pairs = [
        ["01_flipvertical_mircs", "08_flip_padded"]  # ,
        # ["04_inverse_mircs", "07_inverse_padded"],
        # ["06_texture_mircs", "09_texture_padded"]
    ]
    for p in pairs:
        sou = p[0]
        tar = p[1]
        for _, row in frame.iterrows():
            cla, ide, _, tl, br = row.items()
            name = "\\" + cla[1] + "\\patch_id" + str(ide[1]) + ".png"
            source1 = sou + name
            target1 = tar
            pic = cv.imread(source1)
            result = paint_padding(pic, tl[1], br[1])
            cv.imwrite(target1 + "\\" + cla[1] + "_" + str(ide[1]) + ".png", result)
