import pandas as pd
import matplotlib.pyplot as plt

# full name of each class
name_dict = {
    "bike": "bicycle-built-for-two",
    "car": "sports car",
    "eagle": "bald eagle",
    "fly": "fly",
    "glasses": "sunglass",
    "horse": "sorrel",
    "plane": "airliner",
    "ship": "aircraft carrier",
    "suit": "suit"
}

# list of real participants IDs
participants = [61303, 80807, 81251, 88028, 142132, 265849, 269406, 277651, 361420, 427666, 428462, 483133, 538855,
                544660, 546579, 561124, 646799, 660266, 667470, 678171, 703908, 706696, 732075, 799458, 867624,
                873113, 897171, 917296, 959667, 983443]


def combine_model_results():
    """
    Combines the results of the 3 models Resnet, AlexNet and BagNet into a single csv
    """
    index_dict = {
        "bike": 444,
        "car": 817,
        "eagle": 22,
        "fly": 308,
        "glasses": 836,
        "horse": 339,
        "plane": 404,
        "ship": 403,
        "suit": 834
    }
    full_file = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\alexnet_results.csv")
    full_file = full_file[["effect", "real_class", "id"]]
    full_file["real_class"] = full_file["real_class"].map(name_dict)
    files = ["alexnet", "bagnet", "resnet"]
    for model in files:
        df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\" + model + "_results.csv")
        df["indexing"] = df["real_class"].map(index_dict)
        df["real_class"] = df["real_class"].map(name_dict)
        tp_confidence = []
        class_guessed = []
        class_conf = []
        for index, row in df.iterrows():
            guessed_class_index = row["indexing"] * 2 + 3  # offset for first 3 columns
            guessed_class_confidence = df.iloc[index, guessed_class_index + 1]
            tp_confidence.append(guessed_class_confidence)
            max_value = float("-inf")
            max_col = "fail"
            for col in row.index:
                value = row[col]
                if isinstance(value, float) and value > max_value:
                    max_value = value
                    max_col = col
            class_conf.append(max_value)
            class_guessed.append(max_col.split("_")[0])
        full_file["tp confidence " + model] = tp_confidence
        full_file["class guessed " + model] = class_guessed
        full_file["guess confidence " + model] = class_conf
        full_file["hit" + model] = full_file["real_class"] == full_file["class guessed " + model]
    full_file.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\model_results.csv", sep=",", index=False)


def filter_experiment_results():
    """
    Splits the full pavlovia result file into one csv file per participant containing only the relevant information
    """
    data = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\expdata_1905.csv")
    test_columns = [
        'degradationtrials.thisIndex',
        'degrade_name',
        'horse_button.numClicks',
        'dont_button.numClicks',
        'suit_button.numClicks',
        'glasses_button.numClicks',
        'plane_button.numClicks',
        'duck_button.numClicks',
        'eagle_button.numClicks',
        'car_button.numClicks',
        'bike_button.numClicks',
        'ship_button.numClicks']

    control_columns = [
        'trials.thisIndex',
        'mirc_name',
        'horse_button_2.numClicks',
        'dont_button_2.numClicks',
        'suit_button_2.numClicks',
        'glasses_button_2.numClicks',
        'plane_button_2.numClicks',
        'duck_button_2.numClicks',
        'eagle_button_2.numClicks',
        'car_button_2.numClicks',
        'bike_button_2.numClicks',
        'ship_button_2.numClicks']

    grouped = data.groupby('participant')
    dfs = {}
    participant_list = []
    for name, group in grouped:
        participant_list.append(name)
        dfs[name] = group
    for p in participant_list:
        df = dfs[p]
        df_test = df[test_columns].dropna()
        df_control = df[control_columns].dropna()
        df_test.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\participants\\test\\" + str(p) + ".csv",
                       sep=",", index=False)
        df_control.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\participants\\control\\" + str(p) + ".csv",
                          sep=",", index=False)


def check_participants():
    """
    Checks if all participants managed to recognize all control images
    """
    for p in participants:
        df_full = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\participants\\control\\" + str(p) + ".csv")
        df = df_full.drop(columns=['trials.thisIndex', 'mirc_name'])
        numpy_array = df.to_numpy()
        idx = []
        for row_idx, row in enumerate(numpy_array):
            for col_idx, val in enumerate(row):
                if val == 1:
                    idx.append(col_idx)
                    numpy_array[:, col_idx] -= 1
        order = [str(df.columns[i]).split("_")[0] for i in idx]
        real = list(df_full["mirc_name"])
        real = [item.split("_")[0] for item in real]
        check = set(real) == set(order)
        check = "passed" if check else "failed"
        print("Participant " + str(p) + " has " + check + " the control")


def get_test_overview():
    """
    Combines the individual csv files of 'filter_experiment_results' into a single result file
    """
    key = pd.read_excel("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\experiment.xlsx")
    true_class = list(key["Class"])
    second_df = pd.DataFrame()
    for p in participants:
        df_full = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\participants\\test\\" + str(p) + ".csv")
        sequence = list(df_full["degradationtrials.thisIndex"])
        df = df_full.drop(columns=['degradationtrials.thisIndex', 'degrade_name'])
        numpy_array = df.to_numpy()
        idx = []
        for row_idx, row in enumerate(numpy_array):
            check = False
            for col_idx, val in enumerate(row):
                if val == 1:
                    idx.append((row_idx, col_idx))
                    numpy_array[:, col_idx] -= 1
                    check = True
            if not check:
                idx.append((row_idx, -1))
        df["noclick"] = "skipped"
        order = [str(df.columns[i[1]]).split("_")[0] for i in idx]
        zipped_lists = list(zip(sequence, order))
        sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[0])
        sorted_list_strings = [element[1] for element in sorted_zipped_lists]
        second_df[str(p)] = ([1 if a == b else 0 for a, b in zip(sorted_list_strings, true_class)])
    key['Total'] = second_df.iloc[:, :].sum(axis=1)
    key['Average'] = second_df.iloc[:, :].mean(axis=1)
    key = pd.concat((key, second_df), axis=1)
    key.to_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\participant_results.csv", sep=",", index=False)


def draw_line_matrix_classes():
    """
    Draws a line graph for each class, visualizing the recognition rate of the participants
    """
    df = pd.read_csv('C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\participant_results.csv')
    fig, axes = plt.subplots(2, 4, figsize=(15, 15))
    classes = df["Class"].unique().tolist()
    row = 0
    for i, c in enumerate(classes):
        x = i
        if i == 4:
            row = 1
        if i > 3:
            x -= 4
        filtered = df[df["Class"] == c]
        axes[row, x].set_ylim(0, 1.1)
        axes[row, x].plot(filtered["degradation"], filtered["Average"], color='black')
        axes[row, x].set_title(f'{c}')
    plt.tight_layout()
    plt.show()


def draw_line_matrix_effect():
    """
    Draws a line graph for each effect, visualizing the recognition rate of the participants
    """
    df = pd.read_csv('C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\participant_results.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 15))
    classes = df["degradation"].unique().tolist()
    for i, c in enumerate(classes):
        filtered = df[df["degradation"] == c]
        axes[i].set_ylim(0, 1.1)
        axes[i].plot(filtered["Class"], filtered["Average"], color='black')
        axes[i].set_title(f'{c}')
    plt.tight_layout()
    plt.show()
