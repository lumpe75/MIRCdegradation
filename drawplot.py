import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from pretrained_models import run_comparison_rcnn

bw_data = [
    ["patch_id109", "bike", 0.70],
    ["patch_id147", "bike", 0.63],
    ["patch_id89", "bike", 0.52],
    ["patch_id22", "car", 0.80],
    ["patch_id32", "car", 0.65],
    ["patch_id60", "car", 0.65],
    ["patch_id135", "eagle", 0.58],
    ["patch_id15", "eagle", 0.72],
    ["patch_id28", "eagle", 0.79],
    ["patch_id132", "glasses", 0.61],
    ["patch_id149", "glasses", 0.56],
    ["patch_id24", "glasses", 0.65],
    ["patch_id1", "horse", 0.93],
    ["patch_id30", "horse", 0.52],
    ["patch_id94", "horse", 0.55],
    ["patch_id113", "plane", 0.58],
    ["patch_id136", "plane", 0.68],
    ["patch_id85", "plane", 0.55],
    ["patch_id111", "ship", 0.63],
    ["patch_id19", "ship", 0.57],
    ["patch_id2", "ship", 0.54],
    ["patch_id101", "suit", 0.64],
    ["patch_id130", "suit", 0.53],
    ["patch_id54", "suit", 0.52]
]


def create_recognition_drop_humans():
    participants_df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\participant_results.csv", sep=",")
    participants_df = participants_df[['OldID', 'Class', 'Average', "degradation"]]
    old_values = pd.DataFrame(bw_data, columns=['OldID', 'Class', 'Old_Average'])
    participants_df = participants_df.merge(old_values, right_on=['Class', 'OldID'], left_on=['Class', 'OldID'])
    participants_df["drop"] = participants_df["Old_Average"] - participants_df["Average"]
    return participants_df


def create_recognition_drop_model(model="refitted"):
    old_mircs = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\"+model+"_results_originals.csv", sep=",")
    new_mircs = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\"+model+"_results.csv", sep=",")
    reduced = pd.DataFrame(bw_data, columns=['id', 'real_class', 'Old_Average'])
    participants_df = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\participant_results.csv", sep=",")
    participants_df = participants_df[['OldID', 'Class', 'degradation']]
    reduced = reduced.merge(participants_df, left_on=['id', 'real_class'], right_on=['OldID', 'Class'])
    reduced['id'] = reduced['id'].str.replace(r'^patch_', '', regex=True)

    old_mircs['old_confidence'] = old_mircs.apply(lambda row: row[f"{row['real_class']}_confidence"], axis=1)
    old_mircs = old_mircs[["old_confidence", "id", "real_class"]]
    reduced = reduced.merge(old_mircs, left_on=['id', 'real_class'], right_on=['id', 'real_class'])

    replacement_dict = {
        "01_flipvertical_mircs": "flip",
        "04_inverse_mircs": "inverse",
        "06_texture_mircs": "texture"
    }
    new_mircs['effect'] = new_mircs['effect'].replace(replacement_dict)

    reduced = reduced.merge(new_mircs, left_on=['id', 'real_class', 'degradation'], right_on=['id', 'real_class', 'effect'], how="left")

    reduced['new_confidence'] = reduced.apply(lambda row: row[f"{row['real_class']}_confidence"], axis=1)

    reduced = reduced[["id", "real_class", "new_confidence", "old_confidence", "effect"]]
    #new_reduced = new_reduced[["id", "real_class", "new_confidence", "effect"]]
    #combined = old_reduced.merge(new_reduced, left_on=['id', 'real_class'], right_on=['id', 'real_class'])
    reduced["machine_drop"] = reduced["old_confidence"] - reduced["new_confidence"]
    return reduced


def create_recognition_drop_model_full(model="refitted"):
    old_mircs = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\"+model+"_results_originals.csv", sep=",")
    new_mircs = pd.read_csv("C:\\Users\\Lumpe\\Synced\\_CogCoVI\\results\\"+model+"_results.csv", sep=",")

    replacement_dict = {
        "01_flipvertical_mircs": "flip",
        "04_inverse_mircs": "inverse",
        "06_texture_mircs": "texture"
    }
    new_mircs['effect'] = new_mircs['effect'].replace(replacement_dict)

    old_reduced = old_mircs
    new_reduced = new_mircs

    old_reduced['old_confidence'] = old_reduced.apply(lambda row: row[f"{row['real_class']}_confidence"], axis=1)
    new_reduced['new_confidence'] = new_reduced.apply(lambda row: row[f"{row['real_class']}_confidence"], axis=1)

    old_reduced = old_reduced[["id", "real_class", "old_confidence"]]
    new_reduced = new_reduced[["id", "real_class", "new_confidence", "effect"]]
    combined = old_reduced.merge(new_reduced, left_on=['id', 'real_class'], right_on=['id', 'real_class'])
    combined["machine_drop"] = combined["old_confidence"] - combined["new_confidence"]
    combined = combined.rename(columns={"effect": "degradation"})
    return combined


def draw_scatter_plot():
    human_data = create_recognition_drop_humans()
    machine_data = create_recognition_drop_model()
    x = human_data["Average"]
    y = machine_data["new_confidence"]
    plt.scatter(x, y)
    plt.xlabel('Human Recognition Rate')
    plt.ylabel('Model Recognition Rate')
    plt.title('Scatter Plot vs Human Bagnet')
    plt.show()


def draw_scatter_plot_full_mircs():
    bw_frame = pd.DataFrame(bw_data, columns=["id", "real_class", "confidence"])
    x = bw_frame["confidence"]
    machine_data = create_recognition_drop_model()
    y = machine_data["old_confidence"]
    plt.scatter(x, y)
    plt.xlabel('Human Recognition Rate')
    plt.ylabel('Model Recognition Rate')
    plt.title('Scatter Plot MIRCs Human vs ResNet')
    plt.show()


def draw_linear_scatter_plot():
    human_data = create_recognition_drop_humans()
    machine_data = create_recognition_drop_model()
    human = human_data["Average"]
    machine = machine_data["new_confidence"]
    n = len(human)
    x = np.arange(n)
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, human, color='b', label='human', marker='.')
    plt.scatter(x, machine, color='r', label='machine', marker='.')

    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Recognition Threshold')
    # Draw vertical dotted lines between pairs of points
    for i in range(n):
        plt.vlines(x[i], min(human[i], machine[i]), max(human[i], machine[i]), color='gray', linestyle=':', linewidth=1)

    # Add some text for labels, title, and legend
    plt.xlabel('Pair Index')
    plt.ylabel('Score')
    plt.title('Comparison Picture Wise Bagnet')
    plt.legend()

    plt.show()

def draw_recog_comparison_bar_chart8():
    human_data = create_recognition_drop_humans()
    #machine_data = create_recognition_drop_model()
    machine_data = create_recognition_drop_model_full()
    bins = np.arange(-1, 1.1, 0.1)
    labels = [f'{round(b, 1)}' for b in bins[:-1]]
    # Number of categories
    n = len(labels)
    # Create an array with the positions for the categories
    ind = np.arange(n)
    # Width of the bars
    width = 0.35
    human_data['drop_bin'] = pd.cut(human_data['drop'], bins=bins, labels=labels, include_lowest=True)
    machine_data['machine_drop_bin'] = pd.cut(machine_data['machine_drop'], bins=bins, labels=labels, include_lowest=True)
    degradation_levels = human_data['degradation'].unique()
    for i, deg in enumerate(degradation_levels):
        h_data = human_data[human_data["degradation"] == deg]
        m_data = machine_data[machine_data["degradation"] == deg]
        h_counts = h_data['drop_bin'].value_counts().sort_index().values
        m_counts = m_data['machine_drop_bin'].value_counts().sort_index().values
        m_average = m_data["machine_drop"].mean()
        h_average = h_data["drop"].mean()
        # Side-by-Side Bar Chart
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create bars for the first set of values
        bars1 = ax.bar(ind - width / 2, h_counts, width, label='Human')

        # Create bars for the second set of values
        bars2 = ax.bar(ind + width / 2, m_counts, width, label='Machine')
        plt.axvline(x=h_average*10+10, color='blue', linestyle='-', linewidth=2, label=f'Average Human: {h_average:.2f}')
        plt.axvline(x=m_average*10+10, color='orange', linestyle='-', linewidth=2, label=f'Average Machine: {m_average:.2f}')
        # Add some text for labels, title, and axes ticks
        ax.set_xlabel('Buckets')
        ax.set_ylabel('Frequency')
        ax.set_ylim(0, 40)
        ax.set_title('Recognition Gap ResNet for ' + deg)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels, rotation=90)
        ax.legend()

        plt.show()


def calculate_relevance_all():
    human_data = create_recognition_drop_humans()
    human_data = list(human_data['drop'])

    print("ResNet")
    machine_data = create_recognition_drop_model()
    machine_data = list(machine_data['machine_drop'])

    stat, p_value = stats.ttest_ind(human_data, machine_data)

    print("Reduced Data")
    print('Statistics:', stat)
    print('p-value:', p_value)

    machine_data = create_recognition_drop_model_full()
    machine_data = list(machine_data['machine_drop'])

    stat, p_value = stats.ttest_ind(human_data, machine_data)
    print("Full Data")
    print('Statistics:', stat)
    print('p-value:', p_value)

    print("=====================")
    print("BagNet")
    machine_data = create_recognition_drop_model(model="bagnet")
    machine_data = list(machine_data['machine_drop'])

    stat, p_value = stats.ttest_ind(human_data, machine_data)

    print("Reduced Data")
    print('Statistics:', stat)
    print('p-value:', p_value)

    machine_data = create_recognition_drop_model_full(model="bagnet")
    machine_data = list(machine_data['machine_drop'])

    stat, p_value = stats.ttest_ind(human_data, machine_data)
    print("Full Data")
    print('Statistics:', stat)
    print('p-value:', p_value)

if __name__ == "__main__":
    run_comparison_rcnn()