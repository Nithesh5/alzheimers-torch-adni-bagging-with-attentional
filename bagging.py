import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

torch.cuda.empty_cache()

if __name__ == "__main__":

    # path to load the labels and images
    CSV_FILE = '../All_Data.csv'
    ITERATIONS = 10
    scores = list()

    labels_df = pd.read_csv(CSV_FILE)
    whole_labels_df_AD = labels_df[labels_df.label == "AD"]
    whole_labels_df_CN = labels_df[labels_df.label == "CN"]

    # resampling the images order
    whole_labels_df_AD = whole_labels_df_AD.sample(frac=1, random_state=5)
    whole_labels_df_AD = whole_labels_df_AD.reset_index(drop=True)

    whole_labels_df_CN = whole_labels_df_CN.sample(frac=1, random_state=5)
    whole_labels_df_CN = whole_labels_df_CN.reset_index(drop=True)

    # Test dataset
    test_labels_df_AD = whole_labels_df_AD.iloc[0:50, :]  # 420:471
    test_labels_df_CN = whole_labels_df_CN.iloc[0:50, :]
    test_ds = test_labels_df_AD.append(test_labels_df_CN, ignore_index=True)

    # Train dataset
    labels_df_AD = whole_labels_df_AD.iloc[50:471, :]  # 0:420
    labels_df_CN = whole_labels_df_CN.iloc[50:650, :]

    data = labels_df_AD.append(labels_df_CN, ignore_index=True)

    train, val_ds = train_test_split(data, test_size=0.1)

    # saving train, validation, and test data in csv file for 10 different bootstrap versions.
    for i in range(ITERATIONS):
        train_ds = resample(train, n_samples=len(train))

        print("len of train and val and test")
        print(len(train_ds), len(val_ds), len(test_ds))

        train_ds.to_csv(str(i) + 'train_ds_new.csv')
        val_ds.to_csv(str(i) + 'val_ds_new.csv')
        test_ds.to_csv(str(i) + 'test_ds_new.csv')
