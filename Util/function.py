from sklearn.model_selection import KFold


def shuffle_set(emg1, emg3, emg4, emg6, y1, y3, y4, y6):
    train_list = [emg1, emg3, emg4, emg6]
    for train_index, test_index in train_list:
        print("test index: ", test_index)
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]