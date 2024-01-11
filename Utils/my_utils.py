import pandas as pd
import scipy.sparse as sps
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

def storeRecommendations(users, items, filename='output.csv'):
    import csv
    with open(filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['user_id', 'item_list'])
        for user_id, items_list in zip(users, items):
            items_string = ' '.join(map(str, items_list))
            csv_writer.writerow([user_id, items_string])
    print("Recommendations stored in " + str(filename))


def readData():
    csv = pd.read_csv('Dataset/data_train.csv', sep=',')
    users = csv['row'].unique()
    items = csv['col'].unique()
    users_csv = pd.read_csv('Dataset/data_target_users_test.csv')
    users_to_eval = users_csv["user_id"].values
    return users, items, users_to_eval

def preProcess():
    csv = pd.read_csv('Dataset/data_train.csv', sep=',')
    URM = sps.coo_matrix((csv["data"].values, (csv["row"].values, csv["col"].values)))   
    return URM