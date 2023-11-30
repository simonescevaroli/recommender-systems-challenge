import pandas as pd
import scipy.sparse as sps
import numpy as np
from Recommenders.MyRecommenders import *
from Evaluation.MyEvaluation import *
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample


def storeRecommendations(users_original, items_indices, filename='output.csv'):
    import csv
    item_original = [index_to_item_original_ID[it] for it in items_indices]
    with open(filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['user_id', 'item_list'])
        for user_id, items_list in zip(users_original, item_original):
            items_string = ' '.join(map(str, items_list))
            csv_writer.writerow([user_id, items_string])


# read data
csv = pd.read_csv('data_train.csv', sep=',')
users = csv['row'].unique()
items = csv['col'].unique()
users_csv = pd.read_csv('data_target_users_test.csv')
users_to_eval = users_csv["user_id"].values

# remove empty values
mapped_id, original_id = pd.factorize(csv["row"].unique())
user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

mapped_id, original_id = pd.factorize(csv["col"].unique())
item_original_ID_to_index = pd.Series(mapped_id, index=original_id)
index_to_item_original_ID = pd.Series(original_id, index=mapped_id)
csv["row"] = csv["row"].map(user_original_ID_to_index)
csv["col"] = csv["col"].map(item_original_ID_to_index)

# define URM
URM = sps.coo_matrix((csv["data"].values, (csv["row"].values, csv["col"].values)))   
URM_csr = URM.tocsr()

# split train-test sets
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)


# define and train recommender
# TODO


# recommend items to users
rec_items_ind = []
for user_id in users_to_eval:
    if(user_id in users):
        # recommend users for which we already have some ratings
        # TODO
        print("Nice recommendation!")
    else:
        # recommend users with no stored rating
        # TODO
        print("Nice recommendation!")


# evaluate recommender
# TODO


# store recommendations to ouput file        
storeRecommendations(users_to_eval, rec_items_ind)