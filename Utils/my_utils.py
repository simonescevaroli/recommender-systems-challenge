import pandas as pd
import scipy.sparse as sps
import numpy as np
from Recommenders.MyRecommenders import *
from Evaluation.MyEvaluation import *
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import scipy.sparse as sp

def newStoreRecommendations(users, items, filename='output.csv'):
    import csv
    with open(filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['user_id', 'item_list'])
        for user_id, items_list in zip(users, items):
            items_string = ' '.join(map(str, items_list))
            csv_writer.writerow([user_id, items_string])
    print("Recommendations stored in " + str(filename))


def readData():
    csv = pd.read_csv('data_train.csv', sep=',')
    users = csv['row'].unique()
    items = csv['col'].unique()
    users_csv = pd.read_csv('data_target_users_test.csv')
    users_to_eval = users_csv["user_id"].values
    return users, items, users_to_eval


def preProcess():
    csv = pd.read_csv('data_train.csv', sep=',')
    mapped_id, original_id = pd.factorize(csv["row"].unique())
    user_originalID_to_index = pd.Series(mapped_id, index=original_id)

    mapped_id, original_id = pd.factorize(csv["col"].unique())
    item_originalID_to_index = pd.Series(mapped_id, index=original_id)
    item_index_to_originalID = pd.Series(original_id, index=mapped_id)
    csv["row"] = csv["row"].map(user_originalID_to_index)
    csv["col"] = csv["col"].map(item_originalID_to_index)

    # define URM
    URM = sps.coo_matrix((csv["data"].values, (csv["row"].values, csv["col"].values)))   
    return URM, user_originalID_to_index, item_originalID_to_index, item_index_to_originalID

def newPreProcess():
    csv = pd.read_csv('data_train.csv', sep=',')
    # define URM
    URM = sps.coo_matrix((csv["data"].values, (csv["row"].values, csv["col"].values)))   
    return URM


def splitURM(URM):
    URM_train_full, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_full, train_percentage = 0.80)
    return URM_train_full, URM_train, URM_test, URM_validation


def MAP10(URM_test, recommender):
    from Evaluation.Evaluator import EvaluatorHoldout
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_df, _ = evaluator_test.evaluateRecommender(recommender)
    print("MAP10 = " + str(result_df.loc[10]["MAP"]))

class SaveResults(object):
    
    def __init__(self):
        self.results_df = pd.DataFrame(columns=["result"])
    
    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
        
        self.results_df = self.results_df.append(hyperparam_dict, ignore_index=True)