import numpy as np

def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score

def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score

def AP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    ap_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return ap_score


def merge_recommendations(rec1, rec2):
    merged_list = []
    merged_item = []
    for item1, item2 in zip(rec1, rec2):
        
        if item1 not in merged_item:
            merged_item.append(item1)
        if item2 not in merged_item:
            merged_item.append(item2)
            
    merged_list.append(merged_item[:10])
    return merged_list


def evaluate_algorithm(URM_test, recommender_object, at=10):
    cumulative_AP = 0.0 
    num_eval = 0

    for user_id in range(URM_test.shape[0]):
        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id+1]]
        if len(relevant_items)>0:
            recommended_items = recommender_object.recommend(user_id, cutoff=at)
            num_eval+=1
            cumulative_AP += AP(recommended_items, relevant_items)
            
    MAP = cumulative_AP / num_eval
    print("MAP = {:.4f}".format(MAP)) 


def evaluate_recommendations(URM_test, recommender_object1, recommender_object2, at=10):
    cumulative_AP = 0.0 
    num_eval = 0

    for user_id in range(URM_test.shape[0]):
        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id+1]]
        if len(relevant_items)>0:
            num_eval+=1
            recommended_items1 = recommender_object1.recommend(user_id, cutoff=at)
            recommended_items2 = recommender_object2.recommend(user_id, cutoff=at)
            recommended_items = merge_recommendations(recommended_items1, recommended_items2)
            cumulative_AP += AP(recommended_items, relevant_items)
            
    MAP = cumulative_AP / num_eval
    print("MAP = {:.4f}".format(MAP)) 
