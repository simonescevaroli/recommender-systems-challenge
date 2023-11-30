import numpy as np


# define top pop recommender
class TopPopRecommender(object):

    def fit(self, URM_train):
        self.URM_train = URM_train
        item_popularity = np.ediff1d(URM_train.tocsc().indptr)
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis = 0)
    
    def recommend(self, user_id, at=10, remove_seen=True):
        if remove_seen:
            row_start = self.URM_train.indptr[user_id]
            row_end = self.URM_train.indptr[user_id+1]
            seen_items = self.URM_train.indices[row_start:row_end]
            unseen_items_mask = np.in1d(self.popular_items, seen_items, assume_unique=True, invert = True)
            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popular_items[0:at]
        return recommended_items