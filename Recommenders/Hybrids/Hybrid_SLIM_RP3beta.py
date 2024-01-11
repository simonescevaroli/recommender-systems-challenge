#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrids.BaseHybridSimilarity import BaseHybridSimilarity
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

output_root_path = "./result_experiments/"


class Hybrid_SLIM_RP3beta(BaseHybridSimilarity):
    RECOMMENDER_NAME = "Hybrid_SLIM_RP3beta"

    def __init__(self, URM_train):
        slim = SLIMElasticNetRecommender(URM_train)
        slim.load_model(output_root_path, file_name="SLIMEN_best.zip")
        #slim.fit(l1_ratio=0.06500647156021869, alpha=0.0010484127187467302, topK=549)

        rp3 = RP3betaRecommender(URM_train)
        rp3.load_model(output_root_path, file_name="RP3beta_best.zip")
        #rp3.fit(topK=36, alpha=0.28449935654625247, beta=0.11366112485546402, normalize_similarity=True)

        super(Hybrid_SLIM_RP3beta, self).__init__(URM_train, slim, rp3)