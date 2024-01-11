from Recommenders.Hybrids.Hybrid_SLIM_RP3beta import Hybrid_SLIM_RP3beta
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import pandas as pd

class SaveResults(object):
    
    def __init__(self):
        self.results_df = pd.DataFrame(columns=["result"])
    
    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
        
        self.results_df = self.results_df.append(hyperparam_dict, ignore_index=True)


class TrainHyperparams(object):

    def __init__(self, URM_train, evaluator_validation):
        self.URM_train = URM_train
        self.evaluator = evaluator_validation
    
    def objective_function_hybrid(self, optuna_trial):
        recommender_object = Hybrid_SLIM_RP3beta(self.URM_train)
        recommender_object.fit(alpha = optuna_trial.suggest_float("alpha", 0, 1),
                                beta = optuna_trial.suggest_float("beta", 0, 1),
                                topK = optuna_trial.suggest_int("topK", 1, 1500))
   
        result_df, _ = self.evaluator.evaluateRecommender(recommender_object)
        
        return result_df.loc[10]["MAP"]

    def objective_function_SLIM(self, optuna_trial):
        recommender_object = SLIMElasticNetRecommender(self.URM_train)
        recommender_object.fit(l1_ratio = optuna_trial.suggest_float("l1_ratio", 1e-4, 0.1, log=True),
                                alpha = optuna_trial.suggest_float("alpha", 1e-3, 0.01),  
                                topK = optuna_trial.suggest_int("topK", 0, 999))
            
        result_df, _ = self.evaluator.evaluateRecommender(recommender_object)
        
        return result_df.loc[10]["MAP"]

    def objective_function_Rp3beta(self, optuna_trial):
        recommender_object = RP3betaRecommender(self.URM_train)
        recommender_object.fit(alpha = optuna_trial.suggest_float("alpha", 0, 2),
                                beta = optuna_trial.suggest_float("beta", 0, 2),
                                topK = optuna_trial.suggest_int("topK", 5, 1000))
            
        result_df, _ = self.evaluator.evaluateRecommender(recommender_object)
        
        return result_df.loc[10]["MAP"]


