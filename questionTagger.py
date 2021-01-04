import joblib
import os
from dataset import SOFData
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support as scores
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
import time
import datetime

N_CPU = 48 # -1 for using all CPUs


class Model(object):

    kind = ('sgd', )

    def __init__(self, kind='', nj=N_CPU):
        
        assert kind in Model.kind, \
            f"{kind} does not exist. Choose from {Model.kind}"
        
        self.nj = nj
        self.model = None
        self.model_root = './models'
        print("="*60)
        print("Creating model ...")
        if kind == 'sgd': 
            self.sgd_clf = SGDClassifier(n_jobs = self.nj)
            self.model = OneVsRestClassifier(self.sgd_clf)
            print(f"Classifier: {self.sgd_clf}")
            print(f"Model: {self.model}")
            print("="*60)

    def train(self, X, Y, time_it = True, save = True, kw=""):
        print("="*60)
        print(f"Training on {self.nj} CPUs ... ")
        self.tr_start_t = time.time()
        self.model.fit(X, Y)                    
        self.tr_end_t = time.time() - self.tr_start_t
        print("Training time: {} minutes.".format(self.tr_end_t/60))        
        print("="*60)

        if save:
            print("Saving model as", end=" ") 
            if not os.path.exists(self.model_root):
                os.system('mkdir {}'.format(self.model_root))
            self.timestamp = '_'.join(str(datetime.datetime.now()).split(' '))
            self.m_path = f"{self.model_root}/OvR_classifier_{self.timestamp}_{kw}.pkl" 
            joblib.dump(self.model, self.m_path)
            print(f"{self.m_path}.")

    @staticmethod    
    def load_model(model_path):
        print(f"Loading model from {model_path} ...")
        return joblib.load(model_path)    

    @staticmethod
    def val(X, Y, model_path, time_it = True, nj=-1): 
            
        model = Model.load_model(model_path)
 
        print("="*60)
        print("Cross validating ... ") 
        
        # prepare the cross-validation procedure
        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        scores = {'Precision' : "precision_micro",
                  'Recall' : "recall_micro", 
                  'F1_score' : "f1_micro"} 
        vl_start_t = time.time()
        # evaluate model
        results = cross_validate(model, X, Y, scoring=scores, cv=cv, n_jobs=nj) 
        vl_end_t = time.time() - vl_start_t 
        print("Validation time: {} minutes.".format(vl_end_t/60))
        
        print("Mean metrics:")
        for stype, score in results.items():
            print(f"{stype}: {np.mean(score)}")  
        print("="*60)
        
        return results

    @staticmethod
    def test(Xte, Yte, model_path=None, time_it = True): 
        if model_path:
            model = Model.load_model(model_path)
        else:
            model = Model.load_model(self.m_path)
 
        print("="*60)
        print("Testing ... ")
        te_start_t = time.time()
        Ypr = model.predict(Xte)
        te_end_t = time.time() - te_start_t
        print("Testing time: {} minutes.".format(te_end_t/60))    
        
        P, R, F, _ = scores(Yte, Ypr)
        #P, R, F = np.mean(P), np.mean(R), np.mean(F)
        #print(f"Precision:\t{P}")
        #print(f"Recall:\t{R}")
        #print(f"F1-score:\t{F}")        
        print("="*60)
        return (P, R, F) 

def main():
    print("Loading data ... ")
    dset = SOFData(root = './data')
    dset.get_data(train=True, test=True)
    print(dset.tr_data)
    print(dset.te_data)
 

    model = Model(kind="sgd") 
    model.train(dset.tr_data['X'], dset.tr_data['Y'], kw="CV")   

    path = model.m_path 
    #path = f"./models/OvR_classifier_2020-12-12_20:39:20.098594.pkl"

    results =  Model.val(dset.tr_data['X'], dset.tr_data['Y'], path, nj=N_CPU)

    P, R, F = Model.test(dset.te_data['X'], dset.te_data['Y'], path) 
    
    metrics_df = pd.DataFrame(data=[P, R, F],
                              index=["Precision", "Recall", "F-1 score"],
                              columns=dset.cl_data)
    print(metrics_df)
    metrics_df.to_csv(path[:-4] + '_test.csv')


if __name__ == "__main__":
    main()





    












   






         
