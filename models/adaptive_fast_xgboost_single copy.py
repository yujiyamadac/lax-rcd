import numpy as np
import xgboost as xgb

from skmultiflow.utils import get_dimensions
from sklearn.neighbors import KNeighborsRegressor
from skmultiflow.drift_detection import ADWIN
from statistics import mean, median, stdev

from models.remove_tree import rmv_tree

xgb.set_config(verbosity=0)

class AdaptiveFastSingle():
    

    def __init__(self,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 ratio_unsampled=0,
                 small_window_size=0,
                 trees_per_train=1,
                 percent_update_trees = 1.0,
                 max_trees = 40,
                 objective="reg",
                 detect_drift=False,
                 update_strategy="fifo",
                 pre_train=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.detect_drift = detect_drift
        self.update_strategy = update_strategy
        self.pre_train = pre_train
        self.batch_size = 0
        self._first_run = True
        self._booster = None
        self._drift_detector = None
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])

        self._max_trees = max_trees
        self.trees_per_train = trees_per_train
        self.percent_update_trees = percent_update_trees

        self._ratio_unsampled = ratio_unsampled
        self._X_small_buffer = np.array([])
        self._y_small_buffer = np.array([])
        self._samples_seen = 0
        self._model_idx = 0
        self._small_window_size = small_window_size
        self._count_buffer = 0

        self._count_trees = 0
        self.window_size = 0

        self._fingerprint_model = {}

        if objective == "reg":
            self.objective = "reg:squarederror"
            self.eval_metric = "rmse"
        elif objective == "class":
            self.objective = "binary:logistic"
            self.eval_metric = "logloss"
        elif objective == "mclass":
            self.objective = "multi:softmax"
            self.eval_metric = "mlogloss"

        self._configure()

    def _configure(self):
        self._reset_window_size()
        self._init_margin = 0.0
        self._boosting_params = {
            "objective": self.objective,
            "eta": self.learning_rate,
            "eval_metric": self.eval_metric,
            "max_depth": self.max_depth}

        self._boosting_params_update = {
            "objective": self.objective,
            "eta": self.learning_rate,
            "eval_metric": self.eval_metric,
            "max_depth": self.max_depth,
            "process_type": "update",
            "updater": "refresh"}
        
        if self.detect_drift:
            self._drift_detector = ADWIN()
        
        self._drift_detector_array = []

    def reset(self):
        self._first_run = True
        self._configure()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.batch_size = X.shape[0]
        if self.pre_train:
            self._reset_window_size()
        
        for i in range(X.shape[0]):
            self._partial_fit(np.array([X[i, :]]), np.array([y[i]]))
        return self

    def _change_small_window(self, npArrX, npArrY):
        if npArrX.shape[0] < self._small_window_size:
            sizeToRemove = 0
            nextSize = self._X_small_buffer.shape[0] + npArrX.shape[0]
            if nextSize > self._small_window_size:
                sizeToRemove = nextSize - self._small_window_size
            #deleta os dados velhos
            delete_idx = [i for i in range(sizeToRemove)]

            if len(delete_idx) > 0:
                self._X_small_buffer = np.delete(self._X_small_buffer, delete_idx, axis=0)
                self._y_small_buffer = np.delete(self._y_small_buffer, delete_idx, axis=0)
            
            self._X_small_buffer = np.concatenate((self._X_small_buffer, npArrX))
            self._y_small_buffer = np.concatenate((self._y_small_buffer, npArrY))
        else:
            self._X_small_buffer = npArrX[0:self._small_window_size]
            self._y_small_buffer = npArrY[0:self._small_window_size]

    def _partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_buffer = np.array([])
            self._X_small_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_small_buffer = np.array([])
            self._first_run = False
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))

        while self._X_buffer.shape[0] >= self.window_size:
            self._count_buffer = self._count_buffer + 1
            npArrX, npArrY = self._X_buffer, self._y_buffer

            if npArrX.shape[0] > 0:
                self._train_on_mini_batch(X=npArrX,
                                        y=npArrY)
                                    
            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)

            # Check window size and adjust it if necessary
            self._adjust_window_size()

        # Support for concept drift
        if self.detect_drift and self._drift_detector is not None and self.update_strategy == "fifo":
            error = abs(self.predict(X) - y)  
            self._drift_detector.add_element(float(error))

            if self._drift_detector.detected_change():
                #print("Drift detected! Resetting window")
                self._reset_window_size()

    def _adjust_window_size(self):
        if self._dynamic_window_size != self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self.window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.pre_train:
            self._dynamic_window_size = self.batch_size
        elif self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self.window_size = self._dynamic_window_size
        #print('Window size aft reset'+ str(self.window_size))

    def _train_on_mini_batch(self, X, y):
        
        booster = self._train_booster(X, y, self._booster)
        fingerprint = []
        means = []
        for col in range(X.shape[1]):
                        means.append(mean(X[:, col]))
        self._fingerprint_model[booster] = booster.get_score()
        # Update ensemble
        self._booster = booster

    def _train_booster(self, X: np.ndarray, y: np.ndarray, currentBooster):
        d_mini_batch_train = xgb.DMatrix(X, y)

        if currentBooster:
            new_trees = 0
            prev_num_boosted_rounds = currentBooster.num_boosted_rounds()

            if self.update_strategy == "fifo":

                booster: xgb.Booster = currentBooster[int(prev_num_boosted_rounds * (1 - self.percent_update_trees)):]

            elif self.update_strategy == "target":

                booster = currentBooster
                #booster: xgb.Booster = currentBooster[int(prev_num_boosted_rounds * (1 - self.percent_update_trees)):]
                #self._drift_detector_array = self._drift_detector_array[int(prev_num_boosted_rounds * (1 - self.percent_update_trees)):]

                #for idx, tree in reversed(list(enumerate(currentBooster))):
                for idx, tree in list(enumerate(booster)):
                    #if idx < booster.num_boosted_rounds()/2:
                    if idx < booster.num_boosted_rounds():
                        error = abs(tree.inplace_predict(X) - y)
                        #print(error) 
                        #print('Checking tree ' + str(idx))  
                        
                        for e in error:
                            self._drift_detector_array[idx].add_element(float(e))

                            #print("Drift detected! Delete until tree ", idx)
                            #self._count_trees = self._count_trees + 1
                            #print("Replaced tree "+ str(idx) + ". Total: " + str(self._count_trees) + " trees")
                            if self._drift_detector_array[idx].detected_change():
                                #print("Drift detected! Delete until tree ", idx)

                                self._reset_window_size() # tb resetar a janela
                                #diminuir o tamanho da janela, não resetar

                                if idx != booster.num_boosted_rounds()-1:
                                    #print('Removing tree ' + str(idx))
                                    booster.save_model('target.json')

                                    rmv_tree('target.json', idx)

                                    del self._drift_detector_array[idx]
                                    booster.load_model('target.json')

                                #print('Removing before tree ' + str(idx))
                                #self._drift_detector_array = self._drift_detector_array[(idx+1):]
                                #booster: xgb.Booster = booster[(idx+1):]
                                
                                break

                        else:
                            continue
                        
                        #break
            
            else:

                booster = currentBooster

                for idx, tree in enumerate(currentBooster):
                    error = abs(tree.inplace_predict(X) - y)
                    #print(error)  
                    
                    for e in error: #fazer pela média do erro
                        self._drift_detector_array[idx].add_element(float(e))

                        if self._drift_detector_array[idx].detected_change():
                            #print("Drift detected! Delete until tree ", idx)

                            self._reset_window_size() # tb resetar a janela

                            if idx != currentBooster.num_boosted_rounds()-1:
                                booster: xgb.Booster = currentBooster[(idx+1):]
                                del self._drift_detector_array[:idx]

                            break

                    else:
                        continue
                    
                    break

        # ATUALIZAR ÁRVORES REMANESCENTES
            # booster = xgb.train(
            #     params=self._boosting_params_update,
            #     dtrain=d_mini_batch_train,
            #     num_boost_round=booster.num_boosted_rounds(),
            #     xgb_model=booster,
            # )
                                    
            if prev_num_boosted_rounds >= self._max_trees:
                new_trees = prev_num_boosted_rounds - booster.num_boosted_rounds()
            else:
                new_trees = prev_num_boosted_rounds - booster.num_boosted_rounds() + self.trees_per_train

            #if prev_num_boosted_rounds >= self._max_trees:
            #    new_trees = 0 + int(prev_num_boosted_rounds * (1 - self.percent_update_trees))
            #else:
            #    new_trees = self.trees_per_train + int(prev_num_boosted_rounds * (1 - self.percent_update_trees))

            booster = xgb.train(
                params=self._boosting_params,
                dtrain=d_mini_batch_train,
                num_boost_round=new_trees,
                xgb_model=booster,
            )

            for i in range(new_trees):
                self._drift_detector_array.append(self._drift_detector)
        
        elif self.pre_train:
            booster = xgb.train(params=self._boosting_params,
                                dtrain=d_mini_batch_train,
                                num_boost_round=self._max_trees,
                                verbose_eval=False)
            self.pre_train = False
        
        else:
            booster = xgb.train(params=self._boosting_params,
                                dtrain=d_mini_batch_train,
                                num_boost_round=self.trees_per_train,
                                verbose_eval=False)

            if self.update_strategy != "fifo":
                for tree in booster:
                    self._drift_detector_array.append(self._drift_detector)


        #print(booster.num_boosted_rounds())
        return booster

    def predict(self, X):
        if self._booster:            
            predicted = self._booster.inplace_predict(X)    
            return predicted
        # Ensemble is empty, return default values (0)
        return np.zeros(get_dimensions(X)[0])

    def predict_proba(self, X):
        """
        Not implemented for this method.
        """
        raise NotImplementedError(
            "predict_proba is not implemented for this method.")
