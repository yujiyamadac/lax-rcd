import sys
sys.setrecursionlimit(10000)


from models.adaptive_fast_xgboost_single import AdaptiveFastSingle

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream

from river import drift
from river import rules
from river import preprocessing


# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
max_window_size = 1000  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = False    # Enable/disable drift detection
ratio_unsampled = 0
small_window_size = 150
max_buffer = 25
pre_train = 15
max_depth = 6           # Max depth for each tree in the ensemble
learning_rate = 0.3     # Learning rate or eta

max_samples = 500000

streams = [
    'elec',
]

dataset_unit = [24, 15*12, 6*24, 6*24, 24*100, 60*60, 60*60, 60*60, 60*60]
train_size = [90, 36, 90, 90, 30, 24, 24, 24, 24]

# refazer testes
for i, s in enumerate(streams):  

    AFXGBRegSingleTarget = AdaptiveFastSingle(learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    max_window_size=max_window_size,
                                    min_window_size=100,
                                    ratio_unsampled=ratio_unsampled,
                                    small_window_size=small_window_size,
                                    trees_per_train=5,
                                    max_trees=50,
                                    percent_update_trees=0.95,
                                    objective="class",
                                    pre_train=False,
                                    detect_drift=True,
                                    update_strategy = "target",
                                    history_size=50)
    
    AMRules = (
        preprocessing.StandardScaler() |
        rules.AMRules(
            drift_detector=drift.ADWIN()
        )
    )

    model = [
            AFXGBRegSingleTarget,
            ]
    
    model_name = [
            "LAX-T",
            ]

    stream = FileStream("datasets/{}.csv".format(s))
    
    for j, m in enumerate(model):  
        evaluator = EvaluatePrequential(n_wait=1,
                                        #pretrain_size=dataset_unit[i]*train_size[i],
                                        pretrain_size=0,
                                        max_samples=max_samples,
                                        batch_size=dataset_unit[i],
                                        #batch_size=1,
                                        show_plot=False,
                                        #metrics=["mean_square_error", "running_time"],
                                        metrics=["accuracy", "running_time", "model_size"],
                                        output_file=("results/{0}_{1}_size.csv".format(s,model_name[j]))
                                        )
        
        evaluator.evaluate(stream=stream,
                        model=[m],
                        model_names=[model_name[j]])