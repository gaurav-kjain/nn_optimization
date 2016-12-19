import sys
sys.path.append("dnn_optimization")
from ModelRunner import *
db="MNIST"
bmode="outliers_iqr"
#run_experiment_config(db, bmode=bmode,maindir="/home/gaurav.kjain/Thesis/Results",bmult=4,verbose=1)
run_all_config(db, maindir="/home/gaurav.kjain/Thesis/Results",bmult=4,verbose=1)