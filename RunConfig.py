#import sys
#sys.path.append("dnn_optimization")
#from ModelRunner1 import *
#db="MNIST"
#bmode="outliers_iqr"
#run_experiment_config(db, bmode=bmode,maindir="/home/gauravkj/Thesis/Results",bmult=4,verbose=1)
#run_all_config(db, maindir="/home/gaurav.kjain/Thesis/Results",bmult=4,verbose=1)

import sys
sys.path.append("nn_optimization")
from ModelPlotter import *
db="MNIST"
bmode="normal"
save_csv_config(db, bmode=bmode,maindir="./Results_pkk",bmult=4,verbose=1)
