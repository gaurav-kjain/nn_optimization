import sys
sys.path.append("dnn_optimization")
from ModelRunner import *
db="MNIST"
bmode="outliers_iqr_npass"
run_experiment_config(db, bmode=bmode,maindir="/home/gaurav.kjain/Thesis/Results",bmult=4,verbose=1)


from ModelRunner import *
db="CIFAR10"
bmode="outliers_iqr_npass"
run_experiment_config(db, bmode=bmode,maindir="/home/gauravkj/Thesis/Results",bmult=4,verbose=1)


pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git