#dnn_globals.py
import os

class DirGlobals(object):
    

    def setmaindir(self,path):
        self.MAIN_DIR=os.path.abspath(path)
        self.MNIST_DIR=[]
        self.CIFAR10_DIR=[]
        self.CIFAR20_DIR=[]
        self.CIFAR100_DIR=[]
        self.CALTECH_101_DIR=[]
        self.STL_10_DIR=[]
        self.FLOWERS_102_DIR=[]
        
        self.dbDict = {
            'MNIST'     : self.MNIST_DIR,
            'CIFAR10'   : self.CIFAR10_DIR,
            'CIFAR100'  : self.CIFAR100_DIR,
            'CIFAR20'   : self.CIFAR20_DIR,
            'FLOWERS102': self.FLOWERS_102_DIR,
            'CALTECH101': self.CALTECH_101_DIR,
            'STL10'     : self.STL_10_DIR   
        }

    def updatepaths(self):
        self.MNIST_DIR = os.path.join(self.MAIN_DIR, self.MNIST_DIR)
        self.CIFAR10_DIR = os.path.join(self.MAIN_DIR, self.CIFAR10_DIR)
        self.CIFAR100_DIR = os.path.join(self.MAIN_DIR, self.CIFAR100_DIR)
        self.CIFAR20_DIR = os.path.join(self.MAIN_DIR, self.CIFAR20_DIR)
        self.CALTECH_101_DIR = os.path.join(self.MAIN_DIR, self.CALTECH_101_DIR)
        self.STL_10_DIR_DIR = os.path.join(self.MAIN_DIR, self.STL_10_DIR)
        self.FLOWERS_102__DIR = os.path.join(self.MAIN_DIR, self.FLOWERS_102_DIR)
        self.dbDict = {
            'MNIST'     : self.MNIST_DIR,
            'CIFAR10'   : self.CIFAR10_DIR,
            'CIFAR100'  : self.CIFAR100_DIR,
            'CIFAR20'   : self.CIFAR20_DIR,
            'FLOWERS102': self.FLOWERS_102_DIR,
            'CALTECH101': self.CALTECH_101_DIR,
            'STL10'     : self.STL_10_DIR   
        }

    def setnormal(self):
        self.MNIST_DIR = 'mnist'
        self.CIFAR10_DIR = 'cifar10'
        self.CIFAR100_DIR = 'cifar100'
        self.CIFAR20_DIR = 'cifar20'
        self.FLOWERS_102_DIR = 'flowers102'
        self.STL_10_DIR = 'stl10'
        self.CALTECH_101_DIR = 'caltech101'
        self.updatepaths()

    def setnormalmse(self):
        self.MNIST_DIR = 'mnist_mse'
        self.CIFAR10_DIR = 'cifar10_mse'
        self.CIFAR100_DIR = 'cifar100_mse'
        self.CIFAR20_DIR = 'cifar20_mse'
        self.FLOWERS_102_DIR = 'flowers102_mse'
        self.STL_10_DIR = 'stl10_mse'
        self.CALTECH_101_DIR = 'caltech101_mse'
 
        self.updatepaths()

    def setcare(self):
        self.MNIST_DIR = 'mnist_carebatch'
        self.CIFAR10_DIR = 'cifar10_carebatch'
        self.CIFAR100_DIR = 'cifar100_carebatch'
        self.CIFAR20_DIR = 'cifar20_carebatch'
        self.FLOWERS_102_DIR = 'flowers102_carebatch'
        self.STL_10_DIR = 'stl10_carebatch'
        self.CALTECH_101_DIR = 'caltech101_carebatch'
 
        self.updatepaths()

    def setup(self):
        self.MNIST_DIR = 'mnist_up'
        self.CIFAR10_DIR = 'cifar10_up'
        self.CIFAR100_DIR = 'cifar100_up'
        self.CIFAR20_DIR = 'cifar20_up'
        self.FLOWERS_102_DIR = 'flowers102_up'
        self.STL_10_DIR = 'stl10_up'
        self.CALTECH_101_DIR = 'caltech101_up'

        self.updatepaths()

    def setcareup(self):
        self.MNIST_DIR = 'mnist_careup'
        self.CIFAR10_DIR = 'cifar10_careup'
        self.CIFAR100_DIR = 'cifar100_careup'
        self.CIFAR20_DIR = 'cifar20_careup'
        self.FLOWERS_102_DIR = 'flowers102_careup'
        self.STL_10_DIR = 'stl10_careup'
        self.CALTECH_101_DIR = 'caltech101_careup'

        self.updatepaths()

    def setcareup_init(self):
        self.MNIST_DIR = 'mnist_careup_init'
        self.CIFAR10_DIR = 'cifar10_careup_init'
        self.CIFAR100_DIR = 'cifar100_careup_init'
        self.CIFAR20_DIR = 'cifar20_careup_init'
        self.FLOWERS_102_DIR = 'flowers102_careup_init'
        self.STL_10_DIR = 'stl10_careup_init'
        self.CALTECH_101_DIR = 'caltech101_careup_init'

        self.updatepaths()
    
    def setoutliers(self):
        self.MNIST_DIR = 'mnist_outliers'
        self.CIFAR10_DIR = 'cifar10_outliers'
        self.CIFAR100_DIR = 'cifar100_outliers'
        self.CIFAR20_DIR = 'cifar20_outliers'
        self.FLOWERS_102_DIR = 'flowers102_outliers'
        self.STL_10_DIR = 'stl10_outliers'
        self.CALTECH_101_DIR = 'caltech101_outliers'

        self.updatepaths()

    def setcareoutliers(self):
        self.MNIST_DIR = 'mnist_care_outliers'
        self.CIFAR10_DIR = 'cifar10_care_outliers'
        self.CIFAR100_DIR = 'cifar100_care_outliers'
        self.CIFAR20_DIR = 'cifar20_care_outliers'
        self.FLOWERS_102_DIR = 'flowers102_care_outliers'
        self.STL_10_DIR = 'stl10_care_outliers'
        self.CALTECH_101_DIR = 'caltech101_care_outliers'

        self.updatepaths()
    
    def setrandoutliers(self):
        self.MNIST_DIR = 'mnist_rand_outliers'
        self.CIFAR10_DIR = 'cifar10_rand_outliers'
        self.CIFAR100_DIR = 'cifar100_rand_outliers'
        self.CIFAR20_DIR = 'cifar20_rand_outliers'
        self.FLOWERS_102_DIR = 'flowers102_rand_outliers'
        self.STL_10_DIR = 'stl10_rand_outliers'
        self.CALTECH_101_DIR = 'caltech101_rand_outliers'

        self.updatepaths()
    
    def setiqroutliers(self):
        self.MNIST_DIR = 'mnist_iqr_outliers'
        self.CIFAR10_DIR = 'cifar10_iqr_outliers'
        self.CIFAR100_DIR = 'cifar100_iqr_outliers'
        self.CIFAR20_DIR = 'cifar20_iqr_outliers'
        self.FLOWERS_102_DIR = 'flowers102_iqr_outliers'
        self.STL_10_DIR = 'stl10_iqr_outliers'
        self.CALTECH_101_DIR = 'caltech101_iqr_outliers'

        self.updatepaths()
    
    def setcareiqroutliers(self):
        self.MNIST_DIR = 'mnist_care_iqr_outliers'
        self.CIFAR10_DIR = 'cifar10_care_iqr_outliers'
        self.CIFAR100_DIR = 'cifar100_care_iqr_outliers'
        self.CIFAR20_DIR = 'cifar20_care_iqr_outliers'
        self.FLOWERS_102_DIR = 'flowers102_care_iqr_outliers'
        self.STL_10_DIR = 'stl10_care_iqr_outliers'
        self.CALTECH_101_DIR = 'caltech101_care_iqr_outliers'

        self.updatepaths()
    
    def setoutliersiqrnpass(self):
        self.MNIST_DIR = 'mnist_iqr_outliers_npass'
        self.CIFAR10_DIR = 'cifar10_iqr_outliers_npass'
        self.CIFAR100_DIR = 'cifar100_iqr_outliers_npass'
        self.CIFAR20_DIR = 'cifar20_iqr_outliers_npass'
        self.FLOWERS_102_DIR = 'flowers102_iqr_outliers_npass'
        self.STL_10_DIR = 'stl10_iqr_outliers_npass'
        self.CALTECH_101_DIR = 'caltech101_iqr_outliers_npass'

        self.updatepaths()
    
    def printGlobals(self):
        print(self.MAIN_DIR)
        print(self.MNIST_DIR)
        print(self.CIFAR10_DIR)
        print(self.CIFAR100_DIR)
        print(self.CIFAR20_DIR)
        print(self.CALTECH_101_DIR)
        print(self.FLOWERS_102_DIR)
        print(self.STL_10__DIR)

#dnn_globals.py
