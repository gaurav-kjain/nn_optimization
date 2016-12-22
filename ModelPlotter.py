#ModelPlotter.py
import sys
from dnn_globals import DirGlobals
#from IPython.display import clear_output
import os
from ModelConfig import *

try:
    import cPickle as pickle
except:
    import pickle

sys.setrecursionlimit(600000)
print("SETTING RECURSION LIMIT")
epoc__ = 1000
ext__='.pkk'
mext__='.mdd'
cext__='.csv'
gloss='categorical_crossentropy'

def save_csv_config(db, bmode,maindir,bmult=6,c=10,verbose=0):    
    #set the directories now
    drg=DirGlobals()
    drg.setmaindir(path=maindir)
    print(drg.MAIN_DIR)
    MAIN_DIR=drg.MAIN_DIR

    if bmode is 'normal':
        print('setting normal')
        drg.setnormal()
    elif bmode is 'careful':
        print('setting care')
        drg.setcare()
    elif bmode is 'up':
        print('setting up')
        drg.setup()
    elif bmode is 'carefulup':
        print('setting careful up')
        drg.setcareup()
    elif bmode is 'careful_init':
        print('setting careful init')
        drg.setcareup_init()
    
    elif bmode is 'outliers':
        print('setting outliers')
        drg.setoutliers()
    
    elif bmode is 'care_outliers':
        print('setting care outliers')
        drg.setcareoutliers()
    
    elif bmode is 'outliers_randthr':
        print('setting rand outliers')
        drg.setrandoutliers()


    elif bmode is 'outliers_iqr':
        print('setting iqr outliers')
        drg.setiqroutliers()
    
    elif bmode is 'outliers_iqr_npass':
        print('setting iqr outliers npass')
        drg.setoutliersiqrnpass()
    
    elif bmode is 'care_outliers_iqr':
        print('setting care iqr outliers')
        drg.setcareiqroutliers()
    
    else:
        print('CHOOSE ON OF SUPPORTED MODE')
        return
    
    dbDict=drg.dbDict
    sys.stdout.flush()
    
    for opti in optimizerDict:
        for inti in initDict:
            print(inti)
            sys.stdout.flush()
            for nw in nwDict:
                for bs in range(1,bmult,1):
                    b=bs*c
                    if opti is 'SGD':
                        finstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=cext__)
                    else:
                        finstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=cext__)
                    dp=getdirectorypath(MAIN_DIR,dbDict[db],nwDict[nw])
                    finfilename=getfinresname(dp,finstr)
                    mfinfilename=getfinresname(dp,mfinstr)
                    #if os.path.isfile(finfilename) is True and os.path.isfile(mfinfilename) is False:
                    if os.path.isfile(finfilename) is True:
                        print(finfilename)
                        print(mfinfilename)
                        makecsv_global_write(finfilename,mfinfilename,bmode)
                        sys.stdout.flush()
                    #clear_output()
                    os.system("clear")


def save_all_config(maindir, bmult=6,verbose=0):
    drg=DirGlobals()
    drg.setmaindir(path=maindir)
    for db in drg.dbDict:
	print(db)
        c=10
        if db is 'CIFAR100':
		   c=100
        elif db is 'CIFAR20':
		    c=20
	for config in configName:
	    print(configName[config])
            save_csv_config(db=db, bmode=configName[config],maindir=maindir,bmult=bmult,c=c,verbose=verbose)



#ModelRunner1.py
