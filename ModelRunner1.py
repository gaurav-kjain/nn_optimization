#ModelRunner1.py
print("version="+'22_2016_10.54_pm')
import sys
from dnn_globals import DirGlobals
#from IPython.display import clear_output
import os
#import stl10
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
gloss='categorical_crossentropy'

def run_experiment_config(db, bmode,maindir,bmult=6,verbose=0):    
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
    
    
    print(dbName[db])
    data=loadData(dbName[db])
    h,w,d,c,x_t,y_t,x_v,y_v,y_f=data#loadData(dbName[db])
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
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=mext__)
                    else:
                        finstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=mext__)
                    dp=getdirectorypath(MAIN_DIR,dbDict[db],nwDict[nw])
                    finfilename=getfinresname(dp,finstr)
                    mfinfilename=getfinresname(dp,mfinstr)
                    if os.path.isfile(finfilename) is False:
                        print(finfilename)
                        print(h,w,d,c,b)
                        sys.stdout.flush()
                        if nw is 'MERGE_WEAK_2':
                            hist,score=runAnalysisOptimizersMerge(nwFunc[nw],x_t,y_t,x_v,y_v,y_f,b_size=b,epoc=epoc__,v=verbose,loss=gloss,
                                  optimizer=optimizerDict[opti],img_h=h, img_w=w, d=d,c=c,init=inti, bmode=bmode)
                            save_results(hist, finfilename)
                        else:
                            hist,model=runAnalysisOptimizers(nwFunc[nw],x_t,y_t,x_v,y_v,y_f,b_size=b,epoc=epoc__,v=verbose,loss=gloss,
                                  optimizer=optimizerDict[opti],img_h=h, img_w=w, d=d,c=c,init=inti,bmode=bmode)
                            #if bmode is 'careful_init' or 'outliers' or 'care_outliers' or 'outliers_randthr' or 'outliers_iqr' or 'care_outliers_iqr' or 'care_outliers_iqr' :
                            save_results_via_pickle(hist, finfilename)
                            save_results_via_pickle(model, mfinfilename)
                            #else:
                             #   save_results(hist, finfilename)
                    clear_output()
                    #os.system("clear")


def run_all_config(db, maindir, bmult=6,verbose=0):
    
    for config in configName:
        print(configName[config])
        run_experiment_config(db=db, bmode=configName[config],maindir=maindir,bmult=bmult,verbose=verbose)



#ModelRunner1.py
