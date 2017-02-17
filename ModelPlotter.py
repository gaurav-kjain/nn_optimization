#ModelPlotter.py
import sys
from dnn_globals import DirGlobals
#from IPython.display import clear_output
import os
from ModelConfig import *
import pandas as pd
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
    #print(drg.MAIN_DIR)
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
    
    l1=[]
    l2=[]
    l4=[]
    l6=[]
    l8=[]
    l10=[]

    for nw in nwDict:
        i1=[]
        i2=[]
        i4=[]
        i6=[]
        i8=[]
        i10=[]
        for inti in initDict:
            for opti in optimizerDict:
                for bs in range(1,bmult,1):
                    b=bs*c
                    if opti is 'SGD':
                        finstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=cext__)
                        mdescstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=cext__)
                    else:
                        finstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=cext__)
                        mdesctr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=cext__)
                    dp=getdirectorypath(MAIN_DIR,dbDict[db],nwDict[nw])
                    finfilename=getfinresname(dp,finstr)
                    mfinfilename=getfinresname(MAIN_DIR,dbDict[db]+'_'+nwDict[nw]+'_'+mfinstr)
                    mdescfilename=getfinresname(MAIN_DIR,dbDict[db]+'_'+nwDict[nw]+'_describe'+mfinstr)
                    if os.path.isfile(finfilename) is True:
                        towrdata=makecsv_global_write(finfilename,mfinfilename,mdescfilename,bmode)
                        if (bs==1):
                            l1.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i1.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if bs is 2:
                            l2.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i2.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if bs is 4:
                            l4.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i4.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if bs is 6:
                            l6.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i6.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if bs is 8:
                            l8.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i8.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if bs is 10:
                            l10.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i10.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

        id1=pd.DataFrame(i1, columns=list(['acc','loss','val_acc','val_loss']))
        id2=pd.DataFrame(i2, columns=list(['acc','loss','val_acc','val_loss']))
        id4=pd.DataFrame(i4, columns=list(['acc','loss','val_acc','val_loss']))
        id6=pd.DataFrame(i6, columns=list(['acc','loss','val_acc','val_loss']))
        id8=pd.DataFrame(i8, columns=list(['acc','loss','val_acc','val_loss']))
        id10=pd.DataFrame(i10, columns=list(['acc','loss','val_acc','val_loss']))
    
    
        i1name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+nw+'_1x.csv')
        i2name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+nw+'_2x.csv')
        i4name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+nw+'_4x.csv')
        i6name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+nw+'_6x.csv')
        i8name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+nw+'_8x.csv')
        i10name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+nw+'_10x.csv')
        id1=id1.reset_index(drop=True)
        id1.to_csv(i1name)
    
        id2=id2.reset_index(drop=True)
        id2.to_csv(i2name)
    
        id4=id4.reset_index(drop=True)
        id4.to_csv(i4name)
    
        id6=id6.reset_index(drop=True)
        id6.to_csv(i6name)
    
        id8=id8.reset_index(drop=True)
        id8.to_csv(i8name)
    
        id10=id10.reset_index(drop=True)
        id10.to_csv(i10name)

    pd1=pd.DataFrame(l1, columns=list(['acc','loss','val_acc','val_loss']))
    pd2=pd.DataFrame(l2, columns=list(['acc','loss','val_acc','val_loss']))
    pd4=pd.DataFrame(l4, columns=list(['acc','loss','val_acc','val_loss']))
    pd6=pd.DataFrame(l6, columns=list(['acc','loss','val_acc','val_loss']))
    pd8=pd.DataFrame(l8, columns=list(['acc','loss','val_acc','val_loss']))
    pd10=pd.DataFrame(l10, columns=list(['acc','loss','val_acc','val_loss']))
    
    
    b1name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+'_1x.csv')
    b2name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+'_2x.csv')
    b4name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+'_4x.csv')
    b6name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+'_6x.csv')
    b8name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+'_8x.csv')
    b10name=getfinresname(MAIN_DIR,'batch_'+bmode+'_'+dbName[db]+'_10x.csv')
    print(b1name)
    pd1=pd1.reset_index(drop=True)
    pd1.to_csv(b1name)
    
    pd2=pd2.reset_index(drop=True)
    pd2.to_csv(b2name)
    
    pd4=pd4.reset_index(drop=True)
    pd4.to_csv(b4name)
    
    pd6=pd6.reset_index(drop=True)
    pd6.to_csv(b6name)
    
    pd8=pd8.reset_index(drop=True)
    pd8.to_csv(b8name)
    
    pd10=pd10.reset_index(drop=True)
    pd10.to_csv(b10name)


def save_csv_config_opti(db, bmode,maindir,bmult=6,c=10,verbose=0):    
    #set the directories now
    drg=DirGlobals()
    drg.setmaindir(path=maindir)
    #print(drg.MAIN_DIR)
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
    
    l1=[]
    l2=[]
    l4=[]
    l6=[]
    l8=[]
    l10=[]

    for nw in nwDict:

        i1=[]
        i2=[]
        i4=[]
        i6=[]
        i8=[]
        i10=[]
        for inti in initDict:
            for bs in range(1,bmult,1):
                b=bs*c
                for opti in optimizerDict:
                    if opti is 'SGD':
                        finstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=cext__)
                        mdescstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=cext__)
                    else:
                        finstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=cext__)
                        mdesctr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=cext__)
                    dp=getdirectorypath(MAIN_DIR,dbDict[db],nwDict[nw])
                    finfilename=getfinresname(dp,finstr)
                    mfinfilename=getfinresname(MAIN_DIR,dbDict[db]+'_'+nwDict[nw]+'_'+mfinstr)
                    mdescfilename=getfinresname(MAIN_DIR,dbDict[db]+'_'+nwDict[nw]+'_describe'+mfinstr)
                    if os.path.isfile(finfilename) is True:
                        towrdata=makecsv_global_write(finfilename,mfinfilename,mdescfilename,bmode)
                        if opti is 'ADL':
                            l1.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i1.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if opti is 'ADG':
                            l2.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i2.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if opti is 'SGD':
                            l4.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i4.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if opti is 'ADM':
                            l6.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i6.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if opti is 'ADMX':
                            l8.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i8.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if opti is 'NDM':
                            l10.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i10.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

        id1=pd.DataFrame(i1, columns=list(['acc','loss','val_acc','val_loss']))
        id2=pd.DataFrame(i2, columns=list(['acc','loss','val_acc','val_loss']))
        id4=pd.DataFrame(i4, columns=list(['acc','loss','val_acc','val_loss']))
        id6=pd.DataFrame(i6, columns=list(['acc','loss','val_acc','val_loss']))
        id8=pd.DataFrame(i8, columns=list(['acc','loss','val_acc','val_loss']))
        id10=pd.DataFrame(i10, columns=list(['acc','loss','val_acc','val_loss']))
    
    
        i1name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+nw+'_adadelta.csv')
        i2name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+nw+'_adagrad.csv')
        i4name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+nw+'_sgd.csv')
        i6name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+nw+'_adam.csv')
        i8name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+nw+'_adamax.csv')
        i10name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+nw+'_nadam.csv')
        id1=id1.reset_index(drop=True)
        id1.to_csv(i1name)
    
        id2=id2.reset_index(drop=True)
        id2.to_csv(i2name)
    
        id4=id4.reset_index(drop=True)
        id4.to_csv(i4name)
    
        id6=id6.reset_index(drop=True)
        id6.to_csv(i6name)
    
        id8=id8.reset_index(drop=True)
        id8.to_csv(i8name)
    
        id10=id10.reset_index(drop=True)
        id10.to_csv(i10name)

    pd1=pd.DataFrame(l1, columns=list(['acc','loss','val_acc','val_loss']))
    pd2=pd.DataFrame(l2, columns=list(['acc','loss','val_acc','val_loss']))
    pd4=pd.DataFrame(l4, columns=list(['acc','loss','val_acc','val_loss']))
    pd6=pd.DataFrame(l6, columns=list(['acc','loss','val_acc','val_loss']))
    pd8=pd.DataFrame(l8, columns=list(['acc','loss','val_acc','val_loss']))
    pd10=pd.DataFrame(l10, columns=list(['acc','loss','val_acc','val_loss']))
    
    
    b1name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+'_adadelta.csv')
    b2name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+'_adagrad.csv')
    b4name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+'_sgd.csv')
    b6name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+'_adam.csv')
    b8name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+'_adamax.csv')
    b10name=getfinresname(MAIN_DIR,'opti_'+bmode+'_'+dbName[db]+'_nadam.csv')
    print(b1name)
    pd1=pd1.reset_index(drop=True)
    pd1.to_csv(b1name)
    
    pd2=pd2.reset_index(drop=True)
    pd2.to_csv(b2name)
    
    pd4=pd4.reset_index(drop=True)
    pd4.to_csv(b4name)
    
    pd6=pd6.reset_index(drop=True)
    pd6.to_csv(b6name)
    
    pd8=pd8.reset_index(drop=True)
    pd8.to_csv(b8name)
    
    pd10=pd10.reset_index(drop=True)
    pd10.to_csv(b10name)


def save_csv_config_init(db, bmode,maindir,bmult=6,c=10,verbose=0):    
    #set the directories now
    drg=DirGlobals()
    drg.setmaindir(path=maindir)
    #print(drg.MAIN_DIR)
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
    
    l1=[]
    l2=[]
    l4=[]
    l6=[]
    l8=[]
    l10=[]

    for nw in nwDict:
        i1=[]
        i2=[]
        i4=[]
        i6=[]
        i8=[]
        i10=[]
        for opti in optimizerDict:
            for bs in range(1,bmult,1):
                b=bs*c
                for inti in initDict:
                    if opti is 'SGD':
                        finstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=cext__)
                        mdescstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=cext__)
                    else:
                        finstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=cext__)
                        mdesctr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=cext__)
                    dp=getdirectorypath(MAIN_DIR,dbDict[db],nwDict[nw])
                    finfilename=getfinresname(dp,finstr)
                    mfinfilename=getfinresname(MAIN_DIR,dbDict[db]+'_'+nwDict[nw]+'_'+mfinstr)
                    mdescfilename=getfinresname(MAIN_DIR,dbDict[db]+'_'+nwDict[nw]+'_describe'+mfinstr)
                    if os.path.isfile(finfilename) is True:
                        towrdata=makecsv_global_write(finfilename,mfinfilename,mdescfilename,bmode)
                        if inti is 'unif':
                            l1.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i1.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if inti is 'norm':
                            l2.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i2.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if inti is 'zero':
                            l4.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i4.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if inti is 'gl_n':
                            l6.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i6.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

                        if inti is 'gl_u':
                            l8.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i8.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                        
                        if inti is 'he_u':
                            l10.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])
                            i10.append([towrdata['acc'][7],towrdata['loss'][3], towrdata['val_acc'][7],towrdata['val_loss'][3]])

        id1=pd.DataFrame(i1, columns=list(['acc','loss','val_acc','val_loss']))
        id2=pd.DataFrame(i2, columns=list(['acc','loss','val_acc','val_loss']))
        id4=pd.DataFrame(i4, columns=list(['acc','loss','val_acc','val_loss']))
        id6=pd.DataFrame(i6, columns=list(['acc','loss','val_acc','val_loss']))
        id8=pd.DataFrame(i8, columns=list(['acc','loss','val_acc','val_loss']))
        id10=pd.DataFrame(i10, columns=list(['acc','loss','val_acc','val_loss']))
    
    
        i1name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+nw+'_unif.csv')
        i2name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+nw+'_norm.csv')
        i4name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+nw+'_zero.csv')
        i6name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+nw+'_gl_n.csv')
        i8name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+nw+'_gl_u.csv')
        i10name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+nw+'_he_u.csv')
        id1=id1.reset_index(drop=True)
        id1.to_csv(i1name)
    
        id2=id2.reset_index(drop=True)
        id2.to_csv(i2name)
    
        id4=id4.reset_index(drop=True)
        id4.to_csv(i4name)
    
        id6=id6.reset_index(drop=True)
        id6.to_csv(i6name)
    
        id8=id8.reset_index(drop=True)
        id8.to_csv(i8name)
    
        id10=id10.reset_index(drop=True)
        id10.to_csv(i10name)

    pd1=pd.DataFrame(l1, columns=list(['acc','loss','val_acc','val_loss']))
    pd2=pd.DataFrame(l2, columns=list(['acc','loss','val_acc','val_loss']))
    pd4=pd.DataFrame(l4, columns=list(['acc','loss','val_acc','val_loss']))
    pd6=pd.DataFrame(l6, columns=list(['acc','loss','val_acc','val_loss']))
    pd8=pd.DataFrame(l8, columns=list(['acc','loss','val_acc','val_loss']))
    pd10=pd.DataFrame(l10, columns=list(['acc','loss','val_acc','val_loss']))
    
    
    b1name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+'_unif.csv')
    b2name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+'_norm.csv')
    b4name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+'_zero.csv')
    b6name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+'_gl_n.csv')
    b8name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+'_gl_u.csv')
    b10name=getfinresname(MAIN_DIR,'init_'+bmode+'_'+dbName[db]+'_he_u.csv')
    print(b1name)
    pd1=pd1.reset_index(drop=True)
    pd1.to_csv(b1name)
    
    pd2=pd2.reset_index(drop=True)
    pd2.to_csv(b2name)
    
    pd4=pd4.reset_index(drop=True)
    pd4.to_csv(b4name)
    
    pd6=pd6.reset_index(drop=True)
    pd6.to_csv(b6name)
    
    pd8=pd8.reset_index(drop=True)
    pd8.to_csv(b8name)
    
    pd10=pd10.reset_index(drop=True)
    pd10.to_csv(b10name)

def save_for_batch(db, bmode,maindir,bmult=6,c=10,verbose=0):    
    #set the directories now
    drg=DirGlobals()
    drg.setmaindir(path=maindir)
    #print(drg.MAIN_DIR)
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
            #print(inti)
            sys.stdout.flush()
            for nw in nwDict:
                majord=pd.DataFrame()
                for bs in range(1,bmult,1):
                    b=bs*c
                    if opti is 'SGD':
                        finstr=makefinresults_filename_init(b=b,e=0,opti='SGD',init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b='batchall',e=0,opti='SGD',init=inti, ext=cext__)
                    else:
                        finstr=makefinresults_filename_init(b=b,e=0,opti=optimizerDict[opti],init=inti, ext=ext__)
                        mfinstr=makefinresults_filename_init(b='batchall',e=0,opti=optimizerDict[opti],init=inti, ext=cext__)
                    dp=getdirectorypath(MAIN_DIR,dbDict[db],nwDict[nw])
                    finfilename=getfinresname(dp,finstr)
                    mfinfilename=getfinresname(MAIN_DIR,dbDict[db]+'_'+nwDict[nw]+'_'+mfinstr)
                    mdescfilename=getfinresname(MAIN_DIR,dbDict[db]+'_'+nwDict[nw]+'_describe'+mfinstr)
                    if os.path.isfile(finfilename) is True:
                        print(finfilename)
                        print(mfinfilename)
                        df=makedf_global_from_npz(finfilename)
                        n=df.shape[0]
                        df['batch']=np.full(n,bs)
                        majord=pd.concat([majord,df])
                if majord.empty is False:
                    majord=majord.reset_index()
                    majord.to_csv(mfinfilename,sep=' ')
                #clear_output()
                #os.system("clear")


def save_all_config(maindir, bmult=6,verbose=0):
    drg=DirGlobals()
    drg.setmaindir(path=maindir)
    for db in drg.dbDict:
	#print(db)
        c=10
        if db is 'CIFAR100':
            c=100
        elif db is 'CIFAR20':
            c=20
        elif db is 'MNIST':
            c=10
        #else:
         #   continue
	for config in configName:
	    #print(configName[config])
            save_csv_config(db=db, bmode=configName[config],maindir=maindir,bmult=bmult,c=c,verbose=verbose)
            save_csv_config_opti(db=db, bmode=configName[config],maindir=maindir,bmult=bmult,c=c,verbose=verbose)
            save_csv_config_init(db=db, bmode=configName[config],maindir=maindir,bmult=bmult,c=c,verbose=verbose)


