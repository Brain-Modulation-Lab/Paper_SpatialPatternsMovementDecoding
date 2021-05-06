#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:30:37 2020

@author: victoria
"""

#%%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import mne
from mne.decoding import SPoC
mne.set_log_level(verbose='warning') #to avoid info at terminal
import pickle 
import sys
# from Utilities folder
sys.path.insert(1, './Utilities/icn_m1')
import os
sys.path.insert(1, './Utilities/')
from FilterBank import FilterBank
from ML_models import get_model

from collections import OrderedDict
from sklearn.model_selection import KFold

import gc
from sklearn.preprocessing import StandardScaler

#%%
settings = {}
settings['data_path'] = "/home/PARTNERS/vp820/media/Nexus3/Paper_code_repository/VPeterson_SpatialFilters_2021/Data/Epoched_data/"
settings['out_path_process'] = "/home/PARTNERS/vp820/media/Nexus3/Paper_code_repository/VPeterson_SpatialFilters_2021/Scripts/Results/Unimodal/tlag/alpha05/"
settings['frequencyranges'] = [[4, 8], [8, 12], [13, 20], [20, 35], [13, 35], [60, 80], [90, 200], [60, 200]]
settings['seglengths']=[1, 2, 2, 3, 3, 3, 10, 10, 10]
settings['num_patients'] = ['000', '001', '004', '005', '006', '007', '008', '009', '010', '013', '014']
#this line  useful when workng in Windows OS.
settings['data_path'] = settings['data_path'].replace("\\", "/")
# subfolders
settings['subfolders']=[['ses-right'], 
                        ['ses-right', 'ses-left'], 
                        ['ses-right', 'ses-left'],
                        ['ses-right', 'ses-left'], 
                        ['ses-right', 'ses-left'],
                        ['ses-left'], 
                        ['ses-left'],
                        ['ses-left'], 
                        ['ses-right', 'ses-left'],
                        ['ses-left'],
                        ['ses-right']
                        ]
#%%
def DetecBadTrials(X,y, verbose=True):
    #based on eq.16 of the paper "Dimensionality reduction for the analysis of brain oscillations"
    #calcule single global variance value (GVV)
    nt, nc, ns= np.shape(X)
    TrialVar=np.zeros((nt,nc,1))

    for c in range(nc):
        for n in range(nt):
            TrialVar[n,c,:]=np.var(X[n,c,:])
    
    GVV=np.squeeze(np.mean(TrialVar, axis=1))
    Q5=np.percentile(GVV, 5)
    Q95=np.percentile(GVV, 95)

    Thr=Q95+3*(Q95-Q5)
    
    #elimiate trails with large variance
    index_good=np.where(GVV<Thr)[0]
    index_bad=np.where(GVV>Thr)[0]

    if verbose:
        if len(index_bad)>0 :
            print('Detected bad trials')
    
    X_clean=X[index_good,:,:]
    Y_clean=y[index_good]
    
    return X_clean,Y_clean

def append_time_dim(arr, y_, time_stamps):
    """
    apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
    """
    time_arr = np.zeros([arr.shape[0]-time_stamps, int(time_stamps*arr.shape[1])])
    for time_idx, time_ in enumerate(np.arange(time_stamps, arr.shape[0])):
        for time_point in range(time_stamps):
            time_arr[time_idx, time_point*arr.shape[1]:(time_point+1)*arr.shape[1]] = arr[time_-time_point,:]
    return time_arr, y_[time_stamps:]         
#%%
laterality=["CON", "IPS"]
signal=["STN", "ECOG"]

cv = KFold(n_splits=5, shuffle=False)
spoc= SPoC(n_components=1, log=True, reg='oas', transform_into ='average_power', rank='full')
USED_MODEL = 3# 2 == tweedie, 3 == GLM
#%%
len(settings['num_patients'])
for s in range(len(settings['num_patients'])):
    gc.collect()

    subfolder=settings["subfolders"][s]

    for ss in range(len(subfolder)):
        X_ECOG = [] # to append data
        X_STN =[] # to append data
        Y_con = []
        Y_ips = []
        list_of_files_ecog = os.listdir(settings['data_path']+'ECOG') # list of files in the current directory
        list_of_files_stn = os.listdir(settings['data_path']+'STN') # list of files in the current directory
    
        file_name_ = 'ECOG_epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
    
        file_ecog = [each_file for each_file in list_of_files_ecog if each_file.startswith(file_name_)]
        file_name_='STN_epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
    
        file_stn= [each_file for each_file in list_of_files_stn if each_file.startswith(file_name_)]
        idx_file = [f for f in file_stn if list(set() & set(file_ecog))]
        matching_stn = [f for f in file_stn if any(f[4:] in xs for xs in file_ecog)]
        matching_ecog = [f for f in file_ecog if any(f[4:] in xs for xs in file_stn)]

        if len(matching_ecog) != len(matching_stn):
            raise('Error loading data')
        
        for e in range(len(matching_ecog)):
            with open(settings['data_path'] +'ECOG/' + matching_ecog[e], 'rb') as handle:
                sub_ = pickle.load(handle)    
                data = sub_['epochs']
                X_ECOG.append(data)
                label_ips = sub_['label_ips']
                label_con = sub_['label_con']
                Y_con.append(label_con)
                Y_ips.append(label_ips)
            with open(settings['data_path'] +'STN/' + matching_stn[e], 'rb') as handle:
                sub_ = pickle.load(handle)
                data = sub_['epochs']
                X_STN.append(data)           
            gc.collect()
        if np.size(X_ECOG) == 0:
            continue
           
        X_ECOG = np.concatenate(X_ECOG, axis=0)
        X_STN = np.concatenate(X_STN, axis=0)
        Y_con = np.concatenate(Y_con, axis=0)
        Y_ips = np.concatenate(Y_ips, axis=0)  
        # declare variable
        Ypre_tr = OrderedDict()
        score_tr = OrderedDict()
        Ypre_te = OrderedDict()
        score_te = OrderedDict()
        Patterns = OrderedDict()
        Filters = OrderedDict()
        Coef = OrderedDict()
        hyperparams = OrderedDict()
        Label_tr = OrderedDict()
        Label_te = OrderedDict()

        gc.collect()
        
        for m, eeg in enumerate(signal): 
            if eeg == "ECOG":
                X = X_ECOG
            else:
                X = X_STN
            print('RUNNIN SUBJECT_'+ settings['num_patients'][s]+ '_SESS_'+ str(subfolder[ss]) + '_SIGNAL_' + eeg)
            for t in range(1,11):
                print("time_lag %s" %t)
                for ll, mov in enumerate(laterality):
                    print("training %s" %mov)
                    score_tr[mov] = []
                    score_te[mov] = []
                    Ypre_tr[mov] = []
                    Ypre_te[mov] = []
                    Label_tr[mov] = []
                    Label_te[mov] = []
                    Patterns[mov] = []
                    Filters[mov] = []
                    Coef[mov] = []
                    hyperparams[mov] = []
                    if ll == 0:
                        label = Y_con
                    else:
                        label = Y_ips

                    features = FilterBank(estimator=spoc)
                    gc.collect()

                    for train_index, test_index in cv.split(label):
                        Ztr, Zte = label[train_index], label[test_index]
                        gtr = features.fit_transform(X[train_index], Ztr)
                        gte = features.transform(X[test_index])
                        
                         
                        dat_tr,label_tr = append_time_dim(gtr, Ztr,time_stamps=t)
                        dat_te,label_te = append_time_dim(gte, Zte,time_stamps=t)
                    
                        Label_te[mov].append(label_te)
                        Label_tr[mov].append(label_tr)
                                                   
                        clf, optimizer = get_model(USED_MODEL, x=dat_tr, y=label_tr)
        
                        scaler = StandardScaler()
                        scaler.fit(dat_tr)
                        dat_tr = scaler.transform(dat_tr)
                        dat_te = scaler.transform(dat_te)
        
                        
                        clf.fit(dat_tr, label_tr)
                        Ypre_te[mov].append(clf.predict(dat_te))
                        Ypre_tr[mov].append(clf.predict(dat_tr))
                        
                        r2_te = r2_score(label_te, clf.predict(dat_te))
                        if r2_te < 0: r2_te = 0
                        score_te[mov].append(r2_te)
                        r2_tr = r2_score(label_tr,clf.predict(dat_tr))
                        if r2_tr < 0: r2_tr = 0
        
                        score_tr[mov].append(r2_tr)
                              
                        # Filters[mov].append(filters)
                        # Patterns[mov].append(patterns)
                        if USED_MODEL > 1:
                            Coef[mov].append(clf.beta_)
                        else:
                            Coef[mov].append(clf.coef_)
                        hyperparams[mov].append(optimizer['params'])
                print(np.mean(score_te["CON"]))
                fig, ax = plt.subplots(nrows=2)
                ax[0].plot(label_te)
                ax[0].plot(clf.predict(dat_te))
                ax[1].stem(Coef[mov][-1])

           
                # %% save 
                predict_ = {
                    "y_pred_test": Ypre_te,
                    "y_test": Label_te,
                    "y_pred_train": Ypre_tr,
                    "y_train": Label_tr,
                    "score_tr": score_tr,
                    "score_te": score_te,
                    "coef": Coef,
                    "classifiers": clf,
                    "model_hyperparams": hyperparams,
                    "model": USED_MODEL,
                    "cv": 5
                    
                }
                out_path_file = os.path.join(settings['out_path_process'] + settings['num_patients'][s]+'predictions_'+eeg+'_tlag'+str(t)+'_'+str(subfolder[ss])+'.npy')
                np.save(out_path_file, predict_)        
                
                gc.collect()
        del X
            
            
            
