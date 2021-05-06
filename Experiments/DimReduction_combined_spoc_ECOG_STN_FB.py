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
from pyglmnet import GLM
from scipy.linalg import norm
from sklearn.linear_model import LinearRegression
#%%
settings = {}
settings['data_path'] = "/home/PARTNERS/vp820/media/Nexus3/Paper_code_repository/VPeterson_SpatialFilters_2021/Data/Epoched_data/"
settings['out_path_process'] = "/home/PARTNERS/vp820/media/Nexus3/Paper_code_repository/VPeterson_SpatialFilters_2021/Scripts/Results/Multimodal/FeatureSelection/"
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
def append_time_dim(arr, y_, time_stamps):
    """
    apply added time dimension for the data array and label given time_stamps (with downsample_rate=100) in 100ms / need to check with 1375Hz
    """
    time_arr = np.zeros([arr.shape[0]-time_stamps, int(time_stamps*arr.shape[1])])
    for time_idx, time_ in enumerate(np.arange(time_stamps, arr.shape[0])):
        for time_point in range(time_stamps):
            time_arr[time_idx, time_point*arr.shape[1]:(time_point+1)*arr.shape[1]] = arr[time_-time_point,:]
    return time_arr, y_[time_stamps:]

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
  
#%%
laterality=["CON", "IPS"]
signal=["ECOG","STN"]

cv = KFold(n_splits=5, shuffle=False)
spoc= SPoC(n_components=1, log=True, reg='oas', transform_into ='average_power', rank='full')
USED_MODEL = 'GLM' # 2 == TWEEDIE, 3 == GLM 
#%% CV split
len(settings['num_patients'])
for s in range(len(settings['num_patients'])):
    gc.collect()

    subfolder=settings["subfolders"][s]

   
    for ss in range(len(subfolder)):
        X_ECOG=[] #to append data
        X_STN=[] #to append data
        Y_con=[]
        Y_ips=[]
        list_of_files_ecog = os.listdir(settings['data_path']+'ECOG') #list of files in the current directory
        list_of_files_stn = os.listdir(settings['data_path']+'STN') #list of files in the current directory
    
        file_name_='ECOG_epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
    
        file_ecog= [each_file for each_file in list_of_files_ecog if each_file.startswith(file_name_)]
        file_name_='STN_epochs_sub_' + settings['num_patients'][s] + '_sess_'+subfolder[ss][4:]
    
        file_stn= [each_file for each_file in list_of_files_stn if each_file.startswith(file_name_)]
        idx_file= [f for f in file_stn if list(set() & set(file_ecog))]
        matchers = ['abc','def']
        matching_stn = [f for f in file_stn if any(f[4:] in xs for xs in file_ecog)]
        matching_ecog = [f for f in file_ecog if any(f[4:] in xs for xs in file_stn)]
        
        if len(matching_ecog) != len(matching_stn):
            raise('Error loading data')
        
        for e in range(len(matching_ecog)):
            with open(settings['data_path'] +'ECOG/' + matching_ecog[e], 'rb') as handle:
                sub_ = pickle.load(handle)    
       
                data=sub_['epochs']
                X_ECOG.append(data)

                label_ips=sub_['label_ips']
                label_con=sub_['label_con']
                Y_con.append(label_con)
                Y_ips.append(label_ips)
            with open(settings['data_path'] +'STN/' + matching_stn[e], 'rb') as handle:
                sub_ = pickle.load(handle)    
       
                data=sub_['epochs']
                X_STN.append(data)         
            gc.collect()
        if np.size(X_ECOG) == 0:
            continue
           
        X_ECOG=np.concatenate(X_ECOG, axis=0)
        X_STN=np.concatenate(X_STN, axis=0)
        Y_con=np.concatenate(Y_con, axis=0)
        Y_ips=np.concatenate(Y_ips, axis=0)  
        
        #declare variable
        # declare variable
        Ypre_tr = OrderedDict()
        score_tr = OrderedDict()
        Ypre_te = OrderedDict()
        score_te = OrderedDict()
        Patterns_ecog = OrderedDict()
        Filters_ecog = OrderedDict()
        Patterns_stn = OrderedDict()
        Filters_stn = OrderedDict()
        Coef = OrderedDict()
        df = OrderedDict()
        Label_tr = OrderedDict()
        Label_te = OrderedDict()           
        reg_params = OrderedDict()  
    
        for l, mov in enumerate(laterality):
            print("training %s" %mov)
            score_tr[mov] = []
            score_te[mov] = []
            Ypre_tr[mov] = []
            Ypre_te[mov] = []
            Label_tr[mov] = []
            Label_te[mov] = []
            Patterns_ecog[mov] = []
            Filters_ecog[mov] = []
            Patterns_stn[mov] = []
            Filters_stn[mov] = []
            Coef[mov] = []
            df[mov] = []
            reg_params[mov] = []
            if l==0:
                label=Y_con
            else:
                label=Y_ips
            # result_lm=[]
           
            
            # label_test=[]
            # label_train=[]
            
            # onoff_test=[]
            # onoff_train=[]
            features_ecog=FilterBank(estimator=spoc)
            features_stn=FilterBank(estimator=spoc)

            X_ECOG=X_ECOG.astype('float64')
            X_STN=X_STN.astype('float64')
           
            gtr=[]
            gte=[]

            
            for train_index, test_index in cv.split(label):
                    
                Ztr, Zte=label[train_index], label[test_index]
                Xtr_ecog, Xte_ecog=X_ECOG[train_index], X_ECOG[test_index]
                Xtr_stn, Xte_stn=X_STN[train_index], X_STN[test_index]
                Gtr_ecog_cspoc, Gtr_stn_cspoc = features_ecog.fit_transform(Xtr_ecog, Ztr), features_stn.fit_transform(Xtr_stn, Ztr)
                Gte_ecog_cspoc, Gte_stn_cspoc = features_ecog.transform(Xte_ecog), features_stn.transform(Xte_stn)

                dat_tr=np.hstack((Gtr_ecog_cspoc,Gtr_stn_cspoc))
                dat_te=np.hstack((Gte_ecog_cspoc,Gte_stn_cspoc))
                
                Label_te[mov].append(Zte)
                Label_tr[mov].append(Ztr)
                                       
                # dimenstionallity reduction is going to be done by manually adjusting the reg_lambda
                scaler = StandardScaler()
                scaler.fit(dat_tr)
                dat_tr = scaler.transform(dat_tr)
                dat_te = scaler.transform(dat_te)
                #ols solution
                ols=LinearRegression()
                ols.fit(dat_tr, Ztr)
                beta_ols=ols.coef_
                lambda_min=norm((dat_tr*beta_ols).T*Ztr)/len(Ztr) #all values are non-zeros
                lambda_max=norm((dat_tr.T*Ztr), np.inf)/len(Ztr) # all values are zero
                rango=[lambda_min]*np.logspace(-2, 1,10)
                reg_params[mov].append(rango)
                for rr, reg in enumerate(rango):
                
                    clf=GLM(distr='poisson', alpha=0.5, score_metric='pseudo_R2', reg_lambda=reg)
    
                    clf.fit(dat_tr, Ztr)
                    Ypre_te[mov].append(clf.predict(dat_te))
                    Ypre_tr[mov].append(clf.predict(dat_tr))
                    
                    r2_te = r2_score(Zte, clf.predict(dat_te))
                    if r2_te < 0: r2_te = 0
                    score_te[mov].append(r2_te)
                    r2_tr = r2_score(Ztr,clf.predict(dat_tr))
                    if r2_tr < 0: r2_tr = 0
                    score_tr[mov].append(r2_tr)
                          
                    Filters_ecog[mov].append(features_ecog.filters)
                    Patterns_ecog[mov].append(features_ecog.patterns)
                    
                    Filters_stn[mov].append(features_stn.filters)
                    Patterns_stn[mov].append(features_stn.patterns)
                    print(np.count_nonzero(clf.beta_))
                    if USED_MODEL == 'GLM':
                        Coef[mov].append(clf.beta_)
                    else:
                        Coef[mov].append(clf.coef_)
    
                    df[mov].append(np.count_nonzero(clf.beta_)) #degree of freedom
    
                del Xtr_ecog, Xte_ecog, Xtr_stn, Xte_stn

                # print(np.mean(score_te["CON"]))
       
                #%% save 
                predict_ = {
                    "y_pred_test": Ypre_te,
                    "y_test": Label_te,
                    "y_pred_train": Ypre_tr,
                    "y_train": Label_tr,
                    "score_tr": score_tr,
                    "score_te": score_te,
                    "filters_ecog": Filters_ecog,
                    "patterns_ecog": Patterns_ecog,
                    "filters_stn": Filters_stn,
                    "patterns_stn": Patterns_stn,
                    "coef": Coef,
                    "model_df": df,
                    "model": USED_MODEL,
                    "reg_values": reg_params,
                    "cv": 5
                  
                }
                
                out_path_file = os.path.join(settings['out_path_process']+ settings['num_patients'][s]+ 'dimreduction_combined_predictions_model'+USED_MODEL+'_'+str(subfolder[ss])+'.npy')
                np.save(out_path_file, predict_)        
                
                gc.collect()
            
                # # %% Plot the True mov and the predicted
                # fig, ax = plt.subplots(1, 1, figsize=[10, 4])
                # ind_best=np.argmax(score_te['CON'])
                # Ypre_te_best=Ypre_te['CON'][ind_best]
                # label_test_best=Label_te['CON'][ind_best]
                # #times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
                # ax.plot(Ypre_te_best, color='b', label='Predicted mov')
                # ax.plot(label_test_best, color='r', label='True mov')
                # ax.set_xlabel('Time (s)')
                # ax.set_ylabel('Movement')
                # ax.set_title('SPoC mov Predictions')
                # ax.text(0.33, 0.9, 'R2={:0.02f}'.format(score_te['CON'][ind_best]),
                # verticalalignment='bottom', horizontalalignment='right',
                # transform=ax.transAxes,fontsize=12) 
                # fig.suptitle('Subject_'+ settings['num_patients'][s], fontsize=14, fontweight='bold')
                # plt.legend()
                # plt.show()
                
                
