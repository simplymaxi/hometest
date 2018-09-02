import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



with h5py.File('src/groundtruth.h5','r') as act, h5py.File('src/predicted.h5','r') as pred:

    '''DATASETS GROUNDTRUTH TRUSTED RESULTS'''

    bbox_A = np.array(act['bbox2d'])
    label_name_A = np.array(act['label_name'])
    obj_type_A = np.array(act['obj_type'])

    '''actual data to pandas dataframe'''
    df_bbox_A = pd.DataFrame(bbox_A, columns=('Xa', 'Ya', 'X1a', 'Y1a'))
    df_label_name_A = pd.DataFrame(label_name_A, columns=['LabelA'])
    df_obj_type_A = pd.DataFrame(obj_type_A, columns=['obj_typeA'])

    '''merge to one dataframe ACTUAL label_name  label obj'''
    merge_actual_data = df_label_name_A.join(df_obj_type_A)

    # fix columns was as index
    # print merge_actual_data.sort_values('obj_typeA').groupby('LabelA').count().reset_index()
    '''!!!!!!!!!number of detectives for each picture!!!!!!!!!!!!!!!!!'''
    normalize_actual_data = merge_actual_data.sort_values('obj_typeA').groupby('LabelA').count().reset_index()



    '''DATASETS PREDICTED  RESULTS'''

    bbox_P = np.array(pred['bbox2d'])
    label_name_P = np.array(pred['label_name'])
    obj_type_P = np.array(pred['obj_type'])

    '''predicted data to pandas dataframe'''
    df_bbox_P = pd.DataFrame(bbox_P, columns=('Xp', 'Yp', 'X1p', 'Y1p'))
    df_label_name_P = pd.DataFrame(label_name_P, columns=['LabelP'])
    df_obj_type_P = pd.DataFrame(obj_type_P, columns=['obj_typeP'])

    '''merge predicted label obj '''
    merge_predicted_data = df_label_name_P.join(df_obj_type_P)

    '''prediction detectives for each picture #contains pictures from Actual data only'''
    ranged_merge_predicted_data = merge_predicted_data.loc[merge_predicted_data['LabelP'].isin(normalize_actual_data['LabelA'])]
    # print ranged_merge_predicted_data

    '''number of detectives for each picture by each detection class'''
    normalize_predicted_data = merge_predicted_data.groupby(['LabelP', 'obj_typeP']).size().unstack(fill_value=0).reset_index()
    # print normalize_predicted_data

    '''number of detectives for each picture by each detection class  #contains pictures from Actual data only'''
    ranged_normalize_pradicted_data = normalize_predicted_data.loc[normalize_predicted_data['LabelP'].isin(normalize_actual_data['LabelA'])]