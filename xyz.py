import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support


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
    '''      LabelA   obj_typeA
        0     000007  pedestrian
        1     000007  pedestrian
        2     000009  pedestrian
        3     000014  pedestrian
        4     000017  pedestrian
        5     000017  pedestrian
        6     000017  pedestrian
        7     000017  pedestrian
        8     000017  pedestrian
        9     000017  pedestrian
        10    000020  pedestrian
    '''
    merge_actual_data = df_label_name_A.join(df_obj_type_A)
    #print merge_actual_data

    # fix columns was as index
    # print merge_actual_data.sort_values('obj_typeA').groupby('LabelA').count().reset_index()
    '''!!!!!!!!!number of detectives for each picture!!!!!!!!!!!!!!!!!'''
    '''      LabelA   obj_typeA
        0     000007          2
        1     000009          1
        2     000014          1
        3     000017          6
        4     000020          3
        5     000021          1
        6     000025          1
        7     000031          3
        8     000036          1
        9     000040          1
        10    000042          1
    '''
    normalize_actual_data = merge_actual_data.sort_values('obj_typeA').groupby('LabelA').count().reset_index()
    # print normalize_actual_data


    '''DATASETS PREDICTED  RESULTS'''

    bbox_P = np.array(pred['bbox2d'])
    label_name_P = np.array(pred['label_name'])
    obj_type_P = np.array(pred['obj_type'])

    '''predicted data to pandas dataframe'''
    df_bbox_P = pd.DataFrame(bbox_P, columns=('Xp', 'Yp', 'X1p', 'Y1p'))
    df_label_name_P = pd.DataFrame(label_name_P, columns=['LabelP'])
    df_obj_type_P = pd.DataFrame(obj_type_P, columns=['obj_typeP'])

    '''merge predicted label obj '''
    '''       LabelP    obj_typeP
        0      000000        None
        1      000001        None
        2      000002        None
        3      000003        None
        4      000004        None
        5      000005        None
        6      000006  Pedestrian
        7      000007        None
        8      000008        None
        9      000009  Pedestrian
        10     000010        None
    '''
    merge_predicted_data = df_label_name_P.join(df_obj_type_P)
    # print merge_predicted_data

    '''prediction detectives for each picture #contains pictures from Actual data only'''
    '''       index   LabelP    obj_typeP
        0         7  000007        None
        1         9  000009  Pedestrian
        2        14  000014  Pedestrian
        3        17  000017  Pedestrian
        4        18  000017  Pedestrian
        5        19  000017  Pedestrian
        6        20  000017  Pedestrian
        7        21  000017  Pedestrian
        8        22  000017  Pedestrian
        9        23  000017  Pedestrian
        10       24  000017  Pedestrian
        11       27  000020  Pedestrian
        12       28  000020  Pedestrian
    '''
    ranged_merge_predicted_data = merge_predicted_data.loc[merge_predicted_data['LabelP'].isin(normalize_actual_data['LabelA'])].reset_index()
    # print ranged_merge_predicted_data


    '''number of detectives for each picture by each detection class'''
    '''            LabelP  Cyclist  None  Pedestrian
        0          000000        0     1           0
        1          000001        0     1           0
        2          000002        0     1           0
        3          000003        0     1           0
        4          000004        0     1           0
        5          000005        0     1           0
        6          000006        0     0           1
        7          000007        0     1           0
        8          000008        0     1           0
        9          000009        0     0           1
        10         000010        0     1           0
    '''
    normalize_predicted_data = merge_predicted_data.groupby(['LabelP','obj_typeP']).size().unstack(fill_value=0).reset_index()
    # print normalize_predicted_data

    '''number of detectives for each picture by each detection class  #contains pictures from Actual data only'''
    '''['index' 'LabelP' 'Cyclist' 'None' 'Pedestrian']'''
    '''             index  LabelP  Cyclist  None  Pedestrian
        0              7  000007        0     1           0
        1              9  000009        0     0           1
        2             14  000014        0     0           1
        3             17  000017        0     0           8
        4             20  000020        0     0           2
        5             21  000021        0     1           0
        6             25  000025        0     0           1
        7             31  000031        0     0           4
        8             36  000036        0     1           0
        9             40  000040        0     0           1
        10            42  000042        1     0           0
    '''
    ranged_normalize_pradicted_data = normalize_predicted_data.loc[normalize_predicted_data['LabelP'].isin(normalize_actual_data['LabelA'])].reset_index()
    # print ranged_normalize_pradicted_data


    '''#######PRECISION RECALL METRICS##########'''

