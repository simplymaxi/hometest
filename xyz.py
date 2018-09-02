# import h5py
# import numpy as np
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import precision_recall_fscore_support as score
#
# with h5py.File('src/groundtruth.h5','r') as act, h5py.File('src/predicted.h5','r') as pred:
#     actual_data = np.array(act['obj_type'])
#     predicted_data = np.array([x.lower() if isinstance(x, str) else x for x in pred['obj_type'][:5928]])
#
#     precision, recall, fscore, support = score(actual_data, predicted_data,labels=['pedestrian','cyclist','none'])
#     print('precision: {}'.format(precision))
#     print('recall: {}'.format(recall))
#     print('fscore: {}'.format(fscore))
#     print('support: {}'.format(support))


import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



with h5py.File('src/groundtruth.h5','r') as act, h5py.File('src/predicted.h5','r') as pred:

    '''dataset actual'''
    bbox = np.array(act['bbox2d'])
    label_name = np.array(act['label_name'])
    obj_type = np.array(act['obj_type'])
    # print label_name
    '''actual data to pandas dataframe'''
    df1 = pd.DataFrame(bbox, columns=('Xa', 'Ya', 'X1a', 'Y1a'))
    df2 = pd.DataFrame(label_name, columns=['LabelA'])
    df3 = pd.DataFrame(obj_type, columns=['obj_typeA'])
    '''merge data label obj'''
    merge_actual_data = df2.join(df3)

    '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    normalize_actual_data = merge_actual_data.sort_values('obj_typeA').groupby('LabelA').count().reset_index()

    # fix columns was as index
    # print merge_actual_data.sort_values('obj_typeA').groupby('LabelA').count().reset_index()

    # print merge_actual_data
    '''dataset predicted'''
    bbox_2 = np.array(pred['bbox2d'])
    label_name_2 = np.array(pred['label_name'])
    obj_type_2 = np.array(pred['obj_type'])
    # print label_name_2

    '''predicted data to pandas dataframe'''
    df1_predicted = pd.DataFrame(bbox_2, columns=('Xp', 'Yp', 'X1p', 'Y1p'))
    df2_predicted = pd.DataFrame(label_name_2, columns=['LabelP'])
    df3_predicted = pd.DataFrame(obj_type_2, columns=['obj_typeP'])
    '''merge predicted label obj'''
    merge_predicted_data = df2_predicted.join(df3_predicted)
    # print merge_predicted_data
    ranged_merge_predicted_data = merge_predicted_data.loc[merge_predicted_data['LabelP'].isin(normalize_actual_data['LabelA'])]

    '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    normalize_predicted_data= merge_predicted_data.groupby(['LabelP','obj_typeP']).size().unstack(fill_value=0).reset_index()
    # print normalize_predicted_data['Pedestrian']
    # print normalize_predicted_data
    ranged_normalize_pradicted_data = normalize_predicted_data.loc[normalize_predicted_data['LabelP'].isin(normalize_actual_data['LabelA'])]

    Predict_A = np.array([x.lower() if isinstance(x, str) else x for x in ranged_merge_predicted_data['obj_typeP']])
    Gold_A = np.array(merge_actual_data['obj_typeA'][:5789])

    print confusion_matrix(Gold_A,Predict_A,labels=['pedestrian','cyclist','none'])
    print classification_report(Gold_A,Predict_A)
    
    # print normalize_actual_data['obj_typeA'].values
    # print normalize_predicted_data['Pedestrian'][]
    # print type(normalize_actual_data['obj_typeA'])
    # print confusion_matrix(normalize_actual_data['obj_typeA'].tolist(),ranged_normalize_pradicted_data['Pedestrian'].tolist())
    # A = normalize_actual_data['obj_typeA'].tolist()
    # P = ranged_normalize_pradicted_data['Pedestrian'].tolist()
    # C = []
    # for i in range(len(A)):
    #     if A[i] == P[i]:
    #         C.append(1)
    #     else:C.append(0)
    # print 'C'
    # print C
    # print len(C)
    # t,f = 0,0
    # for i in range(len(C)):
    #     if C[i] == 1:
    #         t+=1
    #     else: f+=1
    # print t, ' ',f
    # for i in range(len(A)):
    #     if A[i]!= 0:
    #         A[i] = 1
    # for i in range(len(P)):
    #     if P[i] != 0:
    #         P[i] = 1
    # print 'A'
    # print A
    # print len(A)
    # print 'P'
    # print P
    # print len(P)
    # print confusion_matrix(A,C)
    # print accuracy_score(A,C)
    # print classification_report(A,C)










    # # print merge_predicted_data
    # '''dataframes in one table'''
    # # print merge_predicted_data.join(merge_actual_data, lsuffix='_Pred', rsuffix='_Act')
    # print 'Dataframes by columns\n'
    # # print pd.concat([merge_predicted_data,merge_actual_data],axis=1)
    # # print merge
    # a = pd.concat([df1_predicted, df1], axis=1)
    # b = pd.concat([merge_predicted_data,merge_actual_data],axis=1)
    # # print pd.concat([a,b],axis=1)
    # all_data = pd.concat([a,b],axis=1)
    # all_data_2 = pd.DataFrame(all_data)
    # all_data_2.to_csv('out.csv', index=True,header=True)
    #
