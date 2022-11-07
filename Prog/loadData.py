import numpy as np
import os
import nibabel as nib
import pandas as pd
import sys
from IPython.core.ultratb import ColorTB
from tqdm import tqdm
sys.excepthook = ColorTB()

#train : 70%
#test : 15%
#val : 15%

def loadXLS():
    """load xls file"""
    dataxls = pd.read_excel('real_oasis_longitudinal_demographics.xlsx')
    return dataxls

def lstCaseSubjectsN_ND_C(dataxls):

    """return 3 list with name of Demented, Non Demented and Converted Subjects"""

    valueCounts = dataxls['Group'].value_counts()

    numNDTrain = int(valueCounts[0]*0.70) + 1 #to have all subjects
    numNDTest = int(valueCounts[0]*0.15)
    numNDval = int(valueCounts[0]*0.15)

    numDTrain = int(valueCounts[1]*0.70) + 2 #to have all subjects
    numDTest = int(valueCounts[1]*0.15)
    numDval = int(valueCounts[1]*0.15)

    numCTrain = int(valueCounts[2]*0.70) + 2 #to have all subjects
    numCTest = int(valueCounts[2]*0.15)
    numCval = int(valueCounts[2]*0.15)

    lstD = []
    lstND = []
    lstC = []

    for i in range(dataxls.shape[0]):
        if dataxls['Group'][i] == 'Demented' and dataxls['Subject ID'][i] not in lstD:
            lstD.append(dataxls['Subject ID'][i])
        if dataxls['Group'][i] == 'Nondemented' and dataxls['Subject ID'][i] not in lstND:
            lstND.append(dataxls['Subject ID'][i])
        if dataxls['Group'][i] == 'Converted' and dataxls['Subject ID'][i] not in lstC:
            lstC.append(dataxls['Subject ID'][i])

    print(f'Number of Demented subjects : {len(lstD)}\nNumber of Non Demented Subjets : {len(lstND)}\nNumber of Converted Subjects : {len(lstC)}') 
    return lstD, lstND, lstC

def lstByCDRValues(dataxls):

    lst0 = []
    lst05 = []
    lst1= []

    for i in range(dataxls.shape[0]):
        if dataxls['CDR'][i] == 0:
            if dataxls['Subject ID'][i]+'_MR1' in lst0 :
                lst0.append(dataxls['Subject ID'][i]+'_MR2')
            elif dataxls['Subject ID'][i]+'_MR2' in lst0 :
                lst0.append(dataxls['Subject ID'][i]+'_MR3')
            elif dataxls['Subject ID'][i]+'_MR3' in lst0 :
                lst0.append(dataxls['Subject ID'][i]+'_MR4')
            else:
                lst0.append(dataxls['Subject ID'][i]+'_MR1')
        if dataxls['CDR'][i] == 0.5:
            if dataxls['Subject ID'][i]+'_MR1' in lst05 :
                lst05.append(dataxls['Subject ID'][i]+'_MR2')
            elif dataxls['Subject ID'][i]+'_MR2' in lst05 :
                lst05.append(dataxls['Subject ID'][i]+'_MR3')
            elif dataxls['Subject ID'][i]+'_MR3' in lst05 :
                lst05.append(dataxls['Subject ID'][i]+'_MR4')
            else:
                lst05.append(dataxls['Subject ID'][i]+'_MR1')
        if dataxls['CDR'][i] == 1:
            if dataxls['Subject ID'][i]+'_MR1' in lst1 :
                lst1.append(dataxls['Subject ID'][i]+'_MR2')
            elif dataxls['Subject ID'][i]+'_MR2' in lst1 :
                lst1.append(dataxls['Subject ID'][i]+'_MR3')
            elif dataxls['Subject ID'][i]+'_MR3' in lst1 :
                lst1.append(dataxls['Subject ID'][i]+'_MR4')
            else:
                lst1.append(dataxls['Subject ID'][i]+'_MR1')

    print(f'Number of CDR=0 MRI : {len(lst0)}\nNumber of CDR=0.5 MRI : {len(lst05)}\nNumber of CDR=1 MRI : {len(lst1)}') 

    print(f'CDR=0   --> lenTrainSet : {int(len(lst0)*0.7)+2}, lenValidationSet : {int(len(lst0)*0.15)}, lenTestSet : {int(len(lst0)*0.15)}')
    print(f'CDR=0.5 --> lenTrainSet : {int(len(lst05)*0.7)+1}, lenValidationSet : {int(len(lst05)*0.15)}, lenTestSet : {int(len(lst05)*0.15)}')
    print(f'CDR=1   --> lenTrainSet : {int(len(lst1)*0.7)+1}, lenValidationSet : {int(len(lst1)*0.15)}, lenTestSet : {int(len(lst1)*0.15)}')
    return lst0, lst05, lst1


def lstTrainValTestCDR(dataxls):
    lst0, lst05, lst1 = lstByCDRValues(dataxls)
    yTrain = np.zeros((262,1))
    yVal = np.zeros((54,1))
    yTest = np.zeros((54,1))

    lstTrain = lst0[0:146] + lst05[0:87] + lst1[0:29]
    yTrain[0:146,0], yTrain[146:146+87,0], yTrain[146+87:,0] = 0, 1, 2
    
    lstVal = lst0[146:146+30] + lst05[87:87+18] + lst1[29:29+6]
    yVal[0:30,0], yVal[30:30+18,0], yVal[30+18:,0] = 0, 1, 2
    
    lstTest = lst0[146+30:] + lst05[87+18:] + lst1[29+6:]
    yTest[0:30,0], yTest[30:30+18,0], yTest[30+18:,0] = 0, 1, 2

    return lstTrain, lstVal, lstTest, yTrain, yVal, yTest

def loadData(dataxls):

    if os.path.isfile('dataTrain.npy') and os.path.isfile('dataVal.npy') and os.path.isfile('dataTest.npy'):
        dataTrain = np.load('dataTrain.npy')
        print('dataTrain load')
        dataVal = np.load('dataVal.npy')
        print('dataVal load')
        dataTest = np.load('dataTest.npy')
        print('dataTest load')

    else :
        os.chdir('DataOASIS2/')

        lstTrain, lstVal, lstTest, _, _, _ = lstTrainValTestCDR(dataxls)
        
        dataTrain = np.zeros((len(lstTrain), 256, 256, 128, 1))
        dataVal = np.zeros((len(lstVal), 256, 256, 128, 1))
        dataTest = np.zeros((len(lstTest), 256, 256, 128, 1))

        for i, name in tqdm(enumerate(lstTrain)):
            scan = nib.load(name+'/RAW/mpr-1.nifti.img')
            scan = scan.get_fdata() 
            dataTrain[i,:,:,:,:] = scan
        for i, name in tqdm(enumerate(lstVal)):
            scan = nib.load(name+'/RAW/mpr-1.nifti.img')
            scan = scan.get_fdata()
            dataVal[i,:,:,:,:] = scan
        for i, name in tqdm(enumerate(lstTest)):
            scan = nib.load(name+'/RAW/mpr-1.nifti.img')
            scan = scan.get_fdata()
            dataTest[1,:,:,:,:] = scan

        os.chdir('../')

        np.save('dataTrain.npy', dataTrain)
        np.save('dataVal.npy', dataVal)
        np.save('dataTest.npy', dataTest)

    return dataTrain, dataVal, dataTest

