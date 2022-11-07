from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from IPython.core.ultratb import ColorTB
sys.excepthook = ColorTB()

from loadData import *

def plotProportionCDR():

    dataxls = loadXLS()

    CDR = dataxls['CDR']

    proportionCDR = np.zeros((1,3))

    for i in range(CDR.shape[0]):
        if CDR[i] == 0:
            proportionCDR[0,0] += 1
        if CDR[i] == 0.5:
            proportionCDR[0,1] += 1
        if CDR[i] == 1:
            proportionCDR[0,2] += 1

    #plt.hist([206, 123, 41], [0,1,2])
    strength = []
    strength.append(int(proportionCDR[0,0]))
    strength.append(int(proportionCDR[0,1]))
    strength.append(int(proportionCDR[0,2]))
    
    plt.bar(['CDR=0','CDR=0.5','CDR=1'], strength)
    
    plt.title('proportion of CDR')
    plt.ylabel('Number of MRI sessions')

def plotProportionNC_C_D():

    dataxls = loadXLS()
    Group = dataxls['Group']

    proportionNCCD = np.zeros((1,3))

    for i in range(Group.shape[0]):
        if Group[i] == 'Nondemented':
            proportionNCCD[0,0] += 1
        if Group[i] == 'Demented':
            proportionNCCD[0,1] += 1            
        if Group[i] == 'Converted':
            proportionNCCD[0,2] += 1

    grp = []
    grp.append(int(proportionNCCD[0,0]))
    grp.append(int(proportionNCCD[0,1]))
    grp.append(int(proportionNCCD[0,2]))
    
    plt.bar(['Nondemented','Demented','Converted'], grp)
    
    plt.title('proportion of Nondemented, Demented et Converted')
    plt.ylabel('Number of MRI sessions')

def plotAge():

    dataxls = loadXLS()
    Age = dataxls['Age']
    


def plotStat():
    plt.subplot(1,2,1)
    plotProportionCDR()
    plt.subplot(1,2,2)
    plotProportionNC_C_D()
    plt.show()