import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn
import sys

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)
sns.set_style('darkgrid')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Load Data
train = pd.read_csv('Train_data.csv\Train_data.csv')
test = pd.read_csv('Test_data.csv\Test_data.csv')

print(train.head(5))
print(train.shape)
print(train.info())
print(train.isnull().sum())

print(" Training Data has {} rows & {} columns".format(train.shape[0], train.shape[1]))
print(test.head(5))
print(test.shape)

print(" Test Data has {} rows & {} columns".format(test.shape[0], test.shape[1]))
print("Testing Data has {} rows & {} columns".format(test.shape[0], test.shape[1]))

print(train.describe())

