import pandas as pd
import random
import numpy as np
import os

lbw = pd.read_csv('LBW_Dataset.csv')
age = lbw.Age.fillna(random.randint(21,26))
lbw['Age'] = age
lbw = lbw.drop('Education',axis=1)
residence =lbw['Residence'].values
residence[residence==2]=0
lbw['Residence'] = residence
res = lbw['Residence'].fillna(random.randint(1,2))
lbw['Residence'] = res
wts = lbw.Weight.fillna(random.randint(38,52))
lbw['Weight'] = wts
bp = lbw.BP.fillna(lbw.BP.mean())
lbw['BP'] = bp
lbw = lbw.drop('Delivery phase',axis=1)
r = list(np.arange(lbw.HB.mean()-2*lbw.HB.std() , lbw.HB.mean()+2*lbw.HB.std(),lbw.HB.std()))
hb = lbw.HB.fillna(random.sample(r,1)[0])
lbw['HB'] = hb
os.chdir('..')
lbw.to_csv('data/pre_processed.csv',index=False)
