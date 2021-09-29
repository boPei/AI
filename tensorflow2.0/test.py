import tensorflow as tf
import numpy as np
import pandas as pd
print(np.__version__)
print(tf.__version__)

data=pd.read_csv('data.csv')
data['values']=data.iloc[: , 1:].sum(axis=1)
dataset=data[data.values <= 3900]
print(data)
data.to_csv('dataset.csv')