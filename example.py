#IMPORT
import joblib
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from fs_method import evaluate_model

#LOAD DATA
test = pd.read_csv(f'{os.getcwd()}\Datasets\Subset2_Test_Set.csv')
x_test=test[test.columns.values].drop(columns="lower flammability limit (vol%)") 
y_test=test[['lower flammability limit (vol%)']]

#LOAD SCALER AND MODEL
scx = joblib.load('Mixed_Subset_Feature_Scaler.pkl')
scy = joblib.load('Mixed_Subset_Target_Scaler.pkl')
model = pickle.load(open('Mixed_LASSO_model.sav', 'rb'))

#SCALE DATA
x_test_scaled = scx.transform(x_test)
y_pred_scaled = model.predict(x_test_scaled)
y_pred = scy.inverse_transform(np.reshape(y_pred_scaled,(-1,1)))
print(r2_score(y_test,y_pred))


