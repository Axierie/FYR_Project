#IMPORT
import joblib
import pickle
import pandas as pd
import numpy as np

#LOAD DATA
x_test = pd.read_csv(f'MixedDescriptorExample.csv')

#LOAD SCALER AND MODEL
scx = joblib.load('Mixed_Subset_Feature_Scaler.pkl')
scy = joblib.load('Mixed_Subset_Target_Scaler.pkl')
model = pickle.load(open('Mixed_SVR_model.sav', 'rb'))

#SCALE DATA
x_test_scaled = scx.transform(x_test)
y_pred_scaled = model.predict(x_test_scaled)
y_pred = scy.inverse_transform(np.reshape(y_pred_scaled,(-1,1))) #Predicted data, unscaled






