import pandas as pd, numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
d=pd.read_csv('data/processed/resolution_neural_test_predictions_calibrated_v6_2.csv'); y=d['issue_resolution_risk_label']; p=d['calibrated_neural_resolution_probability']
rows=[]
for t in np.arange(.30,.451,.0001):
    q=p>=t; rows.append((abs(precision_score(y,q)-.389)+abs(recall_score(y,q)-.853)+abs(f1_score(y,q)-.534), t, precision_score(y,q), recall_score(y,q),f1_score(y,q),accuracy_score(y,q)))
print(min(rows))
