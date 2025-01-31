'''
Imagine you're working as a data scientist at Google, and you're tasked with developing a predictive model to optimize the placement of advertisements on YouTube videos.
The goal is to maximize the click-through rate (CTR) while minimizing the cost per click (CPC). You have access to a large dataset containing information about:
  
  Video metadata (title, description, tags, etc.)
  Ad metadata (ad creative, targeting options, etc.)
  User behavior (watch history, search queries, etc.)
  Ad performance metrics (CTR, CPC, etc.)
'''
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
np.random.seed(42)

# Create a DataFrame with random data
data = pd.read_csv('youtube_ads.csv')

x = data.drop(columns = ['ctr','cpc'], axis = 1 )
y_ctr = data['ctr']
y_cpc = data['cpc']

scaler = StandardScaler() 
x_scaled = scaler.fit_transform(x)

x_train , x_test, ctr_train, ctr_test , cpc_train, cpc_test = train_test_split(x_scaled, y_ctr, y_cpc,test_size = 0.2, random_state =  42) 

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score , precision_recall_curve

ctr_model = RandomForestClassifier( n_estimators= 200, random_state = 42)
ctr_model.fit(x_train, ctr_train) 

ctr_predict = ctr_model.predict(x_test)

print(f'roc auc score ctr : {roc_auc_score(ctr_test,ctr_predict)}')
print(f'prevision recall curve ctr : {precision_recall_curve(ctr_test,ctr_predict)}')


from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error 

cpc_model = RandomForestRegressor(n_estimators = 200, random_state = 42) 
cpc_model.fit(x_train, cpc_train)

cpc_predict = cpc_model.predict(x_test)

print(f'mean absolute error : {mean_absolute_error(cpc_test, cpc_predict)}')

ctr_weight = 0.7 
cpc_weight = 0.3

final_score = ctr_weight * ctr_predict - cpc_weight * cpc_predict 

top_10_bast_ads = data.iloc[final_score.argsort()[:10]]

 '''
 if your ctr in continuouse '''

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error 

np.random.seed(42)


# Create a DataFrame with random data
data = pd.DataFrame({
    'views': np.random.randint(1000, 10000, 10),
    'likes': np.random.randint(100, 1000, 10),
    'shares': np.random.randint(50, 500, 10),
    'ctr': np.random.rand(10),
    'cpc': np.random.rand(10)
})

# Splitting Features and Targets
x = data.drop(columns=['ctr', 'cpc'], axis=1)
y_ctr = data['ctr']  # Continuous CTR
y_cpc = data['cpc']  # Continuous CPC

# Standardizing Features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Splitting Data
x_train, x_test, ctr_train, ctr_test, cpc_train, cpc_test = train_test_split(
    x_scaled, y_ctr, y_cpc, test_size=0.2, random_state=42
)

# Using RandomForestRegressor for CTR (since it's continuous)
ctr_model = RandomForestRegressor(n_estimators=200, random_state=42)
ctr_model.fit(x_train, ctr_train) 
ctr_predict = ctr_model.predict(x_test)  # Predicting continuous CTR

# Using RandomForestRegressor for CPC (since it's continuous)
cpc_model = RandomForestRegressor(n_estimators=200, random_state=42) 
cpc_model.fit(x_train, cpc_train)
cpc_predict = cpc_model.predict(x_test)

# Evaluating the models
print(f'Mean Absolute Error for CTR: {mean_absolute_error(ctr_test, ctr_predict)}')
print(f'Mean Absolute Error for CPC: {mean_absolute_error(cpc_test, cpc_predict)}')

# Assigning Weights
ctr_weight = 0.7 
cpc_weight = 0.3

# Final Score Calculation
final_score = ctr_weight * ctr_predict - cpc_weight * cpc_predict

# Selecting Top 10 Best Ads
top_10_best_ads = data.iloc[np.argsort(final_score)[-10:]]  # Highest scores

print("\nTop 10 Best Ads based on CTR & CPC weighting:")
print(top_10_best_ads)
