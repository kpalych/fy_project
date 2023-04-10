import numpy as np
import pandas as pd
import requests

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

import sys, os
sys.path.append(os.path.join(os.path.abspath(''), '..', 'shared_libs'))
import data_transform

if __name__ == '__main__':
    df = pd.read_csv('../shared_libs/data/data_valid.csv', dtype={"zipcode": str})
    target = pd.read_csv('../shared_libs/data/data_valid_target.csv')
    
    r = requests.post('http://localhost/predict', json={'data': data_transform.prepare_for_json(df)})
    if r.status_code == 200:
        if r.json()['status'] == 'OK':
            y_pred = np.array(r.json()['predictions'])
            y = target['target']
            
            print('MAPE:', mean_absolute_percentage_error(y, y_pred)*100)
            print('RMSE:', mean_squared_error(y, y_pred)**0.5)
        else:
            print('ERROR...')
            print(r.json()['error'])
    else:
        print('ERROR 500')
        print(r.text)
