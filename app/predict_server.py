import numpy as np
import pandas as pd
import json
import pickle
from flask import Flask, request, jsonify

import sys, os
sys.path.append(os.path.join(os.path.abspath(''), '..', 'shared_libs'))
import data_transform

app = Flask(__name__)

@app.errorhandler(Exception)
def all_exception_handler(error):
    return jsonify({
        'status': 'ERROR',
        'error': str(error)
    }), 200


@app.route('/')
def index_action():
    return 'Use POST method to send data to /predict URI', 200

@app.route('/predict', methods=['POST'])
def predict_action():
    data = data_transform.convert_to_data_frame(np.array(request.json.get('data')))
    
    df = data_transform.clear_data_base_line(
        data, 
        '../shared_libs/data/default_values.pkl', 
        can_drop_rows=False, 
        force_rebuild_cached_data=False
    )
    
    cities_dict = data_transform.get_cities_dict('../shared_libs/data/cities_dict.pkl', force_read=True)
    address_dict = data_transform.get_addresses_dict('../shared_libs/data/address_dict.pkl', force_read=True)
    address_by_zip_dict = data_transform.get_address_by_zipcode_dict('../shared_libs/data/address_by_zip_dict.pkl', force_read=True)
    cities_clusters_dict = data_transform.get_citiess_clusters_dict('../shared_libs/data/cities_clusters_dict.pkl', force_read=True)
    
    df = data_transform.fix_incorrect_states_and_cities(
        df, 
        '../shared_libs/data/default_values.pkl', 
        cities_dict, 
        can_drop_rows=False
    )
    
    df = data_transform.add_city_features(
        df,
        '../shared_libs/data/default_values.pkl', 
        cities_dict, 
        address_dict,
        address_by_zip_dict,
        cities_clusters_dict,
        force_rebuild_cached_data=False
    )
    
    df = data_transform.add_population_features(
        df, 
        '../shared_libs/data/default_values.pkl', 
        '../shared_libs/data/uscities.csv', 
        can_drop_rows=False,
        force_rebuild_cached_data=False
    )
    
    df = data_transform.encode_state_and_city(
        df, 
        '../shared_libs/data/default_values.pkl', 
        can_drop_rows=False, 
        force_rebuild_cached_data=False
    )
    
    df = data_transform.final_tune_pca_and_scale(
        df, 
        '../shared_libs/data/default_values.pkl', 
        force_rebuild_cached_data=False
    )
    
    model = data_transform.get_current_prediction_model('../shared_libs/data/models', 'model_abr')
    
    y_pred = model.predict(df)
    
    return jsonify({
        'status': 'OK',
        'predictions': y_pred.tolist()
    }), 200

if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
    

