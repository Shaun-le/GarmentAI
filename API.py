# Import libraries
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import hashlib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from labels import TSPS

app = Flask(__name__)

def hash_modulo(val, mod):
    md5 = hashlib.md5()
    md5.update(str(val).encode())
    return int(md5.hexdigest(), 16) % mod

class FeatureHasher(BaseEstimator, TransformerMixin):
    def __init__(self, num_buckets: int):
        self.num_buckets = num_buckets

    def fit(self, X: pd.Series):
        return self

    def transform(self, X: pd.Series):
        return X.apply(lambda x: hash_modulo(x, self.num_buckets))

fh = FeatureHasher(num_buckets=1600)

def hash_encode(data):
    dt = pd.Series(extract_values(data))
    data_transformed = fh.fit_transform(dt)
    return data_transformed

def pred_and_decode_classifier(model,data,label,list):
    data_transformed = hash_encode(data)
    pred = model.predict([data_transformed])[0]
    result = []
    for i, p in enumerate(pred):
        if p == 1:
            result.append(label[i])

    for q in list:
        if len(result) == len(q) and all(x in result for x in q) and all(x in q for x in result):
            result = q
            break

    re = ", ".join(result)
    r = jsonify({"prediction": re})
    return r

def pred_and_decode_classifier_TKDT(model,data,label):
    data_transformed = pd.Series(extract_strings_and_numbers_from_dict(data))
    pred = model.predict([data_transformed])[0]
    encoder = LabelEncoder()
    encoder.fit_transform(label)
    re = encoder.inverse_transform([pred])
    r = jsonify({"prediction": re[0]})
    return r

def predict_regression(model,data):
    data_transformed = hash_encode(data)
    pred = model.predict(data_transformed.values.reshape(1, -1))
    r = jsonify({"prediction": pred[0]})
    return r

def predict_regression_multi(model,data):
    data_transformed = extract_strings_and_numbers_from_dict(data)
    pred = model.predict(data_transformed.values.reshape(1, -1))
    re = []
    for i in pred[0]:
        re.append(i)
    r = jsonify({"prediction": re})
    return r

def extract_values(data):
    values = []
    for key in sorted(data.keys()):
        if key.startswith("X"):
            values.append(data[key])
    return values

def extract_strings_and_numbers_from_dict(my_dict):
    string_list = []
    number_list = []
    for key, value in my_dict.items():
        if key.startswith('X'):
            if isinstance(value, str):
                string_list.append(value)
            elif isinstance(value, (int, float)):
                number_list.append(value)
    string_list_hash = fh.fit_transform(pd.Series(string_list))
    number_list = pd.Series(number_list)
    s  = string_list_hash.append(number_list)
    return s


def select_model(model_path:str):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

@app.route('/api', methods = ['POST'])
def main():
    data = request.get_json(force=True)
    if data.get('Product') == "TSPS":
        if data.get('Task') == "NS":
            model = select_model('model/T-shirt/NS_TSPS.pkl')
            result = predict_regression(model,data)
        elif data.get('Task') == "DMC":
            model = select_model('model/T-shirt/DMC_TSPS.pkl')
            result = predict_regression_multi(model,data)
        elif data.get('Task') == "DMV":
            model = select_model('model/T-shirt/DMV_TSPS.pkl')
            result = predict_regression_multi(model,data)
        elif data.get('Task') == 'NCLD':
            model = select_model('model/T-shirt/NCLD_TSPS.pkl')
            result = predict_regression_multi(model,data)
        elif data.get('Task') == 'DMTB':
            model = select_model('model/T-shirt/DMTB_TSPS.pkl')
            result = predict_regression_multi(model,data)
        elif data.get('Task') == "DL":
            model = select_model('model/T-shirt/DL_TSPS.pkl')
            result = pred_and_decode_classifier(model,data,TSPS.DL, TSPS.DL_original)
        elif data.get('Task') == "QTCN":
            model = select_model('model/T-shirt/QTCN_TSPS.pkl')
            result = pred_and_decode_classifier(model,data,TSPS.QTCN,TSPS.QTCN_original)
        elif data.get('Task') == "TKDT":
            model = select_model('model/T-shirt/TKDT_TSPS.pkl')
            result =pred_and_decode_classifier_TKDT(model,data,TSPS.TKDT_original)
        elif data.get('Task') == 'CD':
            model = select_model('model/T-shirt/CD_TSPS.pkl')
            result = pred_and_decode_classifier(model,data,TSPS.CD,TSPS.CD_original)
        elif data.get('Task') == 'TCKT':
            model = select_model('model/T-shirt/TCKT_TSPS.pkl')
            result = pred_and_decode_classifier(model,data,TSPS.TCKT, TSPS.TCKT_original)
    elif data.get('Product') == "SM":
        if data.get("Task") == "QTCN":
            result = jsonify({"prediction":"test"})
    else: result = jsonify({"Prediction":"Incorrect Product"})
    return result

if __name__ == '__main__':
    app.run(port=5005, debug=True)
