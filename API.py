# Import libraries
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import hashlib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from labels import TSPS, SM, Q, J, BHLD, Vest

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
    #print(data_transformed)
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

def predict_regression_V2(model,data):
    data_transformed = extract_strings_and_numbers_from_dict(data)
    #print(data_transformed)
    pred = model.predict(data_transformed.values.reshape(1, -1))
    r = jsonify({"prediction": pred[0][0]})
    return r

def predict_regression_V3(model,data):
    data_transformed = extract_strings_and_numbers_from_dict(data)
    print(data_transformed)
    pred = model.predict(data_transformed.values.reshape(1, -1))
    r = jsonify({"prediction": float(pred[0][0])})
    return r

def predict_regression_multi(model,data):
    data_transformed = extract_strings_and_numbers_from_dict(data)
    pred = model.predict(data_transformed.values.reshape(1, -1))
    re = []
    for i in [round(float(i), 2) for i in pred[0]]:
        re.append(i)
    r = jsonify({"prediction": re})
    return r

def extract_values(data):
    values = []
    for key in data.keys():
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
    product = data.get('Product')
    task = data.get('Task')
    result = {}

    if product == 'TSPS':
        model_map = {
            'NS': 'model/T-shirt/NS_TSPS.pkl',
            'TGGC': 'model/T-shirt/TGGC_TSPS.pkl',
            'DMC': 'model/T-shirt/DMC_TSPS.pkl',
            'DMV': 'model/T-shirt/DMV_TSPS.pkl',
            'NCLD': 'model/T-shirt/NCLD_TSPS.pkl',
            'DMTB': 'model/T-shirt/DMTB_TSPS.pkl',
            'DL': 'model/T-shirt/DL_TSPS.pkl',
            'QTCN': 'model/T-shirt/QTCN_TSPS.pkl',
            'TKDC': 'model/T-shirt/TKDT_TSPS.pkl',
            'CD': 'model/T-shirt/CD_TSPS.pkl',
            'TCKT': 'model/T-shirt/TCKT_TSPS.pkl',
        }
        if task in model_map:
            model_path = model_map[task]
            if task in ['NS','TGGC']:
                result = predict_regression(select_model(model_path),data)
            elif task == 'DMC':
                result = predict_regression_V2(select_model(model_path), data)
            elif task == 'DMV':
                result = predict_regression_V3(select_model(model_path), data)
            elif task == 'DL':
                result = pred_and_decode_classifier(select_model(model_path), data, TSPS.DL, TSPS.DL_original)
            elif task == 'QTCN':
                result = pred_and_decode_classifier(select_model(model_path), data, TSPS.QTCN, TSPS.QTCN_original)
            elif task == 'TKDC':
                result = pred_and_decode_classifier_TKDT(select_model(model_path), data, TSPS.TKDT_original)
            elif task in ['CD', 'TCKT']:
                result = pred_and_decode_classifier(select_model(model_path), data, getattr(TSPS, task), getattr(TSPS, task+'_original'))
            else:
                result = predict_regression_multi(select_model(model_path), data)
        else:
            result = jsonify({"Prediction": "Incorrect Task"})
    elif product == 'SM':
        model_map = {
            'NS': 'model/Sơ Mi/NS_SM.pkl',
            'TGGC': 'model/Sơ Mi/TGGC_SM.pkl',
            'DMC': 'model/Sơ Mi/DMC_SM.pkl',
            'DMV': 'model/Sơ Mi/DMV_SM.pkl',
            'NCLD': 'model/Sơ Mi/NCLD_SM.pkl',
            'DMTB': 'model/Sơ Mi/DMTB_SM.pkl',
            'DL': 'model/Sơ Mi/DL_SM.pkl',
            'QTCN': 'model/Sơ Mi/QTCN_SM.pkl',
            'TKDC': 'model/Sơ Mi/TKDC_SM.pkl',
            'CD': 'model/Sơ Mi/CD_SM.pkl',
            'TCKT': 'model/Sơ Mi/TCKT_SM.pkl',
        }
        if task in model_map:
            model_path = model_map[task]
            if task in ['NS','TGGC']:
                result = predict_regression(select_model(model_path), data)
            elif task == 'DMC':
                result = predict_regression_V2(select_model(model_path), data)
            elif task == 'DL':
                result = pred_and_decode_classifier(select_model(model_path), data, SM.DL, SM.DL_original)
            elif task == 'QTCN':
                result = pred_and_decode_classifier(select_model(model_path), data, SM.QTCN, SM.QTCN_original)
            elif task == 'TKDC':
                result = pred_and_decode_classifier_TKDT(select_model(model_path), data, SM.TKDC_original)
            elif task in ['CD', 'TCKT']:
                result = pred_and_decode_classifier(select_model(model_path), data, getattr(SM, task),
                                                    getattr(SM, task + '_original'))
            else:
                result = predict_regression_multi(select_model(model_path), data)
        else:
            result = jsonify({"Prediction": "Incorrect Task"})
    elif product == "Q":
        model_map = {
            'NS': 'model/Quần/NS_Q.pkl',
            'TGGC': 'model/Quần/TGGC_Q.pkl',
            'DMC': 'model/Quần/DMC_Q.pkl',
            'DMV': 'model/Quần/DMV_Q.pkl',
            'NCLD': 'model/Quần/NCLD_Q.pkl',
            'DMTB': 'model/Quần/DMTB_Q.pkl',
            'DL': 'model/Quần/DL_Q.pkl',
            'QTCN': 'model/Quần/QTCN_Q.pkl',
            'TKDC': 'model/Quần/TKDC_Q.pkl',
            'CD': 'model/Quần/CD_Q.pkl',
            'TCKT': 'model/Quần/TCKT_Q.pkl',
        }
        if task in model_map:
            model_path = model_map[task]
            if task in ['NS','TGGC']:
                result = predict_regression(select_model(model_path), data)
            elif task == 'DMV':
                result = predict_regression_V3(select_model(model_path), data)
            elif task == 'DL':
                result = pred_and_decode_classifier(select_model(model_path), data, Q.DL, Q.DL_original)
            elif task == 'QTCN':
                result = pred_and_decode_classifier(select_model(model_path), data, Q.QTCN, Q.QTCN_original)
            elif task == 'TKDC':
                result = pred_and_decode_classifier_TKDT(select_model(model_path), data, Q.TKDC_original)
            elif task in ['CD', 'TCKT']:
                result = pred_and_decode_classifier(select_model(model_path), data, getattr(Q, task),
                                                    getattr(Q, task + '_original'))
            else:
                result = predict_regression_multi(select_model(model_path), data)
        else:
            result = jsonify({"Prediction": "Incorrect Task"})
    elif product == "J":
        model_map = {
            'NS': 'model/Jacket/NS_J.pkl',
            'TGGC': 'model/Jacket/TGGC_J.pkl',
            'DMC': 'model/Jacket/DMC_J.pkl',
            'DMVC': 'model/Jacket/DMVC_J.pkl',
            'DMVL': 'model/Jacket/DMVL_J.pkl',
            'NCLD': 'model/Jacket/NCLD_J.pkl',
            'DMTB': 'model/Jacket/DMTB_J.pkl',
            'DL': 'model/Jacket/DL_J.pkl',
            'QTCN': 'model/Jacket/QTCN_J.pkl',
            'TKDC': 'model/Jacket/TKDC_J.pkl',
            'CD': 'model/Jacket/CD_J.pkl',
            'TCKT': 'model/Jacket/TCKT_J.pkl',
        }
        if task in model_map:
            model_path = model_map[task]
            if task in ['NS', 'TGGC']:
                result = predict_regression(select_model(model_path), data)
            elif task == 'DL':
                result = pred_and_decode_classifier(select_model(model_path), data, J.DL, J.DL_original)
            elif task == 'QTCN':
                result = pred_and_decode_classifier(select_model(model_path), data, J.QTCN, J.QTCN_original)
            elif task == 'TKDC':
                result = pred_and_decode_classifier_TKDT(select_model(model_path), data, J.TKDC_original)
            elif task in ['CD', 'TCKT']:
                result = pred_and_decode_classifier(select_model(model_path), data, getattr(J, task),
                                                    getattr(J, task + '_original'))
            else:
                result = predict_regression_multi(select_model(model_path), data)
        else:
            result = jsonify({"Prediction": "Incorrect Task"})
    elif product == "BHLD":
        model_map = {
            'NS': 'model/BHLD/NS_BH.pkl',
            'DMC': 'model/BHLD/DMC_BH.pkl',
            'DMV': 'model/BHLD/DMV_BH.pkl',
            'NCLD': 'model/BHLD/NCLD_BH.pkl',
            'DMTB': 'model/BHLD/DMTB_BH.pkl',
            'DL': 'model/BHLD/DL_BH.pkl',
            'QTCN': 'model/BHLD/QTCN_BH.pkl',
            'TKDC': 'model/Quần/TKDC_Q.pkl',
            'CD': 'model/BHLD/CD_BH.pkl',
            'TCKT': 'model/BHLD/TCKT_BH.pkl',
            'TGGC': 'model/BHLD/TGGC_BH.pkl'
        }
        if task in model_map:
            model_path = model_map[task]
            if task in ['NS','TGGC']:
                result = predict_regression(select_model(model_path), data)
            elif task == 'DMC':
                result = predict_regression_V2(select_model(model_path), data)
            elif task == 'DL':
                result = pred_and_decode_classifier(select_model(model_path), data, BHLD.DL, BHLD.DL_original)
            elif task == 'QTCN':
                result = pred_and_decode_classifier(select_model(model_path), data, BHLD.QTCN, BHLD.QTCN_original)
            elif task == 'TKDC':
                result = pred_and_decode_classifier_TKDT(select_model(model_path), data, Q.TKDC_original)
            elif task in ['CD', 'TCKT']:
                result = pred_and_decode_classifier(select_model(model_path), data, getattr(BHLD, task),
                                                    getattr(BHLD, task + '_original'))
            else:
                result = predict_regression_multi(select_model(model_path), data)
        else:
            result = jsonify({"Prediction": "Incorrect Task"})
    elif product == "VEST":
        model_map = {
            'NS': 'model/Vest/NS_V.pkl',
            'DMC': 'model/Vest/DMC_V.pkl',
            'DMV': 'model/BHLD/DMV_BH.pkl',
            'NCLD': 'model/Vest/NCLD_V.pkl',
            'DMTB': 'model/Vest/DMTB_V.pkl',
            'DL': 'model/BHLD/DL_BH.pkl',
            'QTCN': 'model/BHLD/QTCN_BH.pkl',
            'TKDC': 'model/Vest/TKDC_V.pkl',
            'CD': 'model/BHLD/CD_BH.pkl',
            'TCKT': 'model/BHLD/TCKT_BH.pkl',
            'TGGC': 'model/Vest/TGGC_V.pkl'
        }
        if task in model_map:
            model_path = model_map[task]
            if task == 'NS':
                result = predict_regression(select_model(model_path), data)
            elif task == 'DMC':
                result = predict_regression_V2(select_model(model_path), data)
            elif task == 'DL':
                result = pred_and_decode_classifier(select_model(model_path), data, BHLD.DL, BHLD.DL_original)
            elif task == 'QTCN':
                result = pred_and_decode_classifier(select_model(model_path), data, BHLD.QTCN, BHLD.QTCN_original)
            elif task == 'TKDC':
                result = pred_and_decode_classifier_TKDT(select_model(model_path), data, Vest.TKDC_original)
            elif task in ['CD', 'TCKT']:
                result = pred_and_decode_classifier(select_model(model_path), data, getattr(BHLD, task),
                                                    getattr(BHLD, task + '_original'))
            else:
                result = predict_regression_multi(select_model(model_path), data)
        else:
            result = jsonify({"Prediction": "Incorrect Task"})
    else:
        result = jsonify({"Prediction": "Incorrect Product"})
    return result

if __name__ == '__main__':
    app.run(debug=True)
