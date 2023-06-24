from flask import jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from prepare_data import hash_encode, extract_strings_and_numbers_from_dict
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
    pred = model.predict(data_transformed.values.reshape(1, -1))
    r = jsonify({"prediction": pred[0][0]})
    return r

def predict_regression_V3(model,data):
    data_transformed = extract_strings_and_numbers_from_dict(data)
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