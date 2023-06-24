import hashlib
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import pickle
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