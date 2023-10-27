from flask import jsonify
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold, train_test_split
import pickle
from sklearn.metrics import f1_score
import pandas as pd
import re
from prepare_data import fh

def read_data(data):
    df = pd.DataFrame(eval(data)).transpose()
    df.columns = ['X' + str(i) for i in range(1, len(df.columns) + 1)]
    df.rename(columns={df.columns[-1]: 'Y'}, inplace=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
            df.iloc[:,:-1] = fh.transform(df.iloc[:,:-1])
    return df

def trans_output(score, y, is_regression = True):
    if is_regression:
        return jsonify({"prediction": f"Hoàn thành: Loss = {score:.2f} (Tỷ lệ sai số: {score / y.mean() * 100:.2f}%)"})
    return jsonify({"prediction": f"Hoàn thành: Độ chính xác = {score*100:.2f}%"})

def create_dummy_variables(df, column_name):
    values = set()
    for value in df[column_name]:
        values.update(value.split('; '))

    for value in values:
        pattern = re.compile(r'\b{}\b'.format(re.escape(value)))
        df[value] = df[column_name].apply(lambda x: 1 if re.search(pattern, x) else 0)

    df.drop(column_name, axis=1, inplace=True)
    return df

def train_with_Kfolds(X, y, path, is_regression=True, is_multi=True):
    X_train, X_test_g, y_train, y_test_g = train_test_split(X, y, test_size=0.1)

    if is_regression:
        if is_multi:
            model = MultiOutputRegressor(AdaBoostRegressor())
        else:
            model = AdaBoostRegressor()
    else:
        if is_multi:
            model = MultiOutputClassifier(AdaBoostClassifier())
        else:
            model = AdaBoostClassifier()

    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)

    if is_regression:
        y_pred = model.predict(X_test_g)
        score = mean_absolute_error(y_test_g, y_pred)
        if (score / y_test_g.mean() * 100) < 2:
            with open(path, 'wb') as file:
                pickle.dump(model, file)
            return score
    else:
        y_pred = model.predict(X_test_g)
        score = accuracy_score(y_pred.flatten(),y_test_g.flatten())
        #if score > 0.7:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        return score

    return train_with_Kfolds(X, y, path, is_regression, is_multi)

'''def train_with_Kfolds(X, y, path):
    X_train, X_test_g, y_train, y_test_g = train_test_split(X, y, test_size=0.1)

    # Define the model and its hyperparameter grid
    model = AdaBoostRegressor()
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.5],
    }

    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

    # Perform grid search with k-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate the model on the hold-out test set
    score = eval_regression(best_model, X_test_g, y_test_g)

    if (score / y.mean() * 100) < 3:
        with open(path, 'wb') as file:
            pickle.dump(best_model, file)
        return f"{score / y.mean() * 100:.2f}"
    else:
        # If the performance is not satisfactory, try again with K-fold cross-validation
        return train_with_Kfolds(X, y, path)'''