# Import libraries
from flask import Flask, request, jsonify
from re_train.utils import train_with_Kfolds, create_dummy_variables, read_data, trans_output
import shutil

app = Flask(__name__)
@app.route('/rtr', methods = ['POST'])
def main():
    data = request.get_json(force=True)
    product = data.get('Product')
    task = data.get('Task')
    df = read_data(data.get('Data'))
    result = {}
    if product == 'TSPS':
        if task == 'NS':
            X = df.drop(columns=['Y']).values
            y = df['Y'].values
            loss = train_with_Kfolds(X, y, 'D:/May/GarmentAI/models/NS_TSPS.pkl', True, False)
            result = trans_output(loss, y, True)
        elif task == 'DMC':
            X = df.drop(columns=['Y1', 'Y2', 'Y3', 'Y4']).values
            y = df['Y1'].values
            loss = train_with_Kfolds(X, y, 'D:/May/GarmentAI/models/DMC_TSPS.pkl', True, False)
            result = trans_output(loss, y, True)
        elif task == 'QTCN':
            df = create_dummy_variables(df, 'Y')
            X = df.iloc[:,:9].values
            y = df.iloc[:,9:].values
            score = train_with_Kfolds(X, y, 'D:/May/GarmentAI/models/QTCN_TSPS.pkl', False, True)
            print(score)
            result = trans_output(score, y, False)
    elif product == "SM":
        if task == "QTCN":
            print(df)
    elif product == 'J':
        if task == 'QTCN':
            df = create_dummy_variables(df, 'Y')
            X = df.iloc[:,:15].values
            y = df.iloc[:,15:].values
            score = train_with_Kfolds(X, y, 'D:/May/GarmentAI/models/QTCN_J.pkl', False, True)
            result = trans_output(score, y, False)
    return result

@app.route('/dl', methods = ['POST'])
def deploy():
    data = request.get_json(force=True)
    product = data.get('Product')
    task = data.get('Task')
    if product == "J":
        if task == "QTCN":
            shutil.move('models/QTCN_J.pkl', 'D:/May/GarmentAI/model')
    return jsonify({"prediction": f"Mô Hình Đã Cập Nhật"})

if __name__ == '__main__':
    app.run(debug=True)