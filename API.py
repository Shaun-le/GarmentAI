# Import libraries
from flask import Flask, request, jsonify
from prepare_data import select_model
from predict import predict_regression, predict_regression_multi, predict_regression_V2, predict_regression_V3, pred_and_decode_classifier, pred_and_decode_classifier_TKDT
from labels import TSPS, SM, Q, J, BHLD, Vest
from rules.product.rules_J import QTCN_J
from rules.product.rules_Q import QTCN_Q
from rules.product.rules_SM import QTCN_SM
from rules.product.rules_TSPS import QTCN_TSPS

app = Flask(__name__)

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
                result = QTCN_TSPS(select_model(model_path), data, TSPS.QTCN)
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
                result = QTCN_SM(select_model(model_path), data, SM.QTCN)
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
                result = QTCN_Q(select_model(model_path), data, Q.QTCN)
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
                result = QTCN_J(select_model(model_path), data, J.QTCN)
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
            'TKDC': 'model/BHLD/TKDC_BH.pkl',
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
                result = pred_and_decode_classifier_TKDT(select_model(model_path), data, BHLD.TKDC_original)
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
            'DMVC': 'model/Vest/DMVC_V.pkl',
            'DMVL': 'model/Vest/DMVL_V.pkl',
            'NCLD': 'model/Vest/NCLD_V.pkl',
            'DMTB': 'model/Vest/DMTB_V.pkl',
            'DL': 'model/Vest/DL_V.pkl',
            'QTCN': 'model/Vest/QTCN_V.pkl',
            'TKDC': 'model/Vest/TKDC_V.pkl',
            'CD': 'model/Vest/CD_V.pkl',
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
                result = pred_and_decode_classifier(select_model(model_path), data, Vest.DL, Vest.DL_original)
            elif task == 'QTCN':
                result = pred_and_decode_classifier(select_model(model_path), data, Vest.QTCN, Vest.QTCN_original)
            elif task == 'TKDC':
                result = pred_and_decode_classifier_TKDT(select_model(model_path), data, Vest.TKDC_original)
            elif task in ['CD', 'TCKT']:
                result = pred_and_decode_classifier(select_model(model_path), data, getattr(Vest, task),
                                                    getattr(Vest, task + '_original'))
            else:
                result = predict_regression_multi(select_model(model_path), data)
        else:
            result = jsonify({"Prediction": "Incorrect Task"})
    else:
        result = jsonify({"Prediction": "Incorrect Product"})
    return result

if __name__ == '__main__':
    app.run(debug=True)
