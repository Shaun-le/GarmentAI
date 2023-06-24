from flask import jsonify
from rules.labels.TSPS import x1_rules, x3_rules, x4_rules, x5_rules, x6_rules, x8_rules
from rules.predict_for_rules import pred_and_decode_classifier
import re
def QTCN_TSPS(model,data,label):
    output = pred_and_decode_classifier(model, data, label)

    rules = {
        'X1': x1_rules,
        'X3': x3_rules,
        'X4': x4_rules,
        'X5': x5_rules,
        'X6': x6_rules,
        'X8': x8_rules
    }

    for key, rule in rules.items():
        value = data.get(key)
        if value in rule:
            output.add(rule[value])
    sorted_output = sorted(output, key=lambda x: (int(re.findall(r'\d+', x)[0]), x))
    out = ', '.join(sorted_output)
    return jsonify({"prediction": str(out)})