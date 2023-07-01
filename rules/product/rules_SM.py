from flask import jsonify
from rules.labels.SM import x2_rules, x3_rules, x7_rules, x5_rules, x6_rules, x9_rules, x10_rules
from rules.predict_for_rules import pred_and_decode_classifier
import re
def QTCN_SM(model, data, label):
    output = pred_and_decode_classifier(model, data, label)

    rules = {
        'X2': x2_rules,
        'X3': x3_rules,
        'X5': x5_rules,
        'X6': x6_rules,
        'X7': x7_rules,
        'X9': x9_rules,
        'X10': x10_rules
    }

    for key, rule in rules.items():
        value = data.get(key)
        if key == 'X2' and value in rule:
            output = {x for x in output if not x.startswith('D13')}
            output.add(rule[value])
        elif key == 'X6' and value == 'Không túi':
            output = {x for x in output if not x.startswith('D4')}
        elif key == 'X6' and value in rule:
            output = {x for x in output if not x.startswith('D4')}
            output.add(rule[value])
        elif key == 'X5' and value in rule:
            output = {x for x in output if not x.startswith('D2')}
            output.add(rule[value])
        elif key == 'X7' and value in rule:
            output = {x for x in output if not x.startswith('D3')}
            output.add(rule[value])
        elif key == 'X10' and value in rule:
            output = {x for x in output if not x.startswith('D6')}
            output.add(rule[value])
        elif value in rule:
            output.add(rule[value])
    sorted_output = sorted(output, key=lambda x: (int(re.findall(r'\d+', x)[0]), x))
    out = ', '.join(sorted_output)
    return jsonify({"prediction": str(out)})