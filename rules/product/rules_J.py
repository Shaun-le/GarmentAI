from flask import jsonify
from rules.labels.J import x2_rules, x3_rules, x5x6_rules, x7_rules, x8_rules, x10_rules, x13_rules, x14_rules
from rules.predict_for_rules import pred_and_decode_classifier
import re
def QTCN_J(model, data, label):
    output = pred_and_decode_classifier(model, data, label)
    rules = {
        'X2': x2_rules,
        'X3': x3_rules,
        'X5': x5x6_rules,
        'X6': x5x6_rules,
        'X7': x7_rules,
        'X8': x8_rules,
        'X13': x13_rules,
        'X14': x14_rules,
    }
    output = {x for x in output if not x.startswith('F5')}
    for key, rule in rules.items():
        value = data.get(key)
        if key == 'X5' and value in rule:
            output.add(rule[value])
        elif key == 'X6' and value in rule:
            output.add(rule[value])
        elif key == 'X2' and value in rule:
            output = {x for x in output if not x.startswith('F2')}
            output.add(rule[value])
        elif key == 'X3' and value in rule:
            output = {x for x in output if not x.startswith('F13.6')}
            output.add(rule[value])
        elif key == 'X7' and value in rule:
            output = {x for x in output if not x.startswith('F7')}
            output.add(rule[value])
        elif key == 'X8' and value in rule:
            output = {x for x in output if not x.startswith('F8')}
            output.add(rule[value])
        elif key == 'X13' and value in rule:
            output = {x for x in output if not x.startswith('F9')}
            output.add(rule[value])
        elif key == 'X14' and value in rule:
            output = {x for x in output if not x.startswith('F4')}
            output.add(rule[value])
        elif value in rule:
            output.add(rule[value])

    sorted_output = sorted(output, key=lambda x: (int(re.findall(r'\d+', x)[0]), x))
    out = ', '.join(sorted_output)
    return jsonify({"prediction": str(out)})