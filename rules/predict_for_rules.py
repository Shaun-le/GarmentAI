from prepare_data import hash_encode

def pred_and_decode_classifier(model,data,label):
    data_transformed = hash_encode(data)
    pred = model.predict([data_transformed])[0]
    result = set()
    for i, p in enumerate(pred):
        if p == 1:
            result.add(label[i])

    return result