import joblib


class Model:
    def __init__(self):
        self.model = joblib.load('model/catboost_model.pkl')

    def predict(self, inputs):
        import numpy as np
        input_features = np.array([inputs], dtype='object')
        print(input_features)
        return self.model.predict(input_features)[0]
