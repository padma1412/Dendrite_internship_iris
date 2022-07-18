from cmath import log
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(
    open('xgbModel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sepal_length = float(request.form.get('sepal_length'))
    sepal_width =  float(request.form.get('sepal_width'))
    petal_length = float(request.form.get('petal_length'))
    species = request.form.get('species')
    prediction = model.predict([[sepal_length,sepal_width,petal_length]])

    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #log(final_features)
    #prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    print(output)

    return render_template('index.html', prediction_text='The petal width predicted is {}'.format(output))


if __name__ == "__main__":
    app.run()
