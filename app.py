# importing modules and Model
from flask import Flask, render_template, request
from model import Model

# intanciating web application
app = Flask(__name__)

# defining routes


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    oldpeak = request.form['oldpeak']
    ca = request.form['ca']
    thal = request.form['thal']

    features = [age, sex, cp, trestbps, chol, restecg, thalach, oldpeak, ca,
                thal]
    response = Model.predict([features])        # predicting

    return render_template('predict.html', response=response)


if __name__ == '__main__':
    app.run(debug=True)
