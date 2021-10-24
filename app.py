
# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'diabetes.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/index')
def index():
	return render_template('index.html')
@app.route('/dia')
def dia():
	return render_template('dia.html')
@app.route('/ourT')
def ourT():
	return render_template('ourT.html')
@app.route('/contact')
def contact():
	return render_template('contact.html')

@app.route('/')
def home():
	return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=False,host='0.0.0.0')    