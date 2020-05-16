from flask import Flask,request,jsonify,render_template
from flasgger import Swagger
import pandas as pd
import pickle

with open("F:\Docker\Dockers\classifier.pkl",'rb') as f:
    model = pickle.load(f)
    
print('Model Loaded.....!')


app = Flask(__name__)
swagger  = Swagger(app)


@app.route("/")
def welcome():
    return render_template("home.html")

@app.route('/predict_file',methods=['POST'])
def predict_file():
    """Example file endpoint returning a prediction of BanknoteAuth
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    """
    # file_data = pd.read_csv(request.files.get('file'))
    file_data = pd.read_csv(request.files['file'])
    predictions = model.predict(file_data)
    return render_template('home.html', prediction_text='Result of File: {}'.format(str(predictions)))

if __name__=="__main__":
    app.run(port=5000)