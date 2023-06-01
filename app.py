import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from flask import Flask, render_template
import googleapiclient.discovery
from flask_bootstrap import Bootstrap5

app = Flask(__name__)
app.config['SECRET_KEY'] = "2018250045이중혁"
bootstrap = Bootstrap5(app)

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length, NumberRange
from sklearn.model_selection import train_test_split

STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ 12-Dec)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Predict')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = [[
            float(form.longitude.data),
            float(form.latitude.data),
            str(form.month.data),
            str(form.day.data),
            float(form.avg_temp.data),
            float(form.max_temp.data),
            float(form.max_wind_speed.data),
            float(form.avg_wind.data)
        ]]

        #in order to make a prediction, we must scale the data using the same scale as the one used to mkae model
        X_test = pd.DataFrame(X_test, columns = ['longitude', 'latitude', 'month', 'day', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'])

        num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
        cat_attribs = ['month', 'day']

        full_pipeline = ColumnTransformer([
            ("num", StandardScaler(), num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        X = pd.read_csv('./datasets/sanbul2district-divby100.csv', sep=',')
        full_pipeline.fit(X)
        X_prepared = full_pipeline.transform(X_test)

        #create the resource to the model web api on GCP
        project_id = "loc8r-363404"
        model_id = "fire_prediction"
        model_path = "projects/{}/models/{}".format(project_id, model_id)
        model_path += "/versions/f0001/"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "loc8r-363404-73b2d0e15e2c.json"
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

        input_data_json = {"signature_name": "serving_default", "instances": X_prepared.tolist()}
        print(X_prepared.tolist())
        # input_data_json = {"signature_name": "serving_default", "instances": [[
        #     -0.2744229, 0.47931717, 1.05731055, 0.94222231, -1.92883647, -1.21548112,
        #     0, 0, 0, 0, 0, 0,
        #     0, 0, 1, 0, 0, 0,
        #     0, 1, 0, 0, 0, 0,
        #     0, 0.
        # ]]}

        # make a prediction
        request = ml_resource.predict(name=model_path, body=input_data_json)
        response = request.execute()

        if "error" in response:
            raise RuntimeError(response["error"])

        # extract the prediction from the response
        # print(response["predictions"])
        pred = np.array([pred['dense_13'] for pred in response["predictions"]])

        res = pred[0][0]*100
        res = round(res, 1)

        return render_template('result.html', res=res)
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)