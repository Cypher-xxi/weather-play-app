from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv('weather.csv')
le_dict = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

X = df.drop('Play', axis=1)
y = df['Play']

model = DecisionTreeClassifier()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Outlook': request.form['outlook'],
        'Temperature': request.form['temperature'],
        'Humidity': request.form['humidity'],
        'Windy': request.form['windy']
    }

    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        input_df[col] = le_dict[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    result = le_dict['Play'].inverse_transform([prediction])[0]

    return render_template('index.html', prediction=result,
                           input_data=data_in)

if __name__ == '__main__':
    app.run(debug=True)