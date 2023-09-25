import pickle
import pandas as pd
from flask import Flask, render_template_string, request
import zipfile
import os

app = Flask(__name__)

with open('xgboost_model.pkl', 'rb') as model_file:
    loaded_xgb_model = pickle.load(model_file)

zip_file_path = 'label_encoders (3).zip'

with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
    with zip_file.open('label_encoders (3).pkl', 'r') as encoder_file:
        label_encoders = pickle.load(encoder_file)

csv_data = pd.read_csv('unique_values.csv')

feature_values = {}
for feature in csv_data.columns:

    unique_values = csv_data[feature].unique().tolist()

    dropdown_options = [{'label': value, 'value': value} for value in unique_values]

    feature_values[feature] = dropdown_options

templates_zip_path = 'templates1.zip'

with zipfile.ZipFile(templates_zip_path, 'r') as templates_zip:
    with templates_zip.open('templates/index.html', 'r') as template_file:
        template_content = template_file.read().decode('utf-8')


feature_names_used_by_model = csv_data.columns.tolist()

@app.route('/', methods=['GET', 'POST'])
def predict_price():
    if request.method == 'POST':
        input_features = {}
        for feature in csv_data.columns:
            input_value = request.form.get(feature)

            try:
                input_value = float(input_value)
            except ValueError:
                input_value = 0
            input_features[feature] = input_value

        input_data = pd.DataFrame([input_features])

        for column in label_encoders:
            if column in input_features:
                if input_features[column] not in label_encoders[column].classes_:
                    input_data[column] = 0
                else:
                    input_data[column] = label_encoders[column].transform([input_features[column]])
            else:
                
                input_data[column] = 0  

  
        input_data = input_data[feature_names_used_by_model]

        predicted_price = loaded_xgb_model.predict(input_data)

        return f'Predicted Car Price: £{predicted_price[0]:,.2f}'

    return render_template_string(template_content, feature_names=csv_data.columns, feature_values=feature_values)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
