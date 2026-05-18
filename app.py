from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("columns.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get values from form
    overall_qual = float(request.form['OverallQual'])
    gr_liv_area = float(request.form['GrLivArea'])
    garage_cars = float(request.form['GarageCars'])
    total_bsmt_sf = float(request.form['TotalBsmtSF'])
    full_bath = float(request.form['FullBath'])

    # Create dictionary
    input_data = {
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'GarageCars': garage_cars,
        'TotalBsmtSF': total_bsmt_sf,
        'FullBath': full_bath
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Match training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    return render_template(
        'result.html',
        prediction=round(prediction, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)