from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np


app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define route to render the HTML form
@app.route('/')
def man():
    return render_template('index.html')

# Define route to handle prediction request
@app.route('/predict', methods=['POST'])
def home():
    # Get user input from the form
    tenure = int(request.form['tenure'])
    monthly_charges = float(request.form['MonthlyCharges'])
    total_charges = float(request.form['TotalCharges'])

    # Binary features
    senior_citizen = 1 if request.form.get('SeniorCitizen') == '1' else 0
    partner = 1 if request.form.get('Partner') == '1' else 0
    dependents = 1 if request.form.get('Dependents') == '1' else 0
    phone_service = 1 if request.form.get('PhoneService') == '1' else 0
    paperless_billing = 1 if request.form.get('PaperlessBilling') == '1' else 0
    multiple_lines_no = 1 if request.form.get('MultipleLines') == '0' else 1
    multiple_lines_yes = 1 if request.form.get('MultipleLines') == '1' else 0
    internet_service_dsl = 1 if request.form.get('InternetService') == 'DSL' else 0
    internet_service_fiber_optic = 1 if request.form.get('InternetService') == 'Fiber optic' else 0
    internet_service_no = 1 if request.form.get('InternetService') == 'No' else 0
    online_security_no = 1 if request.form.get('OnlineSecurity') == '0' else 0
    online_security_yes = 1 if request.form.get('OnlineSecurity') == '1' else 0
    online_backup_no = 1 if request.form.get('OnlineBackup') == '0' else 0
    online_backup_yes = 1 if request.form.get('OnlineBackup') == '1' else 0
    device_protection_no = 1 if request.form.get('DeviceProtection') == '0' else 0
    device_protection_yes = 1 if request.form.get('DeviceProtection') == '1' else 0
    tech_support_no = 1 if request.form.get('TechSupport') == '0' else 0
    tech_support_yes = 1 if request.form.get('TechSupport') == '1' else 0
    streaming_tv_no = 1 if request.form.get('StreamingTV') == '0' else 0
    streaming_tv_yes = 1 if request.form.get('StreamingTV') == '1' else 0
    streaming_movies_no = 1 if request.form.get('StreamingMovies') == '0' else 0
    streaming_movies_yes = 1 if request.form.get('StreamingMovies') == '1' else 0
    contract_month_to_month = 1 if request.form.get('Contract') == 'Month-to-month' else 0
    contract_one_year = 1 if request.form.get('Contract') == 'One year' else 0
    contract_two_year = 1 if request.form.get('Contract') == 'Two year' else 0
    payment_method_bank_transfer = 1 if request.form.get('PaymentMethod') == 'Bank transfer (automatic)' else 0
    payment_method_credit_card = 1 if request.form.get('PaymentMethod') == 'Credit card (automatic)' else 0
    payment_method_electronic_check = 1 if request.form.get('PaymentMethod') == 'Electronic check' else 0
    payment_method_mailed_check = 1 if request.form.get('PaymentMethod') == 'Mailed check(automatic)' else 0

    # Prepare input data for prediction
    input_data = np.array([[tenure, monthly_charges, total_charges, senior_citizen, partner, dependents,
                            phone_service, paperless_billing, multiple_lines_no, multiple_lines_yes,internet_service_dsl,
                            internet_service_fiber_optic, internet_service_no,online_security_no, online_security_yes, 
                            online_backup_no, online_backup_yes,device_protection_no, device_protection_yes, 
                            tech_support_no, tech_support_yes,streaming_tv_no, streaming_tv_yes, 
                            streaming_movies_no, streaming_movies_yes,contract_month_to_month, contract_one_year, contract_two_year,
                            payment_method_bank_transfer, payment_method_credit_card,
                            payment_method_electronic_check, payment_method_mailed_check]])

    # Perform prediction
    prediction = model.predict(input_data)
    return redirect(url_for('show_results', prediction=prediction))
   

if __name__ == '__main__':
    app.run(debug=True)
