import joblib
file = 'logistic_regression_model_1.pkl'
# Load the pickle file
model_data = joblib.load(file)

# View the contents of the model
print(model_data)
model = model_data['model']
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
