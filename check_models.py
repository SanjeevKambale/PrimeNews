import joblib

# Load the models and vectorizer
vectorizer = joblib.load(r'C:\MyProject\PrimeNews\Models\vectorizer.joblib')
model_lr = joblib.load(r'C:\MyProject\PrimeNews\Models\lr_model.joblib')
model_dtc = joblib.load(r'C:\MyProject\PrimeNews\Models\dtc_model.joblib')

# Check vectorizer vocabulary size and sample features
print("Vectorizer vocabulary size:", len(vectorizer.vocabulary_))
print("Sample feature names:", list(vectorizer.vocabulary_.keys())[:10])

# Check model types
print("Logistic Regression model type:", type(model_lr))
print("Decision Tree model type:", type(model_dtc))

# Print model parameters
print("\nLogistic Regression parameters:")
print(model_lr.get_params())

print("\nDecision Tree parameters:")
print(model_dtc.get_params())

# Optionally, print model scores if you want to check them on the training data
# Uncomment below if you want to see training accuracy
# x = vectorizer.transform(["example news text for testing"])
# print("LR prediction for sample:", model_lr.predict(x))
# print("DTC prediction for sample:", model_dtc.predict(x))
