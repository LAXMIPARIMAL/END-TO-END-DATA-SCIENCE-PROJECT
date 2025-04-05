# END-TO-END-DATA-SCIENCE-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: GURRALA LAXMI PARIMAL

INTERN ID: CT12RER

DURATION: 8 WEEKS

MENTOR: NEELA SANTHU

##Description:This Python script implements a complete end-to-end machine learning pipeline to predict customer churn using a Random Forest Classifier, followed by deployment using a FastAPI web service. The primary objective is to develop a predictive model that can determine whether a customer is likely to churn (i.e., discontinue service) based on historical data. The script begins by loading a customer churn dataset using the pandas library, which provides powerful data manipulation tools. The dataset is assumed to be stored locally in a CSV file and includes various features related to customer behavior and service usage. The target variable, "Churn", is encoded into numerical format using LabelEncoder from scikit-learn, transforming categorical churn labels ("Yes"/"No") into binary format (1/0), making it suitable for machine learning models.
The features (X) are selected by dropping irrelevant columns such as 'Churn' and 'customerID', and categorical variables are converted into numerical form using one-hot encoding via pd.get_dummies(). The dataset is then scaled using StandardScaler, which standardizes feature values to ensure equal contribution to the model, especially important when models are sensitive to data distribution. The dataset is split into training and testing sets using an 80-20 ratio through train_test_split, maintaining randomness with a fixed seed (random_state=42) to ensure reproducibility. The training data is then used to train a RandomForestClassifier, an ensemble learning method that builds multiple decision trees and aggregates their results to improve prediction accuracy and reduce overfitting.
Once the model is trained, it is evaluated on the test set using accuracy_score, which compares predicted labels with true labels. The script prints the model's accuracy, indicating how well it performs on unseen data. The trained model, along with the label encoder, scaler, and feature names, is serialized and saved using the pickle module to allow for reuse without retraining. This forms the basis of model persistence—an essential step in real-world deployment.
Following model training and saving, the script sets up a RESTful API using FastAPI, a high-performance web framework ideal for deploying machine learning models in production environments. The BaseModel class from Pydantic is used to define the expected structure of input data, ensuring validation before making predictions. When a POST request is made to the /predict endpoint with a list of features, the API checks if the number of features matches the expected input size. The features are then scaled using the previously saved scaler and passed to the trained model for prediction. The output is a JSON object containing the churn prediction (True or False) and the model’s accuracy. Error handling is also implemented using FastAPI’s HTTPException, which ensures that any mismatch or processing error is communicated clearly to the user.
This script is designed to run on any machine supporting Python, such as local development environments or cloud platforms like AWS EC2, Google Cloud, or Heroku. Uvicorn, an ASGI server, is used to launch the FastAPI app and expose it over HTTP. The combination of scikit-learn, FastAPI, and model serialization allows this solution to function as a real-time prediction service, exemplifying how machine learning models can be operationalized for business decision-making in a scalable and maintainable manner.
##OUTPUT
[customer_churn_prediction_dataset.csv](https://github.com/user-attachments/files/19614481/customer_churn_prediction_dataset.csv)
