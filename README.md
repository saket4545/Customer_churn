#Customer Churn Prediction Using Artificial Neural Networks (ANN)
📌 Project Overview
This project focuses on predicting customer churn using an Artificial Neural Network (ANN) model. The goal is to analyze customer behavior based on key financial and demographic factors and determine the likelihood of a customer leaving a business. The model helps businesses retain customers by identifying those at risk of churning.

🛠️ Technologies & Tools Used
Python (for model development and deployment)
TensorFlow & Keras (for building and training the ANN model)
Scikit-learn (for preprocessing and feature encoding)
Streamlit (for building an interactive web app)
Pandas & NumPy (for data handling and manipulation)
📊 Dataset Description
The dataset includes customer information such as:

Demographics: Geography, Gender, Age
Financial Data: Credit Score, Balance, Estimated Salary
Account Information: Tenure, Number of Products, Credit Card Ownership, and Active Membership
⚙️ Workflow & Methodology
Data Preprocessing:
Encoding categorical features using Label Encoding and One-Hot Encoding
Standardizing numerical data using StandardScaler
Model Building:
Implemented a deep neural network (ANN) with multiple hidden layers
Used ReLU activation for hidden layers and Sigmoid activation for the output layer
Compiled with binary cross-entropy loss and Adam optimizer
Model Training & Evaluation:
Trained the model on historical customer data
Evaluated using accuracy, precision, recall, and F1-score
Deployment with Streamlit:
Created an interactive UI where users input customer details
The model predicts churn probability and displays whether the customer is likely to churn or stay
🚀 Key Features in the Streamlit App
User Input Fields: Dropdowns and sliders to enter customer details
Predict Button: Generates churn probability when clicked
Result Display: Shows churn probability and customer retention status
User Inputs Table: Displays entered data in a structured format
🎯 Business Impact
Helps businesses take proactive measures to retain high-risk customers
Improves customer satisfaction and loyalty strategies
Reduces customer acquisition costs by focusing on retention
