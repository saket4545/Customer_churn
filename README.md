# Customer Churn Prediction Using Artificial Neural Networks (ANN)

## 📌 Project Overview  
This project focuses on predicting **customer churn** using an **Artificial Neural Network (ANN)** model. The goal is to analyze customer behavior based on key financial and demographic factors to determine the likelihood of a customer leaving a business. This model helps businesses **retain customers** by identifying those at risk of churning.  

---

## 🛠️ Technologies & Tools Used  
- **Python** – Model development and deployment  
- **TensorFlow & Keras** – Building and training the ANN model  
- **Scikit-learn** – Data preprocessing and feature encoding  
- **Streamlit** – Interactive web app for user-friendly predictions  
- **Pandas & NumPy** – Data handling and manipulation  

---

## 📊 Dataset Description  
The dataset includes customer information categorized into:  

- **Demographics**: Geography, Gender, Age  
- **Financial Data**: Credit Score, Balance, Estimated Salary  
- **Account Information**: Tenure, Number of Products, Credit Card Ownership, and Active Membership  

---

## ⚙️ Workflow & Methodology  

### **1️⃣ Data Preprocessing**  
- Encoding categorical features using **Label Encoding** and **One-Hot Encoding**  
- Standardizing numerical data using **StandardScaler**  

### **2️⃣ Model Building**  
- Implemented a **deep neural network (ANN)** with multiple hidden layers  
- Used **ReLU** activation for hidden layers and **Sigmoid** activation for the output layer  
- Compiled with **binary cross-entropy loss** and **Adam optimizer**  

### **3️⃣ Model Training & Evaluation**  
- Trained the model on historical customer data  
- Evaluated using **accuracy, precision, recall, and F1-score**  

### **4️⃣ Deployment with Streamlit**  
- Created an **interactive UI** where users input customer details  
- The model predicts **churn probability** and displays whether the customer is likely to churn or stay  

---

## 🚀 Key Features in the Streamlit App  
- **User Input Fields**: Dropdowns and sliders to enter customer details  
- **Predict Button**: Generates churn probability when clicked  
- **Result Display**: Shows churn probability and customer retention status  
- **User Inputs Table**: Displays entered data in a structured format  

---

## 🎯 Business Impact  
- Helps businesses take **proactive measures** to retain high-risk customers  
- Improves **customer satisfaction and loyalty strategies**  
- Reduces **customer acquisition costs** by focusing on retention  

---

## 📂 Project Structure  
```
Customer_Churn/
│── Churn_Modelling.csv  # Dataset file
│── app.py               # Streamlit application
│── experiments.ipynb    # Model experimentation
│── prediction.ipynb     # Prediction analysis
│── model.h5             # Trained ANN model
│── scaler.pkl           # StandardScaler for data transformation
│── label_encoder_gender.pkl  # Label encoder for gender
│── onehot_encoder_geo.pkl    # One-hot encoder for geography
│── requirements.txt     # Dependencies
│── .gitignore           # Git ignored files
│── README.md            # Project documentation (this file)
```

---

## 💡 How to Run the Project  
### **Step 1: Clone the repository**  
```sh
git clone https://github.com/saket4545/Customer_churn.git
cd Customer_churn
```

### **Step 2: Install dependencies**  
```sh
pip install -r requirements.txt
```

### **Step 3: Run the Streamlit App**  
```sh
streamlit run app.py
```

---

## 🤝 Contributing  
Contributions are welcome! Feel free to submit issues or pull requests.  

---

## 📜 License  
This project is **open-source** and available under the [MIT License](LICENSE).

