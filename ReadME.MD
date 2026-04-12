# End-to-End Machine Learning Pipeline with FastAPI Deployment

##  Overview

This project demonstrates a complete **end-to-end Machine Learning pipeline** for predicting laptop prices using structured data. It covers the entire workflow from data preprocessing to model selection and API deployment.

The system is designed using **Object-Oriented Programming (OOP)** principles and integrates multiple machine learning models with automated hyperparameter tuning.

---

##  Key Features

*  Data Loading and Preprocessing using Pandas
*  Automatic handling of missing values (Imputation)
*  Feature scaling and encoding using Scikit-learn Pipelines
*  Multi-model selection using GridSearchCV:

  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * Support Vector Regressor (SVR)
  * Ridge & Lasso Regression
*  Model evaluation using:

  * R² Score
  * Mean Absolute Error (MAE)
  * Mean Squared Error (MSE)
*  Model persistence using Joblib (`.pkl`)
*  REST API built using FastAPI for real-time predictions

---

##  Project Structure

```
 ML Project
 ┣  ml.py            
 ┣  df.csv            
 ┣  Model.pkl          
 ┗  README.md          
```

---

##  Technologies Used

* Python 
* Pandas & NumPy
* Scikit-learn
* FastAPI
* Joblib
* Uvicorn

---

##  Workflow

1. Load dataset from CSV
2. Perform data analysis and preprocessing
3. Split data into training and testing sets
4. Apply feature transformation using pipelines
5. Train multiple models using GridSearchCV
6. Select the best model automatically
7. Save the trained model using Joblib
8. Deploy the model using FastAPI

---

##  How to Run the Project

###  Install Dependencies

```
pip install pandas scikit-learn fastapi uvicorn joblib
```

###  Run the Application

```
python ml.py
```

OR directly using Uvicorn:

```
uvicorn ml:app --reload
```

---

##  API Endpoints

###  Home

```
GET /
```

###  Prediction Endpoint

```
POST /predict
```

###  Sample Input (JSON)

```json
{
  "Company": "Dell",
  "TypeName": "Notebook",
  "Ram": 8,
  "Weight": 1.5,
  "Touchscreen": 0,
  "Ips": 1,
  "ppi": 141,
  "Cpu_brand": "Intel Core i5",
  "HDD": 0,
  "SSD": 256,
  "Gpu_brand": "Intel",
  "os": "Windows"
}
```

###  Sample Output

```json
{
  "predicted_price": 55000
}
```

---

##  Model Performance

The model is evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

Best model is automatically selected using GridSearchCV.

---

##  Key Highlights

* Built a reusable and scalable ML pipeline using OOP
* Implemented automated model selection (AutoML approach)
* Integrated preprocessing + model into a single pipeline
* Deployed as a REST API for real-time usage

---

## Future Improvements

* Add frontend using Streamlit
* Deploy to cloud (AWS / Render / Azure)
* Add logging and monitoring
* Integrate with real-time data sources

---

## Author

Murali

---

## Conclusion

This project showcases the ability to build production-ready Machine Learning systems with proper architecture, scalability, and deployment.

---
