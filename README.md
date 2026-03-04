# Titanic-survival-prediction

🚢 Titanic Survival Prediction

 Project Overview

This project is a Machine Learning classification system that predicts whether a passenger survived the Titanic disaster based on demographic and ticket-related information. The model is trained using the Kaggle dataset from the Titanic - Machine Learning from Disaster, which contains historical passenger records from the RMS Titanic.

The goal of this project is to demonstrate an end-to-end Machine Learning workflow — from data preprocessing and model training to GUI-based deployment.

<h2> Problem Statement</h2>

Predict passenger survival using features such as:

Passenger Class (Pclass)

Gender (Sex)

Age

Number of Siblings/Spouses (SibSp)

Number of Parents/Children (Parch)

Fare

Port of Embarkation (Embarked)

Target Variable:

Survived (0 = No, 1 = Yes)

<h2> Machine Learning Approach</h2>  
 Data Preprocessing
<ul>
  <li>Handling missing values using SimpleImputer</li>
  
   <li>Feature scaling using StandardScaler</li>
   <li>Categorical encoding using OneHotEncoder</li>
   <li>Implemented using Pipeline & ColumnTransformer</li>
</ul>


 Model Used
<ul>
  <li> Random Forest Classifier </li>
    <li>  300 Decision Trees
</li>
    <li>  80:20 Train-Test Split</li>
    <li> Performance evaluated using Accuracy Score </li>

</ul>

<h2>📊 Features of the Application</h2>  

✔️ Real-time Survival Prediction
✔️ Probability Score Display
✔️ Model Accuracy Display
✔️ Survival Count Table
✔️ Gender-wise Survival per Class
✔️ Graph Visualization
✔️ Modern Styled PyQt5 GUI

<h2>🖥️ Application Workflow</h2>  
Dataset → Data Preprocessing → Model Training
     </br>   ↓</br>
   Model Evaluation (Accuracy)
      </br>  ↓</br>
  Integrated into PyQt5 GUI
       </br> ↓</br>
  User Input → Prediction → Result Display
<h2>🛠️ Technologies Used</h2>  

Python

Pandas

Scikit-learn

Matplotlib

PyQt5

Random Forest Algorithm

<h2>📈 What This Project Demonstrates</h2>  

Supervised Learning (Binary Classification)

Data Cleaning & Feature Engineering

Machine Learning Pipeline Implementation

Model Evaluation Techniques

GUI-based ML Model Deployment

Real-world Dataset Application

<h2> Conclusion</h2>  

This project provides a complete implementation of a Machine Learning system integrated with a graphical user interface. It showcases practical application of AI/ML concepts using a real-world dataset and demonstrates how predictive models can be deployed in user-friendly applications.
