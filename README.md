# ML-Classification-Projects

Social Network Ads Classification Using SVM
Project Overview
This project uses Support Vector Machine (SVM) classification to predict whether a user will click on an ad based on their demographic information from the Social Network Ads dataset. The goal of this project is to classify users into two categories:

1: The user clicked on the ad.
0: The user did not click on the ad.
Additionally, a user-friendly and visually appealing Streamlit frontend has been developed to allow for easy interaction with the model, where users can input features like age, estimated salary, etc., and instantly see whether they are likely to click on an ad.

Features
Support Vector Machine (SVM): The classification model used for predicting whether a user will click on an ad. SVM is a powerful machine learning algorithm for classification tasks, especially when the data has clear boundaries between classes.
Interactive Frontend: A sleek Streamlit interface where users can enter their features (age, salary) and get a real-time prediction of whether they will click on an ad or not.
Data Visualization: The app provides visual insights into the model’s decision boundary and feature importance.
Model Performance: The app also showcases model performance metrics like accuracy and confusion matrix to evaluate how well the SVM model performs on the test dataset.
Installation
1. Clone the repository:
bash
Copy
git clone https://github.com/jansaidaA2/social-network-ads-svm.git
cd social-network-ads-svm
2. Install dependencies:
Make sure you have Python 3.7 or above. Then, install the required libraries using pip:

bash
Copy
pip install -r requirements.txt
3. Run the Streamlit app:
bash
Copy
streamlit run app.py
Once the app starts, a local web page will open where you can input user details and get predictions.

How it Works
Dataset: The Social Network Ads dataset contains user demographics, including Age, Estimated Salary, and whether the user clicked on the ad (Purchased).

Support Vector Machine (SVM): The SVM model is trained to find the best boundary between two classes (ad clicked or not clicked) in a multi-dimensional feature space. The model learns from the data and predicts whether a user will click on an ad based on their features.

Frontend: The Streamlit app provides an intuitive interface where users can enter their Age and Estimated Salary to receive a prediction. The app uses the trained SVM model to output 1 for "click" and 0 for "no click".

Visualization: The frontend also includes a plot that shows the decision boundary of the SVM classifier, helping users understand how the model is making predictions.

Model Evaluation: The SVM model’s performance is evaluated using accuracy, confusion matrix, and other relevant metrics.

File Structure
app.py: Main Streamlit application to run the frontend interface.
model.py: Contains code for training the SVM classifier and making predictions.
data/: Folder containing the Social Network Ads dataset (Social_Network_Ads.csv).
requirements.txt: List of required Python packages.
README.md: This file.
Models Used
Support Vector Machine (SVM): A machine learning model used for binary classification tasks. It works by finding the hyperplane that best separates the two classes in the feature space.
Model Evaluation:
The performance of the SVM model is evaluated based on metrics such as accuracy, precision, recall, and the confusion matrix. The model is trained and tested on the dataset, and the results help assess its effectiveness in predicting ad clicks.
Usage
Once the app is running:

Enter your Age and Estimated Salary into the input fields on the frontend.
Click the Predict button to get a real-time prediction from the trained SVM model.
The model will output 1 if it predicts the user will click on the ad, and 0 if not.
The app also shows the decision boundary visualization and some basic model metrics like accuracy.
Conclusion
This project demonstrates the power of Support Vector Machines (SVM) for classification tasks, specifically predicting user behavior (ad clicks) in the context of social network ads. By building an interactive Streamlit app, users can easily interact with the model, making it accessible and useful for practical purposes.
