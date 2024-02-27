# Heart-Disease-detection
This project involves building a machine learning model to predict the likelihood of heart disease based on various factors such as age, sex, blood pressure, cholesterol levels, and other medical indicators. Here's a step-by-step description of what has been done in this project:

1. **Data Loading and Exploration:**
   - Loaded the dataset using pandas `read_csv` function.
   - Explored the structure of the dataset using `info()` method and checked for missing values.
   - Described the numerical features using `describe()` method.
   - Plotted distribution of categorical features using seaborn's `countplot` function.

2. **Handling Class Imbalance:**
   - Noticed class imbalance in the target variable (HeartDisease).
   - Used the `RandomOverSampler` from `imblearn` to balance the classes.

3. **Model Selection and Hyperparameter Tuning:**
   - Defined a function `find_best_model_using_gridsearchcv` to find the best machine learning model using grid search with cross-validation.
   - Evaluated several classifiers including Decision Tree, Random Forest, Gradient Boosting, SVM, and KNN.
   - Chose Random Forest as the final model based on the highest accuracy score.

4. **Training the Model:**
   - Initialized a Random Forest classifier with the best parameters obtained from hyperparameter tuning.
   - Trained the model on the resampled data (balanced dataset).

5. **Model Evaluation:**
   - Split the data into training and testing sets.
   - Evaluated the model's performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
   - Visualized the confusion matrix to understand the model's performance on different classes.

6. **Model Deployment:**
   - Created a function `preprocess_input` to preprocess input features for prediction.
   - Created a function `predict_heart_disease` to take user input for heart disease prediction using the trained model.
   - Demonstrated the prediction by taking user input and displaying the predicted outcome.

7. **Model Serialization:**
   - Serialized the trained model using `joblib` and saved it as a file (`model.pkl`) for future use.

Overall, this project involved data preprocessing, model selection, training, evaluation, deployment, and serialization to build a predictive model for heart disease detection.
