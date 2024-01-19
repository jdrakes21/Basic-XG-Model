

Project Description:

This project involves the creation and evaluation of a machine learning model for predicting the probability of a shot in a soccer or similar sport resulting in a goal, known as Expected Goals (xG). The project consists of two main parts:

Data Generation (dataset.py):

-Synthetic Data: A synthetic dataset is generated with features such as shot distance and shot angle.

-Labels: The dataset includes labels indicating whether each shot resulted in a goal (1) or a miss (0).

-Class Balance: The script allows for control over the balance between class 0 (misses) and class 1 (goals) in the dataset.

Model Training and Evaluation (main.py):

-Data Loading: The synthetic dataset is loaded into memory.

-Feature and Label Separation: The features (shot_distance and shot_angle) and labels (is_goal) are separated for training and testing the machine learning model.

-Model Selection: A logistic regression model is chosen for training and evaluation.

-Model Training: The model is trained using the training data.

-Model Evaluation: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

-xG Prediction: The trained model can be used to predict the Expected Goals (xG) probability for new shots based on their distance and angle.

The project aims to create a predictive model that can estimate the likelihood of a goal given the input features, which can be useful for decision-making in sports analytics. The synthetic dataset allows for experimentation and testing of the model's performance.

Please note that the model's accuracy and effectiveness may depend on the quality and realism of the dataset and the choice of features. Adjustments to the data generation process may be necessary to achieve more realistic results.






