# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import os

# # Ensure TensorFlow does not use any unavailable optimizations
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# def train_ml_model(file_path):
#     try:
#         data = pd.read_csv(file_path)
#         X = data.iloc[:, :-1]
#         y = data.iloc[:, -1]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#         clf = DecisionTreeClassifier()
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
        
#         model_path = os.path.join('static', 'trained_models', 'trained_ml_model.csv')
#         os.makedirs(os.path.dirname(model_path), exist_ok=True)
#         pd.DataFrame(clf.predict(X)).to_csv(model_path, index=False)
        
#         return model_path
#     except Exception as e:
#         print(f"Error training ML model: {e}")
#         raise

# def train_dl_model(file_path):
#     try:
#         data = pd.read_csv(file_path)
#         X = data.iloc[:, :-1]
#         y = data.iloc[:, -1]

#         y = pd.get_dummies(y)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#         model = Sequential()
#         model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
#         model.add(Dense(32, activation='relu'))
#         model.add(Dense(y_train.shape[1], activation='softmax'))

#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
        
#         model_path = os.path.join('static', 'trained_models', 'trained_dl_model.csv')
#         os.makedirs(os.path.dirname(model_path), exist_ok=True)
#         pd.DataFrame(model.predict(X)).to_csv(model_path, index=False)
        
#         return model_path
#     except Exception as e:
#         print(f"Error training DL model: {e}")
#         raise
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import re
import os
import joblib

# Function to clean feature names
def clean_feature_names(df):
    df.columns = [re.sub(r'[^\w\s]', '_', col) for col in df.columns]
    return df

# Function to encode categorical features
def auto_encoding(df):
    cat_features = [x for x in df.columns if df[x].dtype == "object"]
    le = LabelEncoder()
    for col in cat_features:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# Function to train ML model
def train_ml_model(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Clean the feature names
    data = clean_feature_names(data)
    
    # Preprocess the data
    data = auto_encoding(data)
    
    # Handle missing values if any
    data = data.fillna(0)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model using joblib
    model_path = 'trained_ml_model.model'
    joblib.dump(clf, model_path)
    
    # Create a DataFrame with the test set and predictions
    test_data_with_predictions = X_test.copy()
    test_data_with_predictions['Actual'] = y_test
    test_data_with_predictions['Predicted'] = y_pred
    
    # Concatenate the test set with predictions back with the original dataset for saving
    data_with_predictions = pd.concat([data, test_data_with_predictions], axis=0)
    
    # Save the entire dataset with predictions to CSV
    data_with_predictions.to_csv('data_with_predictions.csv', index=False)
    
    return model_path

# Example usage
# model_path = train_ml_model('your_dataset.csv')