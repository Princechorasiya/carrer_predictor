import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# Function to save the model and label encoders


def save_model_and_encoders(model, label_encoders):
    model.save('career_model.h5')  # Save the trained model
    # Save label encoders
    for col, le in label_encoders.items():
        joblib.dump(le, f'{col}_encoder.pkl')

# Function to load the model and label encoders if available


def load_model_and_encoders():
    if os.path.exists('career_model.h5'):
        model = load_model('career_model.h5')
        label_encoders = {}
        for col in ['Stream', 'Interest1', 'Interest2', 'Suggested_Field']:
            if os.path.exists(f'{col}_encoder.pkl'):
                label_encoders[col] = joblib.load(f'{col}_encoder.pkl')
        return model, label_encoders
    return None, None

# Function to train the model


def train_model(df):
    # Initialize label encoders
    label_encoders = {}

    # Categorical columns to encode
    categorical_columns = ['Stream', 'Interest1',
                           'Interest2', 'Suggested_Field']

    # Encode the categorical columns
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Features and target columns
    X = df[['Stream', 'Interest1', 'Interest2']]
    y = df['Suggested_Field']

    # Check if the dataset is large enough for train-test split
    if len(df) > 1:
        # Train-test split with a smaller test_size for small datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
    else:
        # Use the entire dataset for training if only one sample exists
        X_train, X_test, y_train, y_test = X, X, y, y

    # Build the neural network model
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(32, activation='relu'),
        # Number of output classes
        Dense(
            len(label_encoders['Suggested_Field'].classes_), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16,
              validation_data=(X_test, y_test))

    # Save model and encoders
    save_model_and_encoders(model, label_encoders)

    return model, label_encoders

# Function to predict based on input


def predict(model, label_encoders, sample_input):
    # Encode the input features
    encoded_input = [label_encoders[col].transform([sample_input[col]])[0] for col in [
        'Stream', 'Interest1', 'Interest2']]
    encoded_input = np.array([encoded_input])

    # Predict the suggested field
    prediction = model.predict(encoded_input)
    predicted_field_index = np.argmax(prediction)
    predicted_field = label_encoders['Suggested_Field'].inverse_transform(
        [predicted_field_index])[0]

    return predicted_field


# Main application
if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('career_recommendations.csv')

    # Try to load the model and encoders
    model, label_encoders = load_model_and_encoders()

    # Train the model if it's not already trained
    if model is None or label_encoders is None:
        model, label_encoders = train_model(df)

    # Define sample inputs for prediction
    sample_inputs = [
        {'Stream': 'Science', 'Interest1': 'Engineering', 'Interest2': 'Medical'},
        {'Stream': 'Arts', 'Interest1': 'Political Science', 'Interest2': 'History'},
        {'Stream': 'Commerce', 'Interest1': 'Finance',
            'Interest2': 'Chartered Accountant'},
        {'Stream': 'Science', 'Interest1': 'Medical', 'Interest2': 'Law'},
        {'Stream': 'Arts', 'Interest1': 'History',
            'Interest2': 'Political Science'},
        {'Stream': 'Science', 'Interest1': 'Engineering', 'Interest2': 'Law'},
        {'Stream': 'Commerce', 'Interest1': 'Finance', 'Interest2': 'Economics'},
        {'Stream': 'Arts', 'Interest1': 'History', 'Interest2': 'Economics'},
        {'Stream': 'Science', 'Interest1': 'Medical', 'Interest2': 'History'},
        {'Stream': 'Commerce', 'Interest1': 'Finance',
            'Interest2': 'Chartered Accountant'}
    ]

    # Predict for each sample input
    for sample_input in sample_inputs:
        predicted_field = predict(model, label_encoders, sample_input)
        print(f"""Input: {sample_input}, Predicted Suggested Field: {
              predicted_field}""")
