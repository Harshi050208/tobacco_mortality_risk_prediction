import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Initialize model
model = None
dataset_file = "combined_mortality_dataset.csv"

def train_model():
    """
    Train a Random Forest model for tobacco mortality prediction using combined_mortality_dataset.csv.
    """
    global model
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_file)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {dataset_file} not found!")
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")
    
    # Prepare features for training
    # We'll use available features from the dataset and create synthetic individual records
    # based on the aggregated data patterns
    
    # Extract relevant columns that can inform our model
    feature_cols = []
    
    # Age group columns (we'll use these to create age features)
    age_group_cols = ['16-24', '25-34', '35-49', '50-59', '60 and Over']
    
    # Income and expenditure features
    income_cols = ["Real Households' Disposable Income", 
                   'Household Expenditure on Tobacco',
                   'Expenditure on Tobacco as a Percentage of Expenditure']
    
    # Check which columns exist and have non-null values
    available_cols = []
    for col in age_group_cols + income_cols:
        if col in df.columns and df[col].notna().any():
            available_cols.append(col)
    
    # Create training data from the dataset
    # Since the dataset is aggregated, we'll generate individual records based on patterns
    training_data = []
    training_labels = []
    
    # Use rows with mortality_target and available features
    valid_rows = df.dropna(subset=['mortality_target'])
    
    for idx, row in valid_rows.iterrows():
        # Extract mortality target
        target = int(row['mortality_target'])
        
        # Create multiple synthetic individual records from each aggregated row
        # This allows us to train on the patterns in the data
        n_samples_per_row = 5
        
        for _ in range(n_samples_per_row):
            # Age: sample from age groups or use a representative age
            if pd.notna(row.get('16-24')):
                age = np.random.randint(16, 25)
            elif pd.notna(row.get('25-34')):
                age = np.random.randint(25, 35)
            elif pd.notna(row.get('35-49')):
                age = np.random.randint(35, 50)
            elif pd.notna(row.get('50-59')):
                age = np.random.randint(50, 60)
            elif pd.notna(row.get('60 and Over')):
                age = np.random.randint(60, 80)
            else:
                age = np.random.randint(18, 80)
            
            # Cigarettes per day: estimate from tobacco expenditure
            if pd.notna(row.get('Household Expenditure on Tobacco')):
                # Rough estimate: higher expenditure = more cigarettes
                expenditure = row['Household Expenditure on Tobacco']
                cigarettes = max(0, min(40, int(expenditure / 10))) if expenditure > 0 else np.random.randint(0, 20)
            else:
                cigarettes = np.random.randint(0, 30)
            
            # Years of smoking: estimate based on age
            years = max(0, min(age - 16, 50)) if age > 16 else 0
            
            # Income: use disposable income if available
            if pd.notna(row.get("Real Households' Disposable Income")):
                income = int(row["Real Households' Disposable Income"])
            else:
                income = np.random.randint(20000, 150000)
            
            # Disease: based on mortality target and diagnosis type
            if 'Diagnosis Type' in row and pd.notna(row['Diagnosis Type']):
                disease = 1 if 'smoking' in str(row['Diagnosis Type']).lower() or target == 1 else 0
            else:
                disease = 1 if target == 1 else np.random.randint(0, 2)
            
            training_data.append([age, cigarettes, years, income, disease])
            training_labels.append(target)
    
    # Convert to numpy arrays
    X = np.array(training_data)
    y = np.array(training_labels)
    
    if len(X) == 0:
        raise ValueError("No valid training data could be extracted from the dataset!")
    
    print(f"Training data prepared: {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def predict(input_data):
    """
    Predict mortality risk based on input features.
    
    Args:
        input_data: List of 5 features [age, cigarettes, years, income, disease]
    
    Returns:
        int: 0 for Low Mortality Risk, 1 for High Mortality Risk
    """
    global model
    
    # Train model if not already trained
    if model is None:
        train_model()
    
    # Validate input
    if len(input_data) != 5:
        raise ValueError("Input data must contain exactly 5 features")
    
    # Convert to numpy array and reshape for prediction
    features = np.array(input_data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return int(prediction)

def get_prediction_probability(input_data):
    """
    Get the probability of high mortality risk.
    
    Args:
        input_data: List of 5 features [age, cigarettes, years, income, disease]
    
    Returns:
        float: Probability of high mortality risk (0.0 to 1.0)
    """
    global model
    
    # Train model if not already trained
    if model is None:
        train_model()
    
    # Validate input
    if len(input_data) != 5:
        raise ValueError("Input data must contain exactly 5 features")
    
    # Convert to numpy array and reshape for prediction
    features = np.array(input_data).reshape(1, -1)
    
    # Get prediction probability
    probability = model.predict_proba(features)[0][1]  # Probability of class 1 (high risk)
    
    return float(probability)

# Initialize model on import (optional - can be lazy loaded)
# Uncomment the line below if you want to train the model immediately when the module is imported
# train_model()

if __name__ == "__main__":
    # Train and test the model
    print("Training tobacco mortality prediction model from combined_mortality_dataset.csv...")
    train_model()
    
    # Test prediction
    print("\n" + "="*50)
    print("Testing predictions:")
    print("="*50)
    
    test_cases = [
        [45, 20, 25, 50000, 1],  # High risk: middle-aged, heavy smoker, long history, disease
        [25, 5, 3, 80000, 0],    # Low risk: young, light smoker, short history, no disease
        [60, 30, 40, 30000, 1],  # High risk: older, heavy smoker, very long history, disease
        [30, 0, 0, 100000, 0],   # Low risk: young, non-smoker, no disease
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        age, cig, years, income, disease = test_case
        prediction = predict(test_case)
        probability = get_prediction_probability(test_case)
        risk_level = "High" if prediction == 1 else "Low"
        
        print(f"\nTest Case {i}:")
        print(f"  Age: {age}, Cigarettes/day: {cig}, Years: {years}, Income: ${income:,}, Disease: {disease}")
        print(f"  Prediction: {risk_level} Mortality Risk (Probability: {probability:.2%})")
