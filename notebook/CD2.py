import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker for realistic demographic data
fake = Faker()
np.random.seed(42)  # For reproducibility

# Number of samples
n_samples = 5000

# Initialize dataset dictionary
data = {
    'Age': np.random.normal(40, 15, n_samples).clip(18, 80),
    'Sex': np.random.choice(['Male', 'Female', 'Other'], p=[0.495, 0.495, 0.01], size=n_samples),
    'Ethnicity': np.random.choice(['South Asian', 'Caucasian', 'African', 'Hispanic', 'Other'], p=[0.4, 0.2, 0.15, 0.15, 0.1], size=n_samples),
    'BMI': np.random.normal(25, 5, n_samples).clip(18, 40),
    'Socioeconomic_Status': np.random.choice(['Low', 'Middle', 'High'], p=[0.4, 0.5, 0.1], size=n_samples),
    'Physical_Activity': np.random.choice(['Sedentary', 'Moderate', 'Active'], p=[0.5, 0.3, 0.2], size=n_samples),
    'Alcohol_Consumption': np.random.choice(['None', 'Moderate', 'Heavy'], p=[0.6, 0.3, 0.1], size=n_samples),
    'Smoking_Status': np.random.choice(['Never', 'Former', 'Current'], p=[0.6, 0.2, 0.2], size=n_samples),
    'Sleep_Duration': np.random.normal(7, 1.5, n_samples).clip(4, 10),
    'Sleep_Quality': np.random.choice(['Poor', 'Average', 'Good'], p=[0.2, 0.5, 0.3], size=n_samples),
    'Stress_Levels': np.random.choice(['Low', 'Moderate', 'High'], p=[0.3, 0.5, 0.2], size=n_samples),
    'Family_History_Diabetes': np.random.choice(['Yes', 'No'], p=[0.3, 0.7], size=n_samples),
    'Prediabetes': np.random.choice(['Yes', 'No'], p=[0.2, 0.8], size=n_samples),
    'Gestational_Diabetes': np.zeros(n_samples, dtype=object),
    'Hypertension': np.random.choice(['Yes', 'No'], p=[0.25, 0.75], size=n_samples),
    'PCOS': np.zeros(n_samples, dtype=object),
    'Cardiovascular_Disease': np.random.choice(['Yes', 'No'], p=[0.15, 0.85], size=n_samples),
    'Kidney_Problems': np.random.choice(['Yes', 'No'], p=[0.1, 0.9], size=n_samples),
    'Medications': np.random.choice(['None', 'Steroids', 'Insulin', 'Metformin', 'Other'], p=[0.6, 0.1, 0.1, 0.15, 0.05], size=n_samples),
    'Blood_Pressure_Systolic': np.random.normal(120, 15, n_samples).clip(90, 180),
    'Blood_Pressure_Diastolic': np.random.normal(80, 10, n_samples).clip(60, 120),
    'Heart_Rate': np.random.normal(75, 10, n_samples).clip(50, 120),
    'Skin_Temperature': np.random.normal(32, 1, n_samples).clip(30, 35),
    'Blood_Glucose': np.zeros(n_samples),
    'Sensor_Current_uA': np.zeros(n_samples),
    'Sensor_Voltage_mV': np.zeros(n_samples)
}

# Handle sex-specific features (Gestational Diabetes and PCOS)
for i in range(n_samples):
    if data['Sex'][i] == 'Female':
        data['Gestational_Diabetes'][i] = np.random.choice(['Yes', 'No'], p=[0.1, 0.9])
        data['PCOS'][i] = np.random.choice(['Yes', 'No'], p=[0.1, 0.9])
    else:
        data['Gestational_Diabetes'][i] = 'No'
        data['PCOS'][i] = 'No'

# Introduce correlations for Age and Hypertension
for i in range(n_samples):
    if data['Age'][i] > 50:
        data['Hypertension'][i] = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])

# Generate Blood Glucose and Sensor Outputs with conditional logic and correlations
for i in range(n_samples):
    # Base probabilities for glucose categories: Healthy, Prediabetic, Diabetic
    glucose_probs = [0.6, 0.25, 0.15]
    
    # Adjust probabilities based on features (ensure non-negative and sum to 1)
    if data['BMI'][i] > 30:
        glucose_probs[0] -= 0.2
        glucose_probs[1] += 0.1
        glucose_probs[2] += 0.1
    if data['Family_History_Diabetes'][i] == 'Yes':
        glucose_probs[0] -= 0.1
        glucose_probs[1] += 0.05
        glucose_probs[2] += 0.05
    if data['Prediabetes'][i] == 'Yes':
        glucose_probs[0] = 0.1
        glucose_probs[1] = 0.6
        glucose_probs[2] = 0.3
    if data['Sex'][i] == 'Female' and (data['PCOS'][i] == 'Yes' or data['Gestational_Diabetes'][i] == 'Yes'):
        glucose_probs[0] -= 0.1
        glucose_probs[1] += 0.05
        glucose_probs[2] += 0.05
    if data['Hypertension'][i] == 'Yes':
        glucose_probs[0] -= 0.05
        glucose_probs[1] += 0.03
        glucose_probs[2] += 0.02
    if data['Physical_Activity'][i] == 'Active':
        glucose_probs[0] += 0.1
        glucose_probs[1] -= 0.05
        glucose_probs[2] -= 0.05
    if data['Stress_Levels'][i] == 'High':
        glucose_probs[0] -= 0.05
        glucose_probs[1] += 0.03
        glucose_probs[2] += 0.02
    if data['Medications'][i] in ['Insulin', 'Metformin']:
        glucose_probs[0] += 0.1
        glucose_probs[2] -= 0.1
    if data['Ethnicity'][i] in ['South Asian', 'Hispanic']:
        glucose_probs[0] -= 0.05
        glucose_probs[1] += 0.03
        glucose_probs[2] += 0.02
    
    # Ensure probabilities are non-negative and sum to 1
    glucose_probs = np.array(glucose_probs)
    glucose_probs = np.maximum(glucose_probs, 0)  # Set negative probabilities to 0
    if np.sum(glucose_probs) == 0:  # Handle edge case where all probabilities are 0
        glucose_probs = [0.6, 0.25, 0.15]  # Revert to base probabilities
    else:
        glucose_probs /= np.sum(glucose_probs)  # Normalize to sum to 1
    
    # Sample glucose category
    category = np.random.choice(['Healthy', 'Prediabetic', 'Diabetic'], p=glucose_probs)
    
    # Assign glucose value with clipping and noise
    if category == 'Healthy':
        data['Blood_Glucose'][i] = np.clip(np.random.normal(85, 10), 70, 99) + np.random.normal(0, 0.5)
    elif category == 'Prediabetic':
        data['Blood_Glucose'][i] = np.clip(np.random.normal(110, 10), 100, 125) + np.random.normal(0, 0.5)
    else:
        data['Blood_Glucose'][i] = np.clip(np.random.normal(150, 30), 126, 300) + np.random.normal(0, 0.5)
    
    # Generate Sensor Current (μA) based on glucose level
    # Assume linear relationship: current = a * glucose + noise
    # Based on research, current ~0.1–2 μA for glucose 1–4 mg/dL
    # Since blood glucose is ~10–100x sweat glucose, scale accordingly
    sweat_glucose = data['Blood_Glucose'][i] / 50  # Approximate sweat glucose (mg/dL)
    base_current = 0.5 * sweat_glucose  # Linear scaling (adjusted to fit 0.1–2 μA range)
    temp_effect = (data['Skin_Temperature'][i] - 32) * 0.02  # Small temperature effect
    noise = np.random.normal(0, 0.05)  # Sensor noise
    data['Sensor_Current_uA'][i] = np.clip(base_current + temp_effect + noise, 0.1, 2.0)
    
    # Generate Sensor Voltage (mV) based on current
    # Assume transimpedance gain of ~200 kΩ (converts μA to mV)
    base_voltage = data['Sensor_Current_uA'][i] * 200  # Voltage = Current * Gain
    voltage_noise = np.random.normal(0, 5)  # Circuit noise
    data['Sensor_Voltage_mV'][i] = np.clip(base_voltage + voltage_noise, 50, 500)

# Create DataFrame
df = pd.DataFrame(data)

# Round continuous variables to realistic precision
df['Age'] = df['Age'].round(0).astype(int)
df['BMI'] = df['BMI'].round(1)
df['Sleep_Duration'] = df['Sleep_Duration'].round(1)
df['Blood_Pressure_Systolic'] = df['Blood_Pressure_Systolic'].round(0).astype(int)
df['Blood_Pressure_Diastolic'] = df['Blood_Pressure_Diastolic'].round(0).astype(int)
df['Heart_Rate'] = df['Heart_Rate'].round(0).astype(int)
df['Skin_Temperature'] = df['Skin_Temperature'].round(1)
df['Blood_Glucose'] = df['Blood_Glucose'].round(1)
df['Sensor_Current_uA'] = df['Sensor_Current_uA'].round(2)
df['Sensor_Voltage_mV'] = df['Sensor_Voltage_mV'].round(1)

# Save to CSV
df.to_csv('synthetic_glucose_dataset_2.csv', index=False)

# Display first few rows and basic stats
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())