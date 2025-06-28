import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def get_df():
    # try:
        data = pd.read_csv('artifacts/data.csv')
        data1 = data.drop('blood_glucose', axis=1)  # Use lowercase column name
        return data1, data
    # except Exception as e:
    #     raise CustomException(e, sys)

def normalize_value(value, min_val, max_val):
    """Normalize a value to 0-1 scale based on min and max."""
    return (value - min_val) / (max_val - min_val)

def create_radar_chart(age, bmi, sleep_duration, stress_level, avg_bp, sensor_current):
    """Create an interactive radar chart with normalized features."""
    categories = ['Age', 'BMI', 'Sleep Duration', 'Stress Level', 'Avg Blood Pressure', 'Sensor Current (µA)']
    
    # Map categorical Stress Level to numerical
    stress_map = {'Low': 1, 'Moderate': 2, 'High': 3}
    stress_value = stress_map[stress_level]
    
    # Define min and max for normalization
    ranges = {
        'Age': (1, 120),
        'BMI': (10.0, 60.0),
        'Sleep Duration': (0.0, 12.0),
        'Stress Level': (1, 3),
        'Avg Blood Pressure': (65, 160),
        'Sensor Current (µA)': (0.0, 2.0)
    }
    
    # Normalize values
    values = [
        normalize_value(age, *ranges['Age']),
        normalize_value(bmi, *ranges['BMI']),
        normalize_value(sleep_duration, *ranges['Sleep Duration']),
        normalize_value(stress_value, *ranges['Stress Level']),
        normalize_value(avg_bp, *ranges['Avg Blood Pressure']),
        normalize_value(sensor_current, *ranges['Sensor Current (µA)'])
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient Profile',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12)  
            )
        ),
        showlegend=True,
        title="Patient Profile Radar Chart",
        width=1100,
        height=500
    )
    return fig

def add_sidebar():
    with st.sidebar:
        selected = option_menu(
            'Non Invasive Glucometer: Diabetes Prediction System',
            ['Diabetes Prediction'],
            icons=['activity'],
            default_index=0
        )
    return selected

def main():
    st.set_page_config(
        page_title="Non Invasive Glucometer",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    selected = add_sidebar()
    if selected == 'Diabetes Prediction':
        with st.container():
            st.title("Non Invasive Glucometer")
            st.subheader("AI-based Prediction Powered by Patient Biometrics & Lifestyle Data")
            st.markdown("##### Please connect this app to your glucometer to help diagnose blood sugar level from your test parameters and improve the accuracy. This app predicts the blood sugar level based on diagnostic measures.")

        col11, spacer, col22 = st.columns([5, 0.1, 2])
        with col11:
            data, _ = get_df()
            with st.form("user_input_form"):
                st.header("🧍 Patient Details")
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age", min_value=1, max_value=100, value=30)
                    sex = st.selectbox("Sex", ['Male', 'Female', 'Other'])
                    ethnicity = st.selectbox("Ethnicity", ['South Asian', 'Hispanic', 'Caucasian', 'Other', 'African'])
                    bmi = st.number_input("BMI", min_value=10.0, max_value=30.0, value=24.5)
                    sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0)
                    physical_activity = st.selectbox("Physical Activity", ['Sedentary', 'Moderate', 'Active'])
                with col2:
                    blood_pressure_systolic = st.number_input("Systolic BP", 80, 200, 120)
                    blood_pressure_diastolic = st.number_input("Diastolic BP", 50, 120, 80)
                    heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 80)
                    skin_temperature = st.number_input("Skin Temperature (°C)", 30.0, 38.0, 34.5)
                    sensor_current_ua = st.slider("Sensor Current (µA)", 0.0, 2.0, 0.5)

                st.header("🧬 Lifestyle & Medical History")
                col3, col4 = st.columns(2)
                with col3:
                    stress_levels = st.selectbox("Stress Levels", ['Low', 'Moderate', 'High'])
                    alcohol_consumption = st.selectbox("Alcohol Consumption", ['none', 'Moderate', 'Heavy'])
                    smoking_status = st.selectbox("Smoking Status", ['Never', 'Former', 'Current'])
                    sleep_quality = st.selectbox("Sleep Quality", ['Poor', 'Average', 'Good'])
                    family_history_diabetes = st.selectbox("Family History of Diabetes", ['Yes', 'No'])
                    medications = st.selectbox("Currently on Any Medication?", ['Metformin', 'none','Steroids','Insulin','Other'])
                    bmi_category = st.selectbox("BMI Category", ['Underweight', 'Normal', 'Overweight', 'Obese'])
                with col4:
                    socioeconomic_status = st.selectbox("Socioeconomic Status", ['Low', 'Middle', 'High'])
                    prediabetes = st.selectbox("Prediabetes History", ['Yes', 'No'])
                    gestational_diabetes = st.selectbox("Gestational Diabetes", ['Yes', 'No'])
                    hypertension = st.selectbox("Hypertension", ['Yes', 'No'])
                    pcos = st.selectbox("PCOS", ['Yes', 'No'])
                    cardiovascular_disease = st.selectbox("Cardiovascular Disease", ['Yes', 'No'])
                    kidney_problems = st.selectbox("Kidney Problems", ['Yes', 'No'])

                st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis")
                submitted = st.form_submit_button("Predict Glucose Level")

                if submitted:
                    # Prepare input data using CustomData for prediction pipeline
                    custom_data = CustomData(
                        age=age,
                        bmi=bmi,
                        sleep_duration=sleep_duration,
                        blood_pressure_systolic=blood_pressure_systolic,
                        blood_pressure_diastolic=blood_pressure_diastolic,
                        heart_rate=heart_rate,
                        skin_temperature=skin_temperature,
                        sensor_current_ua=sensor_current_ua,
                        sex=sex,
                        ethnicity=ethnicity,
                        socioeconomic_status=socioeconomic_status,
                        physical_activity=physical_activity,
                        alcohol_consumption=alcohol_consumption,
                        smoking_status=smoking_status,
                        sleep_quality=sleep_quality,
                        stress_levels=stress_levels,
                        family_history_diabetes=family_history_diabetes,
                        prediabetes=prediabetes,
                        gestational_diabetes=gestational_diabetes,
                        hypertension=hypertension,
                        pcos=pcos,
                        cardiovascular_disease=cardiovascular_disease,
                        kidney_problems=kidney_problems,
                        medications=medications,
                        bmi_category=bmi_category
                    )
                    input_df = custom_data.get_data_as_data_frame()

                    # Perform prediction
                    predict_pipeline = PredictPipeline()
                    prediction = predict_pipeline.predict(input_df)[0]
                    st.success(f"Predicted Blood Glucose Level: {prediction:.1f} mg/dL")

                    st.subheader("Prediction Result")
                    res_col1, res_col2 = st.columns([1, 2])

                    with res_col1:
                        st.metric(label="Glucose Level (mg/dL)", value=round(prediction, 2))

                        if prediction < 100:
                            stage = "Normal"
                            color = "🟢"
                        elif prediction < 126:
                            stage = "Pre-diabetic"
                            color = "🟡"
                        else:
                            stage = "Diabetic"
                            color = "🔴"

                        st.markdown(f"**Risk Category:** {color} **{stage}**")

                    with res_col2:
                        st.markdown("### 📌 Doctor's Note")
                        if stage == "Normal":
                            st.success("✅ You're in a healthy range. Keep up your current habits!")
                        elif stage == "Pre-diabetic":
                            st.warning("⚠️ Lifestyle modifications can help prevent diabetes. Consider improving diet & exercise.")
                        else:
                            st.error("❗ This level is considered diabetic. Please consult a healthcare provider immediately.")


        with col22:
            # if submitted:
            # Calculate average blood pressure
            avg_bp = (blood_pressure_systolic + blood_pressure_diastolic) / 2
            # Create and display radar chart
            fig = create_radar_chart(age, bmi, sleep_duration, stress_levels, avg_bp, sensor_current_ua)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("🤖 Training Data Summary")
            st.markdown("Summary statistics of the training data used for model development:")

            _, df = get_df()
            with st.expander("📁 View Training Data Summary"):
                st.dataframe(df.describe().T.style.format(precision=2), use_container_width=True)
                # 📈 Add Multivariable Bar/Line Chart
                # 📊 Summary Stats Table
            _, df = get_df()
            selected_cols = ['age', 'blood_glucose', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'bmi']
            st.subheader("📈 Visualisation")
            # Melt only selected columns
            df_melted = df[selected_cols].melt(var_name="Feature", value_name="Value")
            # Add a dropdown for feature selection
            feature_options = st.multiselect("Select features to visualize", selected_cols, default=selected_cols)
            if feature_options:
                df_melted = df_melted[df_melted['Feature'].isin(feature_options)]
                # Plot using Plotly with bar chart
                fig_bar = px.bar(
                    df_melted,
                    x="Feature",
                    y="Value",
                    color="Feature",
                    title="",
                    labels={"Value": "Feature Value", "Feature": "Features"},
                    height=500
                )
                fig_bar.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=30, r=30, t=60, b=30),
                    bargap=0.2  # Add space between bars
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Add interpretation and health tips using predicted blood glucose
                if submitted:
                    st.subheader("📝 Insights & Recommendations")
                    avg_values = df[feature_options].mean().round(2)
                    st.write("**Average Values in Training Data:**", avg_values.to_dict())
                    if 'bmi' in feature_options and bmi > 25:
                        st.write(f"- **High BMI Alert**: Your BMI ({bmi:.1f}) is above the healthy threshold (25). Consider consulting a doctor and increasing physical activity.")
                    if 'blood_glucose' in feature_options and prediction > 126:
                        st.write(f"- **Elevated Glucose Alert**: Your predicted blood glucose ({prediction:.1f} mg/dL) is above the normal threshold (126 mg/dL). Monitor levels closely and consult a healthcare provider.")
                    if 'age' in feature_options and age > 45:
                        st.write(f"- **Age Consideration**: Your age ({age}) is a risk factor. Regular check-ups are recommended.")

if __name__ == '__main__':
    main()

















# import streamlit as st
# import pandas as pd
# import numpy as np
# from streamlit_option_menu import option_menu
# import plotly.graph_objects as go
# import plotly.express as px
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# def get_df():
#     # try:
#         data = pd.read_csv('artifacts/data.csv')
#         data1 = data.drop('blood_glucose', axis=1)  # Use lowercase column name
#         return data1, data
#     # except Exception as e:
#     #     raise CustomException(e, sys)

# def normalize_value(value, min_val, max_val):
#     """Normalize a value to 0-1 scale based on min and max."""
#     return (value - min_val) / (max_val - min_val)

# def create_radar_chart(age, bmi, sleep_duration, stress_level, avg_bp, sensor_current):
#     """Create an interactive radar chart with normalized features."""
#     categories = ['Age', 'BMI', 'Sleep Duration', 'Stress Level', 'Avg Blood Pressure', 'Sensor Current (µA)']
    
#     # Map categorical Stress Level to numerical
#     stress_map = {'Low': 1, 'Medium': 2, 'High': 3}
#     stress_value = stress_map[stress_level]
    
#     # Define min and max for normalization
#     ranges = {
#         'Age': (1, 120),
#         'BMI': (10.0, 60.0),
#         'Sleep Duration': (0.0, 12.0),
#         'Stress Level': (1, 3),
#         'Avg Blood Pressure': (65, 160),
#         'Sensor Current (µA)': (0.0, 100.0)
#     }
    
#     # Normalize values
#     values = [
#         normalize_value(age, *ranges['Age']),
#         normalize_value(bmi, *ranges['BMI']),
#         normalize_value(sleep_duration, *ranges['Sleep Duration']),
#         normalize_value(stress_value, *ranges['Stress Level']),
#         normalize_value(avg_bp, *ranges['Avg Blood Pressure']),
#         normalize_value(sensor_current, *ranges['Sensor Current (µA)'])
#     ]
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatterpolar(
#         r=values,
#         theta=categories,
#         fill='toself',
#         name='Patient Profile',
#         line=dict(color='blue')
#     ))
    
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1],
#                 tickfont=dict(size=12)  
#             )
#         ),
#         showlegend=True,
#         title="Patient Profile Radar Chart",
#         width=1100,
#         height=500
#     )
#     return fig

# def add_sidebar():
#     with st.sidebar:
#         selected = option_menu(
#             'Non Invasive Glucometer: Diabetes Prediction System',
#             ['Diabetes Prediction'],
#             icons=['activity'],
#             default_index=0
#         )
#     return selected

# def main():
#     st.set_page_config(
#         page_title="Non Invasive Glucometer",
#         page_icon=":female-doctor:",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     selected = add_sidebar()
#     if selected == 'Diabetes Prediction':
#         with st.container():
#             st.title("Non Invasive Glucometer")
#             st.subheader("AI-based Prediction Powered by Patient Biometrics & Lifestyle Data")
#             st.markdown("##### Please connect this app to your glucometer to help diagnose blood sugar level from your test parameters and improve the accuracy. This app predicts the blood sugar level based on diagnostic measures.")

#         col11, spacer, col22 = st.columns([5, 0.1, 2])
#         with col11:
#             data, _ = get_df()
#             with st.form("user_input_form"):
#                 st.header("🧍 Patient Details")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     age = st.number_input("Age", min_value=1, max_value=120, value=30)
#                     sex = st.selectbox("Sex", ['Male', 'Female', 'Other'])
#                     bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5)
#                     sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
#                     physical_activity = st.selectbox("Physical Activity", ['Low', 'Moderate', 'High'])
#                 with col2:
#                     blood_pressure_systolic = st.number_input("Systolic BP", 80, 200, 120)
#                     blood_pressure_diastolic = st.number_input("Diastolic BP", 50, 120, 80)
#                     heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 72)
#                     skin_temperature = st.number_input("Skin Temperature (°C)", 30.0, 42.0, 36.5)
#                     sensor_current_ua = st.slider("Sensor Current (µA)", 0.0, 100.0, 50.0)

#                 st.header("🧬 Lifestyle & Medical History")
#                 col3, col4 = st.columns(2)
#                 with col3:
#                     stress_levels = st.selectbox("Stress Levels", ['Low', 'Medium', 'High'])
#                     alcohol_consumption = st.selectbox("Alcohol Consumption", ['Never', 'Rarely', 'Frequently'])
#                     smoking_status = st.selectbox("Smoking Status", ['Never', 'Former', 'Current'])
#                     sleep_quality = st.selectbox("Sleep Quality", ['Poor', 'Average', 'Good'])
#                     family_history_diabetes = st.selectbox("Family History of Diabetes", ['Yes', 'No'])
#                     medications = st.selectbox("Currently on Medication", ['Yes', 'No'])
#                     bmi_category = st.selectbox("BMI Category", ['Underweight', 'Normal', 'Overweight', 'Obese'])
#                 with col4:
#                     socioeconomic_status = st.selectbox("Socioeconomic Status", ['Low', 'Middle', 'High'])
#                     prediabetes = st.selectbox("Prediabetes History", ['Yes', 'No'])
#                     gestational_diabetes = st.selectbox("Gestational Diabetes", ['Yes', 'No'])
#                     hypertension = st.selectbox("Hypertension", ['Yes', 'No'])
#                     pcos = st.selectbox("PCOS", ['Yes', 'No'])
#                     cardiovascular_disease = st.selectbox("Cardiovascular Disease", ['Yes', 'No'])
#                     kidney_problems = st.selectbox("Kidney Problems", ['Yes', 'No'])

#                 st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis")
#                 submitted = st.form_submit_button("Predict Glucose Level")

#                 if submitted:
#                     # Prepare input data using CustomData for prediction pipeline
#                     custom_data = CustomData(
#                         age=age,
#                         bmi=bmi,
#                         sleep_duration=sleep_duration,
#                         blood_pressure_systolic=blood_pressure_systolic,
#                         blood_pressure_diastolic=blood_pressure_diastolic,
#                         heart_rate=heart_rate,
#                         skin_temperature=skin_temperature,
#                         sensor_current_ua=sensor_current_ua,
#                         sex=sex,
#                         ethnicity="",  # Placeholder, not in form; adjust if needed
#                         socioeconomic_status=socioeconomic_status,
#                         physical_activity=physical_activity,
#                         alcohol_consumption=alcohol_consumption,
#                         smoking_status=smoking_status,
#                         sleep_quality=sleep_quality,
#                         stress_levels=stress_levels,
#                         family_history_diabetes=family_history_diabetes,
#                         prediabetes=prediabetes,
#                         gestational_diabetes=gestational_diabetes,
#                         hypertension=hypertension,
#                         pcos=pcos,
#                         cardiovascular_disease=cardiovascular_disease,
#                         kidney_problems=kidney_problems,
#                         medications=medications,
#                         bmi_category=bmi_category
#                     )
#                     input_df = custom_data.get_data_as_data_frame()

#                     # Perform prediction
#                     predict_pipeline = PredictPipeline()
#                     prediction = predict_pipeline.predict(input_df)[0]
#                     st.success(f"Predicted Blood Glucose Level: {prediction:.1f} mg/dL")

#                     st.subheader("Prediction Result")
#                     res_col1, res_col2 = st.columns([1, 2])

#                     with res_col1:
#                         st.metric(label="Glucose Level (mg/dL)", value=round(prediction, 2))

#                         if prediction < 100:
#                             stage = "Normal"
#                             color = "🟢"
#                         elif prediction < 126:
#                             stage = "Pre-diabetic"
#                             color = "🟡"
#                         else:
#                             stage = "Diabetic"
#                             color = "🔴"

#                         st.markdown(f"**Risk Category:** {color} **{stage}**")

#                     with res_col2:
#                         st.markdown("### 📌 Doctor's Note")
#                         if stage == "Normal":
#                             st.success("✅ You're in a healthy range. Keep up your current habits!")
#                         elif stage == "Pre-diabetic":
#                             st.warning("⚠️ Lifestyle modifications can help prevent diabetes. Consider improving diet & exercise.")
#                         else:
#                             st.error("❗ This level is considered diabetic. Please consult a healthcare provider immediately.")


#         with col22:
#             # if submitted:
#             # Calculate average blood pressure
#             avg_bp = (blood_pressure_systolic + blood_pressure_diastolic) / 2
#             # Create and display radar chart
#             fig = create_radar_chart(age, bmi, sleep_duration, stress_levels, avg_bp, sensor_current_ua)
#             st.plotly_chart(fig, use_container_width=True)
#             st.subheader("🤖 Training Data Summary")
#             st.markdown("Summary statistics of the training data used for model development:")

#             _, df = get_df()
#             with st.expander("📁 View Training Data Summary"):
#                 st.dataframe(df.describe().T.style.format(precision=2), use_container_width=True)
#                 # 📈 Add Multivariable Bar/Line Chart
#                 # 📊 Summary Stats Table
#             _, df = get_df()
#             selected_cols = ['age', 'blood_glucose', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'bmi']
#             st.subheader("📈 Visualisation")
#             # Melt only selected columns
#             df_melted = df[selected_cols].melt(var_name="Feature", value_name="Value")
#             # Add a dropdown for feature selection
#             feature_options = st.multiselect("Select features to visualize", selected_cols, default=selected_cols)
#             if feature_options:
#                 df_melted = df_melted[df_melted['Feature'].isin(feature_options)]
#                 # Plot using Plotly with bar chart
#                 fig_bar = px.bar(
#                     df_melted,
#                     x="Feature",
#                     y="Value",
#                     color="Feature",
#                     title="",
#                     labels={"Value": "Feature Value", "Feature": "Features"},
#                     height=500
#                 )
#                 fig_bar.update_layout(
#                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
#                     margin=dict(l=30, r=30, t=60, b=30),
#                     bargap=0.2  # Add space between bars
#                 )
#                 st.plotly_chart(fig_bar, use_container_width=True)

#                 # Add interpretation and health tips using predicted blood glucose
#                 if submitted:
#                     st.subheader("📝 Insights & Recommendations")
#                     avg_values = df[feature_options].mean().round(2)
#                     st.write("**Average Values in Training Data:**", avg_values.to_dict())
#                     if 'bmi' in feature_options and bmi > 25:
#                         st.write(f"- **High BMI Alert**: Your BMI ({bmi:.1f}) is above the healthy threshold (25). Consider consulting a doctor and increasing physical activity.")
#                     if 'blood_glucose' in feature_options and prediction > 126:
#                         st.write(f"- **Elevated Glucose Alert**: Your predicted blood glucose ({prediction:.1f} mg/dL) is above the normal threshold (126 mg/dL). Monitor levels closely and consult a healthcare provider.")
#                     if 'age' in feature_options and age > 45:
#                         st.write(f"- **Age Consideration**: Your age ({age}) is a risk factor. Regular check-ups are recommended.")

# if __name__ == '__main__':
#     main()


