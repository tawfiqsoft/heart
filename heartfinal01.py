import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from fpdf import FPDF

def dict_to_pdf(data, result, pdf_file):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_text_color(0, 102, 204)
    pdf.set_font("Arial", size=20, style='B')
    pdf.cell(0, 10, txt="HeartLens", ln=True, align='C')
    pdf.ln(1)

    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(0, 10, txt="Heart Disease Prediction Result", ln=True, align='C')
    pdf.ln(1)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt="Website:heart-lens.streamlit.app", ln=True, align='C')
    pdf.cell(0, 10, txt="Email: contact@heartlens.com", ln=True, align='C')
    pdf.cell(0, 10, txt="Phone: +1 234 567 890", ln=True, align='C')
    pdf.ln(2)
    
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)

    for idx, (key, value) in enumerate(data.items()):
        if idx % 2 == 0:
            pdf.set_fill_color(230, 230, 230)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.cell(180, 10, txt=f"{key}: {value}", border=0, ln=True, align='L', fill=True)

    pdf.ln(5)
    pdf.set_font("Arial", size=10, style='B')
    pdf.cell(0, 10, txt="About HeartLens", ln=True, align='L')
    pdf.set_font("Arial", size=8)
    pdf.multi_cell(0, 10, txt=(
        "HeartLens is a cardiovascular health prediction platform using advanced machine learning "
        "to assess heart disease risk. Our system provides early detection insights to help users "
        "take preventive measures and consult healthcare professionals. We aim to make cardiac health "
        "monitoring accessible and reliable for everyone."
    ))

    pdf.ln(15)
    pdf.set_y(-40)
    pdf.set_font("Arial", size=10, style='I')
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, txt="Thank you for using HeartLens. Stay heart healthy!", ln=True, align='C')

    pdf.output(pdf_file)

def perform_prediction(X_train, y_train, sc_X, user_data):
    user_data_array = np.array([[
        user_data['Age'],
        user_data['Sex'],
        user_data['Chest Pain Type'],
        user_data['Resting BP'],
        user_data['Cholesterol'],
        user_data['Fasting Blood Sugar'],
        user_data['Resting ECG'],
        user_data['Max Heart Rate'],
        user_data['Exercise Angina'],
        user_data['ST Depression'],
        user_data['ST Slope'],
        user_data['Major Vessels'],
        user_data['Thalassemia']
    ]])

    userdata_scaled = sc_X.transform(user_data_array)
    classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(userdata_scaled)
    return 'Negative' if prediction[0] == 0 else 'Positive'

# Streamlit UI
st.image("https://github.com/tawfiqsoft/heart/blob/main/heart.jpg?raw=true", width=700)
st.markdown('<h1 style="text-align: center; font-size: 3.5rem; color:red;">HeartLens</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict Heart Disease</h3>", unsafe_allow_html=True)

# Load dataset
data = pd.read_csv(r"https://raw.githubusercontent.com/tawfiqsoft/heart/refs/heads/main/heart.csv")

# Preprocessing
data['thalach'] = data['thalach'].replace(0, np.NaN)
data['thalach'] = data['thalach'].fillna(data['thalach'].mean())

X = data.iloc[:, 0:13]
y = data.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Sidebar inputs
st.sidebar.title('Input Patient Information')
name = st.sidebar.text_input('Name')
age = st.sidebar.number_input('Age', min_value=1, max_value=120, step=1)
sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
cp = st.sidebar.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, step=1)
chol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
restecg = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, step=1)
exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'])
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, step=0.1)
slope = st.sidebar.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.sidebar.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3, step=1)
thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Convert categorical inputs to numerical
sex = 1 if sex == 'Male' else 0
cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
cp = cp_mapping[cp]
fbs = 1 if fbs == 'Yes' else 0
restecg_mapping = {'Normal': 0, 'ST-T Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
restecg = restecg_mapping[restecg]
exang = 1 if exang == 'Yes' else 0
slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
slope = slope_mapping[slope]
thal_mapping = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
thal = thal_mapping[thal]

user_data = {
    'Name': name,
    'Age': age,
    'Sex': 'Male' if sex == 1 else 'Female',
    'Chest Pain Type': cp,
    'Resting BP': trestbps,
    'Cholesterol': chol,
    'Fasting Blood Sugar': fbs,
    'Resting ECG': restecg,
    'Max Heart Rate': thalach,
    'Exercise Angina': exang,
    'ST Depression': oldpeak,
    'ST Slope': slope,
    'Major Vessels': ca,
    'Thalassemia': thal
}

if st.sidebar.button("Submit"):
    prediction_data = {
        'Age': age,
        'Sex': sex,
        'Chest Pain Type': cp,
        'Resting BP': trestbps,
        'Cholesterol': chol,
        'Fasting Blood Sugar': fbs,
        'Resting ECG': restecg,
        'Max Heart Rate': thalach,
        'Exercise Angina': exang,
        'ST Depression': oldpeak,
        'ST Slope': slope,
        'Major Vessels': ca,
        'Thalassemia': thal
    }
    
    result = perform_prediction(X_train, y_train, sc_X, prediction_data)
    
    user_data['Result'] = result
    pdf_file = f"{name}_heart_report.pdf"
    dict_to_pdf(user_data, result, pdf_file)

    container = st.container(border=True)
    container.write(f"Prediction Result: {result}")
    st.markdown("---")

    with open(pdf_file, "rb") as f:
        pdf_data = f.read()
        st.download_button(
            label="Download Full Report",
            data=pdf_data,
            file_name=pdf_file,
            mime="application/pdf"
        )