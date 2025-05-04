import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/model_xgb.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the scaler - we'll create a new one since we don't have the saved scaler
@st.cache_resource
def get_scaler():
    try:
        # Load dataset to fit the scaler
        df = pd.read_csv('data/encoded_data.csv')
        df = df[df['Status'] != 1]  # Remove 'Enrolled' status
        df.loc[:, 'Status'] = df['Status'].replace({2: 1})  # Replace 'Graduate' with 1
        X = df.drop(columns=['Status'], axis=1)
        
        # Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler
    except Exception as e:
        st.error(f"Error loading data or creating scaler: {e}")
        return None

# Function to make prediction
def predict_dropout(features, model, scaler):
    # Create a DataFrame with the input features
    feature_names = ['Marital_status', 'Application_mode', 'Application_order', 'Course',
                    'Daytime_evening_attendance', 'Previous_qualification', 
                    'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
                    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
                    'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
                    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'Age_at_enrollment',
                    'International', 'Curricular_units_1st_sem_credited',
                    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
                    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
                    'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited',
                    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
                    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
                    'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
                    'Inflation_rate', 'GDP']
    
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale features
    scaled_features = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0]
    
    return prediction, probability

# Define mapping dictionaries for categorical features
@st.cache_data
def get_category_mappings():
    # Marital Status
    marital_status_map = {
        1: "Single",
        2: "Married", 
        3: "Widowed", 
        4: "Divorced"
    }
    
    # Nationality mapping - create a more descriptive mapping
    nationality_map = {
        1: "Portuguese",
        2: "German",
        6: "Spanish",
        11: "Italian",
        13: "Dutch",
        14: "English",
        17: "Lithuanian",
        21: "Angolan",
        22: "Cape Verdean",
        24: "Guinean",
        25: "Mozambican",
        26: "Santomean",
        32: "Turkish",
        41: "Brazilian",
        62: "Romanian",
        100: "Moldova (Republic of)",
        101: "Mexican",
        103: "Ukrainian",
        105: "Russian",
        108: "Cuban",
        109: "Colombian"
    }
    
    # Application mode mapping
    application_mode_map = {
        1: "1st phase - general contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores Island)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (bachelor)",
        16: "1st phase - special contingent (Madeira Island)",
        17: "2nd phase - general contingent",
        18: "3rd phase - general contingent",
        26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
        27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)"
    }
    
    # Course mapping
    course_map = {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equiniculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening)",
    }
    
    # Education level mapping (for mother's and father's qualification)
    education_level_map = {
        1: "Primary Education (4th grade)",
        2: "Basic Education 2nd Cycle (6th grade)",
        3: "Basic Education 3rd Cycle (9th grade)",
        4: "Secondary Education (12th grade)",
        5: "Higher Education - Bachelor's Degree",
        9: "Higher Education - Licentiate",
        11: "Higher Education - Master's",
        12: "Higher Education - Doctorate",
        19: "Higher Education (unspecified)",
        30: "Frequency of Higher Education",
        34: "Post-Graduate Specialization",
        36: "Higher Education - Degree (5 years)",
        37: "Cannot read or write",
        38: "Can read without having a 4th grade education",
        39: "Basic Education 1st Cycle (reading capability)",
        40: "Unknown",
        41: "Basic Education (unspecified)",
        42: "Technical-Professional Course"
    }
    
    # Occupation mapping - simplified version
    occupation_map = {
        0: "Student",
        1: "Representatives of Legislative and Executive Bodies",
        2: "Intellectual and Scientific Activities",
        3: "Technicians and Associate Professionals",
        4: "Administrative Staff",
        5: "Personal Service, Security and Safety Workers",
        6: "Farmers and Skilled Agricultural Workers",
        7: "Skilled Manufacturing Workers",
        8: "Plant and Machine Operators",
        9: "Unskilled Workers",
        90: "Armed Forces",
        99: "Unknown",
        121: "Health Professionals",
        124: "Teachers",
        125: "Specialists in Information Technology",
        141: "Legal Professionals",
        144: "Physical and Engineering Science Professionals",
        151: "Sales Workers",
        161: "Personal Care Workers",
        175: "Food Processing and Related Trades Workers",
        182: "Electricity and Electronics Trades Workers",
        191: "Building Construction Workers",
        192: "Metal, Machinery Workers",
        193: "Handicraft and Printing Workers",
        194: "Stationary Plant and Machine Operators",
        195: "Assemblers"
    }
    
    # Previous qualification mapping
    previous_qualification_map = {
        1: "Secondary Education",
        2: "Higher Education - Bachelor's Degree",
        3: "Higher Education - Degree",
        4: "Higher Education - Master's",
        5: "Higher Education - Doctorate",
        6: "Frequency of Higher Education",
        9: "12th Year of Schooling - Not Completed",
        10: "11th Year of Schooling - Not Completed",
        12: "Other - 11th Year",
        14: "10th Year of Schooling",
        15: "10th Year of Schooling - Not Completed",
        19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent",
        38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equivalent",
        39: "Technological Specialization Course",
        40: "Higher Education - Degree (1st Cycle)",
        42: "Professional Higher Technical Course",
        43: "Higher Education - Master (2nd Cycle)"
    }
    
    return {
        "marital_status": marital_status_map,
        "nationality": nationality_map,
        "application_mode": application_mode_map,
        "course": course_map,
        "education_level": education_level_map,
        "occupation": occupation_map,
        "previous_qualification": previous_qualification_map
    }

def generate_intervention_plan(student_data, prediction_result, dropout_probability):
    """Generate a personalized intervention plan based on student data and prediction"""
    st.subheader("Personalized Intervention Plan")
    
    # Create a modern styled container for the plan
    st.markdown("""
    <style>
    .plan-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .risk-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 4px;
        transition: transform 0.2s ease;
    }
    .risk-card:hover {
        transform: translateY(-5px);
    }
    .risk-header {
        display: flex;
        align-items: center;
        margin-bottom: 4px;
    }
    .risk-icon {
        background-color: #f0f8ff;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 15px;
        color: #3498db;
        font-size: 20px;
    }
    .risk-title {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
    }
    .risk-description {
        padding-left: 55px;
        color: #4a5568;
        font-size: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .interventions-container {
        background-color: #f8fafc;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        padding: 8px 14px;
    }
    .intervention-item {
        display: flex;
        margin-bottom: 8px;
        font-size: 20px;
        color: #2d3748;
        line-height: 1.5;
    }
    .intervention-bullet {
        color: #3498db;
        margin-right: 10px;
        font-weight: bold;
    }
    .intervention-text {
        color: #2d3748;
        font-size: 16px;
    }
    .risk-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        margin-left: 10px;
    }
    .risk-high {
        background-color: rgba(231, 76, 60, 0.15);
        color: #c0392b;
    }
    .risk-medium {
        background-color: rgba(243, 156, 18, 0.15);
        color: #d35400;
    }
    .risk-low {
        background-color: rgba(46, 204, 113, 0.15);
        color: #27ae60;
    }
    .timeline-card {
        background-color: #f8fafc;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .timeline-header {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        font-size: 16px;
    }
    .timeline-icon {
        margin-right: 10px;
        color: #3498db;
    }
    .timeline-items {
        padding-left: 25px;
    }
    .timeline-item {
        display: flex;
        margin-bottom: 8px;
        font-size: 15px;
        color: #2d3748;
        line-height: 1.5;
    }
    .timeline-item:last-child {
        margin-bottom: 0;
    }
    .section-header {
        background: linear-gradient(90deg, #3498db, #2c3e50);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 30px 0 20px 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        font-size: 17px;
    }
    .section-icon {
        margin-right: 10px;
    }
    </style>
    <div class="plan-container">
    """, unsafe_allow_html=True)
    
    # Determine risk level
    if dropout_probability > 0.8:
        risk_level = "High Risk"
        risk_class = "risk-high"
        risk_icon = "⚠️"
    elif dropout_probability > 0.6:
        risk_level = "Medium Risk"
        risk_class = "risk-medium"
        risk_icon = "⚠️"
    else:
        risk_level = "Low Risk"
        risk_class = "risk-low"
        risk_icon = "✓"
    
    # Display student overview
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">📊</div>
        Student Risk Assessment
    </div>
    <div class="risk-card">
        <div class="risk-header">
            <div class="risk-icon">{risk_icon}</div>
            <div class="risk-title">Risk Profile <span class="risk-badge {risk_class}">{risk_level}</span></div>
        </div>
        <div class="risk-description">
            Based on our predictive model analysis, this student has a dropout probability of <strong>{dropout_probability:.1%}</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Identify key risk areas
    risk_areas = []
    
    # Academic performance risks
    if student_data['Curricular_units_1st_sem_approved'] / max(student_data['Curricular_units_1st_sem_enrolled'], 1) < 0.6:
        risk_areas.append({
            "area": "Academic Performance (1st Semester)",
            "icon": "📚",
            "description": f"Student passed only {student_data['Curricular_units_1st_sem_approved']} of {student_data['Curricular_units_1st_sem_enrolled']} courses.",
            "interventions": [
                "Weekly meetings with an academic advisor",
                "Enrollment in supplemental instruction sessions",
                "Consideration for course load reduction"
            ]
        })
    
    if student_data['Curricular_units_2nd_sem_approved'] / max(student_data['Curricular_units_2nd_sem_enrolled'], 1) < 0.6:
        risk_areas.append({
            "area": "Academic Performance (2nd Semester)",
            "icon": "📝",
            "description": f"Student passed only {student_data['Curricular_units_2nd_sem_approved']} of {student_data['Curricular_units_2nd_sem_enrolled']} courses.",
            "interventions": [
                "Mandatory tutoring sessions for difficult subjects",
                "Mid-semester progress review with program coordinator",
                "Study skills workshop attendance"
            ]
        })
    
    # Financial risks
    if student_data['Debtor'] == 1 or student_data['Tuition_fees_up_to_date'] == 0:
        risk_areas.append({
            "area": "Financial Stability",
            "icon": "💰",
            "description": "Student has financial difficulties with tuition payments.",
            "interventions": [
                "Financial aid office consultation",
                "Payment plan options review",
                "Emergency scholarship or grant application",
                "Part-time campus employment opportunities"
            ]
        })
    
    # Social integration risks (if relevant data available)
    if student_data['International'] == 1 or student_data['Displaced'] == 1:
        risk_areas.append({
            "area": "Social Integration",
            "icon": "👥",
            "description": "Student may face adjustment challenges due to international/displaced status.",
            "interventions": [
                "Connection with student communities for similar backgrounds",
                "Regular check-ins with international student office",
                "Cultural transition support group",
                "Mentor matching program"
            ]
        })
    
    # If no specific risk areas identified but still at risk
    if not risk_areas and dropout_probability > 0.5:
        risk_areas.append({
            "area": "Multiple Factors",
            "icon": "🔍",
            "description": "No single major risk factor identified, but combination of minor factors increases dropout risk.",
            "interventions": [
                "Holistic student success meeting with advisor",
                "Regular progress monitoring",
                "Student engagement opportunities assessment",
                "Academic and personal goal-setting workshop"
            ]
        })
    
    # Display intervention plan for each risk area
    if risk_areas:
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon">🎯</div>
            Targeted Intervention Areas
        </div>
        """, unsafe_allow_html=True)
        
        for area in risk_areas:
            st.markdown(f"""
            <div class="risk-card">
                <div class="risk-header">
                    <div class="risk-icon">{area["icon"]}</div>
                    <div class="risk-title">{area["area"]}</div>
                </div>
                <div class="risk-description">{area["description"]}</div>
                <div class="interventions-container">
                    {"".join([f'<div class="intervention-item"><div class="intervention-bullet">•</div><div class="intervention-text">{intervention}</div></div>' for intervention in area["interventions"]])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
        # Next steps and timeline
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon">⏱️</div>
            Implementation Timeline
        </div>
        """, unsafe_allow_html=True)
        
        # Timeline cards
        st.markdown(f"""
        <div class="timeline-card">
            <div class="timeline-header">
                <div class="timeline-icon">🔵</div>
                Immediate (Next 7 days)
            </div>
            <div class="timeline-items">
                <div class="timeline-item">• Initial contact with student via email and phone</div>
                <div class="timeline-item">• Schedule first advising appointment</div>
                <div class="timeline-item">• Provide resource information packet</div>
            </div>
        </div>
        
        <div class="timeline-card">
            <div class="timeline-header">
                <div class="timeline-icon">🔵</div>
                Short-term (2-4 weeks)
            </div>
            <div class="timeline-items">
                <div class="timeline-item">• Complete all initial consultations for identified risk areas</div>
                <div class="timeline-item">• Establish regular check-in schedule</div>
                <div class="timeline-item">• Set measurable goals for improvement</div>
            </div>
        </div>
        
        <div class="timeline-card">
            <div class="timeline-header">
                <div class="timeline-icon">🔵</div>
                Long-term (Semester)
            </div>
            <div class="timeline-items">
                <div class="timeline-item">• Monthly progress assessment</div>
                <div class="timeline-item">• Intervention strategy adjustments as needed</div>
                <div class="timeline-item">• End-of-semester comprehensive review</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("This student has a low risk profile. No specific interventions required at this time, but regular academic advising is recommended.")

# Main function
def main():
    # Load model and scaler
    model = load_model()
    scaler = get_scaler()
    
    if not model or not scaler:
        st.error("Failed to load model or scaler. Please check the error messages.")
        return
    
    # Get category mappings
    category_mappings = get_category_mappings()
    
    # Page title and description
    st.title("🎓 Student Dropout Prediction System")
    st.write("""
    This app predicts whether a student is likely to drop out or graduate based on various factors.
    Fill in the form below with student information to get a prediction.
    """)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Prediction Form", "About"])
    
    # State management for intervention plan
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
        st.session_state.prediction_probability = None
        st.session_state.student_data = None
        st.session_state.show_intervention = False
    
    with tab1:
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with st.form("prediction_form"):
            st.subheader("Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                marital_status_options = list(category_mappings["marital_status"].items())
                marital_status = st.selectbox(
                    "Marital Status", 
                    options=[status[0] for status in marital_status_options],
                    format_func=lambda x: category_mappings["marital_status"].get(x, f"Unknown ({x})")
                )
                
                gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                age_at_enrollment = st.slider("Age at Enrollment", 17, 70, 20)
                
                nationality_options = list(category_mappings["nationality"].items())
                nationality = st.selectbox(
                    "Nationality", 
                    options=[nat[0] for nat in nationality_options],
                    format_func=lambda x: category_mappings["nationality"].get(x, f"Unknown ({x})")
                )
                
                international = st.checkbox("International Student")
            
            with col2:
                displaced = st.checkbox("Displaced Student")
                educational_special_needs = st.checkbox("Has Educational Special Needs")
                debtor = st.checkbox("Has Tuition Debt")
                tuition_fees_up_to_date = st.checkbox("Tuition Fees Up to Date")
                scholarship_holder = st.checkbox("Scholarship Holder")
            
            st.subheader("Academic Background")
            col1, col2 = st.columns(2)
            
            with col1:
                previous_qualification_options = list(category_mappings["previous_qualification"].items())
                previous_qualification = st.selectbox(
                    "Previous Qualification Type", 
                    options=[qual[0] for qual in previous_qualification_options],
                    format_func=lambda x: category_mappings["previous_qualification"].get(x, f"Unknown ({x})")
                )
                
                previous_qualification_grade = st.slider("Previous Qualification Grade", 95.0, 200.0, 130.0, 0.5)
                admission_grade = st.slider("Admission Grade", 95.0, 190.0, 125.0, 0.5)
                
                application_mode_options = list(category_mappings["application_mode"].items())
                application_mode = st.selectbox(
                    "Application Mode", 
                    options=[mode[0] for mode in application_mode_options],
                    format_func=lambda x: category_mappings["application_mode"].get(x, f"Unknown ({x})")
                )
                
                application_order = st.slider("Application Preference Order", 1, 6, 1)
                
                course_options = list(category_mappings["course"].items())
                course = st.selectbox(
                    "Course", 
                    options=[c[0] for c in course_options],
                    format_func=lambda x: category_mappings["course"].get(x, f"Unknown ({x})")
                )
                
                daytime_evening_attendance = st.radio("Attendance Time", [0, 1], format_func=lambda x: "Daytime" if x == 0 else "Evening")
            
            with col2:
                mothers_qualification = st.selectbox(
                    "Mother's Qualification", 
                    options=list(category_mappings["education_level"].keys()),
                    format_func=lambda x: category_mappings["education_level"].get(x, f"Unknown ({x})")
                )
                
                fathers_qualification = st.selectbox(
                    "Father's Qualification", 
                    options=list(category_mappings["education_level"].keys()),
                    format_func=lambda x: category_mappings["education_level"].get(x, f"Unknown ({x})")
                )
                
                mothers_occupation = st.selectbox(
                    "Mother's Occupation", 
                    options=list(category_mappings["occupation"].keys()),
                    format_func=lambda x: category_mappings["occupation"].get(x, f"Unknown ({x})")
                )
                
                fathers_occupation = st.selectbox(
                    "Father's Occupation", 
                    options=list(category_mappings["occupation"].keys()),
                    format_func=lambda x: category_mappings["occupation"].get(x, f"Unknown ({x})")
                )
            
            st.subheader("Academic Performance (1st Semester)")
            col1, col2 = st.columns(2)
            
            with col1:
                curricular_units_1st_sem_credited = st.slider("Curricular Units Credited (1st Sem)", 0, 20, 0)
                curricular_units_1st_sem_enrolled = st.slider("Curricular Units Enrolled (1st Sem)", 0, 20, 6)
                curricular_units_1st_sem_evaluations = st.slider("Curricular Units Evaluations (1st Sem)", 0, 25, 6)
            
            with col2:
                curricular_units_1st_sem_approved = st.slider("Curricular Units Approved (1st Sem)", 0, 20, 5)
                curricular_units_1st_sem_grade = st.slider("Average Grade (1st Sem)", 0.0, 20.0, 13.0, 0.1)
                curricular_units_1st_sem_without_evaluations = st.slider("Units Without Evaluations (1st Sem)", 0, 10, 0)
            
            st.subheader("Academic Performance (2nd Semester)")
            col1, col2 = st.columns(2)
            
            with col1:
                curricular_units_2nd_sem_credited = st.slider("Curricular Units Credited (2nd Sem)", 0, 20, 0)
                curricular_units_2nd_sem_enrolled = st.slider("Curricular Units Enrolled (2nd Sem)", 0, 20, 6)
                curricular_units_2nd_sem_evaluations = st.slider("Curricular Units Evaluations (2nd Sem)", 0, 25, 6)
            
            with col2:
                curricular_units_2nd_sem_approved = st.slider("Curricular Units Approved (2nd Sem)", 0, 20, 5)
                curricular_units_2nd_sem_grade = st.slider("Average Grade (2nd Sem)", 0.0, 20.0, 13.0, 0.1)
                curricular_units_2nd_sem_without_evaluations = st.slider("Units Without Evaluations (2nd Sem)", 0, 12, 0)
            
            st.subheader("Economic Context")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                unemployment_rate = st.slider("Unemployment Rate", 7.0, 17.0, 10.8, 0.1)
            
            with col2:
                inflation_rate = st.slider("Inflation Rate", -1.0, 4.0, 1.4, 0.1)
            
            with col3:
                gdp = st.slider("GDP", -5.0, 4.0, 1.0, 0.01)
            
            # Submit button
            submit_button = st.form_submit_button("Predict")
            
            if submit_button:
                # Create features dictionary
                features = {
                    'Marital_status': marital_status,
                    'Application_mode': application_mode,
                    'Application_order': application_order,
                    'Course': course,
                    'Daytime_evening_attendance': daytime_evening_attendance,
                    'Previous_qualification': previous_qualification,
                    'Previous_qualification_grade': previous_qualification_grade,
                    'Nacionality': nationality,
                    'Mothers_qualification': mothers_qualification,
                    'Fathers_qualification': fathers_qualification,
                    'Mothers_occupation': mothers_occupation,
                    'Fathers_occupation': fathers_occupation,
                    'Admission_grade': admission_grade,
                    'Displaced': int(displaced),
                    'Educational_special_needs': int(educational_special_needs),
                    'Debtor': int(debtor),
                    'Tuition_fees_up_to_date': int(tuition_fees_up_to_date),
                    'Gender': gender,
                    'Scholarship_holder': int(scholarship_holder),
                    'Age_at_enrollment': age_at_enrollment,
                    'International': int(international),
                    'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
                    'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
                    'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
                    'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
                    'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
                    'Curricular_units_1st_sem_without_evaluations': curricular_units_1st_sem_without_evaluations,
                    'Curricular_units_2nd_sem_credited': curricular_units_2nd_sem_credited,
                    'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
                    'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
                    'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
                    'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
                    'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_sem_without_evaluations,
                    'Unemployment_rate': unemployment_rate,
                    'Inflation_rate': inflation_rate,
                    'GDP': gdp
                }
                
                # Make prediction
                prediction, probability = predict_dropout(features, model, scaler)
                
                # Convert numpy float32 to Python float to avoid Streamlit error
                probability = [float(p) for p in probability]
                
                # Store results in session state
                st.session_state.prediction_result = int(prediction)
                st.session_state.prediction_probability = probability
                st.session_state.student_data = features
                st.session_state.show_intervention = False
                
                # Display prediction
                st.subheader("Prediction Result")
                
                # Create columns for the prediction display
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 0:
                        st.error("⚠️ High Risk of Dropout")
                        st.write(f"Probability: {probability[0]:.2%}")
                    else:
                        st.success("🎓 Likely to Graduate")
                        st.write(f"Probability: {probability[1]:.2%}")
                
                with col2:
                    # Create a progress bar for the confidence
                    if prediction == 0:
                        st.write("Dropout Probability:")
                        st.progress(probability[0])
                        st.write("Graduation Probability:")
                        st.progress(probability[1])
                    else:
                        st.write("Graduation Probability:")
                        st.progress(probability[1])
                        st.write("Dropout Probability:")
                        st.progress(probability[0])
                
                # Display risk factors if high dropout risk
                if prediction == 0 and probability[0] > 0.7:
                    st.subheader("Potential Risk Factors")
                    risk_factors = []
                    
                    # Check for risk factors based on the input
                    if curricular_units_1st_sem_approved / max(curricular_units_1st_sem_enrolled, 1) < 0.5:
                        risk_factors.append("Low pass rate in 1st semester courses")
                    
                    if curricular_units_2nd_sem_approved / max(curricular_units_2nd_sem_enrolled, 1) < 0.5:
                        risk_factors.append("Low pass rate in 2nd semester courses")
                        
                    if curricular_units_1st_sem_grade < 11:
                        risk_factors.append("Below average grades in 1st semester")
                        
                    if curricular_units_2nd_sem_grade < 11:
                        risk_factors.append("Below average grades in 2nd semester")
                        
                    if debtor == 1:
                        risk_factors.append("Has tuition debt")
                        
                    if tuition_fees_up_to_date == 0:
                        risk_factors.append("Tuition fees not up to date")
                    
                    # Display risk factors as a bulleted list
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("No specific risk factors identified, but the combination of various attributes indicates dropout risk.")
                
                # Display potential interventions if high dropout risk
                if prediction == 0 and probability[0] > 0.6:
                    st.subheader("Recommended Interventions")
                    
                    # Create a container with custom styling for recommendations
                    recommendation_container = st.container()
                    with recommendation_container:
                        st.markdown("""
                        <style>
                        .recommendation-box {
                            background-color: transparent;
                            border-left: 3px solid #27AE60;
                            padding: 12px 18px;
                            margin-bottom: 12px;
                            border-radius: 0;
                        }

                        .recommendation-title {
                            color: #ECF0F1;
                            font-weight: 600;
                            font-size: 17px;
                            margin-bottom: 4px;
                        }

                        .recommendation-desc {
                            color: #BDC3C7;
                            font-size: 14px;
                            line-height: 1.5;
                        }
                        .recommendation-container {
                            background-color: #2C3E50;
                            padding: 20px;
                            border-radius: 8px;
                            margin-bottom: 20px;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        recommendations = [
                            {
                                "title": "Academic Advising",
                                "description": "Schedule a meeting with an academic advisor to discuss course progress and create a structured study plan."
                            },
                            {
                                "title": "Tutoring Services",
                                "description": "Connect with the tutoring center for additional academic support in challenging subjects."
                            },
                            {
                                "title": "Financial Assistance",
                                "description": "Meet with the financial aid office to explore scholarship options, payment plans, or emergency funding."
                            },
                            {
                                "title": "Regular Check-ins",
                                "description": "Establish bi-weekly check-ins with a faculty mentor to track progress and address concerns early."
                            },
                            {
                                "title": "Study Skills Workshop",
                                "description": "Attend workshops on time management, note-taking, and effective study techniques."
                            }
                        ]
                        
                        for rec in recommendations:
                            st.markdown(f"""
                            <div class="recommendation-box">
                                <div class="recommendation-title">{rec["title"]}</div>
                                <div class="recommendation-desc">{rec["description"]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
        
        # Generate Intervention Plan button with functionality
        if st.session_state.prediction_result is not None:
            intervention_button = st.button(
                "Generate Intervention Plan", 
                help="Generate a detailed intervention plan for this student",
                key="intervention_button"
            )
            
            if intervention_button:
                st.session_state.show_intervention = True
            
            # Display intervention plan if button was clicked
            if st.session_state.show_intervention:
                # For dropout prediction, use dropout probability; for graduation, use (1-graduation probability)
                if st.session_state.prediction_result == 0:
                    dropout_prob = st.session_state.prediction_probability[0]
                else:
                    dropout_prob = st.session_state.prediction_probability[0]  # Already the dropout probability
                
                generate_intervention_plan(
                    st.session_state.student_data, 
                    st.session_state.prediction_result,
                    dropout_prob
                )
        
        year = pd.to_datetime("today").year
        name = "[Moh. Wahyu Abrory](http://linkedin.com/in/wahyuabrory 'Moh. Wahyu Abrory | LinkedIn')"
        copyright = 'Copyright © ' + str(year) + ' ' + name
        st.caption(copyright)

    
    with tab2:
        st.subheader("About this Prediction System")
        st.write("""
        ### Model Information
        This prediction system uses an XGBoost machine learning model trained on historical student data 
        to predict whether a student is likely to drop out or graduate. The model analyzes various factors 
        including personal information, academic performance, and socioeconomic indicators.
        
        ### Features Used
        - **Personal Information**: Age, gender, marital status, nationality
        - **Academic Background**: Previous qualification, admission grades
        - **Family Background**: Parents' education and occupation
        - **Financial Factors**: Scholarship status, tuition payment status
        - **Academic Performance**: Course units, grades for multiple semesters
        - **Economic Context**: Unemployment rate, inflation rate, GDP
        
        ### Interpretation
        The model provides a probability score indicating how likely a student is to drop out or graduate.
        Higher dropout probability suggests greater risk that requires intervention.
        
        ### Disclaimer
        This tool is meant to be used as a screening mechanism to identify students who might benefit from 
        additional support. It should not be used as the sole decision-making factor for academic interventions.
        """)
        
        st.subheader("Dataset")
        st.write("""
        The model was trained using a dataset containing anonymized student records with various 
        attributes and their eventual academic outcomes (dropout or graduation).
        """)

if __name__ == "__main__":
    main()