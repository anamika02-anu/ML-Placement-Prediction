import streamlit as st
import pickle
from PIL import Image
import math

# ------------------------------- 
# 🎨 Page Config
# ------------------------------- 
st.set_page_config(page_title="Placement App", page_icon="🎓", layout="wide")

# ------------------------------- 
# Initialize session state for inputs
# ------------------------------- 
if 'clear_inputs' not in st.session_state:
    st.session_state.clear_inputs = False

# ------------------------------- 
# 📌 Sidebar Navigation
# ------------------------------- 
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("🏠 Home", "🔍 Placement Prediction", "📘 Dataset Information")
)

# ------------------------------- 
# 🏠 HOME PAGE
# ------------------------------- 
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#4B0082;'>🏠 Welcome to Placement Prediction System</h1>", 
                unsafe_allow_html=True)
    
    img = Image.open("working_prof.avif")
    st.image(img, width=900)
    
    st.markdown("""
    ### 🌟 What This App Does
    - Predicts whether a student will be placed
    - Uses Machine Learning
    - Displays dataset information
    """)
    
    st.markdown("---")
    st.info("➡ Select *Prediction* from the sidebar to test the ML model.")

# ------------------------------- 
# 🔍 PLACEMENT PREDICTION PAGE
# ------------------------------- 
elif page == "🔍 Placement Prediction":
    st.markdown("<h1 style='text-align:center; color:#4B0082;'>🔍 Placement Prediction</h1>", 
                unsafe_allow_html=True)
    
    # Load ML model and scaler
    model = pickle.load(open("placement_model_svm.sav", "rb"))
    scaler = pickle.load(open("placement_scaler.sav", "rb"))
    
    st.sidebar.header("🧾 Enter Student Details")
    
    # Number inputs with minimum values
    degree_p = st.sidebar.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    internships = st.sidebar.number_input("Internships", min_value=0, value=0, step=1)
    projects = st.sidebar.number_input("Projects", min_value=0, value=0, step=1)
    workshops = st.sidebar.number_input("Certifications", min_value=0, value=0, step=1)
    
    # Number inputs for percentages and scores
    ssc_p = st.sidebar.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    hsc_p = st.sidebar.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    aptitude = st.sidebar.number_input("Aptitude Score", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    
    # Selectboxes with default "- Select -" option
    softskills_options = ["- Select -", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    softskills = st.sidebar.selectbox("Soft Skills (1–10)", softskills_options)
    
    extra_options = ["- Select -", "Yes", "No"]
    extra = st.sidebar.selectbox("Extracurricular", extra_options)
    
    training_options = ["- Select -", "Yes", "No"]
    training = st.sidebar.selectbox("Training Attended", training_options)
    
    # ------------------------------- 
    # 🧹 CLEAR BUTTON
    # ------------------------------- 
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear All Inputs"):
        st.session_state.clear_inputs = True
        st.rerun()
    
    # Reset inputs if clear was clicked
    if st.session_state.clear_inputs:
        st.session_state.clear_inputs = False
        st.rerun()
    
    # ------------------------------- 
    # 🎯 PREDICTION BUTTON
    # ------------------------------- 
    if st.sidebar.button("🔍 Predict Placement"):
        # Validate all fields are filled
        validation_errors = []
        
        if degree_p == 0.0:
            validation_errors.append("Degree Percentage must be greater than 0")
        
        if ssc_p == 0.0:
            validation_errors.append("10th Percentage must be greater than 0")
        
        if hsc_p == 0.0:
            validation_errors.append("12th Percentage must be greater than 0")
        
        if aptitude == 0.0:
            validation_errors.append("Aptitude Score must be greater than 0")
        
        if softskills == "- Select -":
            validation_errors.append("Please select Soft Skills")
        
        if extra == "- Select -":
            validation_errors.append("Please select Extracurricular")
        
        if training == "- Select -":
            validation_errors.append("Please select Training Attended")
        
        # Show validation errors if any
        if validation_errors:
            st.error("⚠ Please complete all required fields:")
            for error in validation_errors:
                st.warning(f"• {error}")
        else:
            # Convert values to numeric
            degree_p_val = float(degree_p)
            internships_val = int(internships)
            projects_val = int(projects)
            workshops_val = int(workshops)
            aptitude_val = float(aptitude)
            softskills_val = float(softskills)
            ssc_p_val = float(ssc_p)
            hsc_p_val = float(hsc_p)
            extra_enc = 1 if extra == "Yes" else 0
            training_enc = 1 if training == "Yes" else 0
            
            features = [[degree_p_val, internships_val, projects_val, workshops_val, 
                        aptitude_val, softskills_val, extra_enc, training_enc, ssc_p_val, hsc_p_val]]
            
            # Make prediction
            scaled_features = scaler.transform(features)
            result = model.predict(scaled_features)
            
            st.markdown("---")
            st.subheader("📄 Student Profile Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🎓 *Degree %:* {degree_p_val}")
                st.info(f"📚 *Internships:* {internships_val}")
                st.info(f"🛠 *Projects:* {projects_val}")
                st.info(f"📜 *Certifications:* {workshops_val}")
                st.info(f"📊 *10th %:* {ssc_p_val}")
            
            with col2:
                st.info(f"🧠 *Aptitude Score:* {aptitude_val}")
                st.info(f"🗣 *Soft Skills:* {softskills_val}/10")
                st.info(f"🎯 *Training:* {training}")
                st.info(f"🎨 *Extracurricular:* {extra}")
                st.info(f"📊 *12th %:* {hsc_p_val}")
            
            st.markdown("---")
            
            # ------------------------------- 
            # ✔ FINAL RESULT
            # ------------------------------- 
            st.subheader("Final Prediction")
            if result[0] == 1:
                st.success("🎉 The student is *LIKELY to be placed!*")
                st.balloons()
                st.info("💡 Tip: Keep improving technical + communication skills. Great potential!")
            else:
                st.error("❌ The student is *NOT likely to be placed.*")
                st.warning("💡 Tip: Focus on projects, aptitude and certifications to improve chances.")
            
            st.markdown("---")

# ------------------------------- 
# 📘 DATASET INFORMATION PAGE
# ------------------------------- 
elif page == "📘 Dataset Information":
    st.markdown("<h1 style='text-align:center; color:#4B0082;'>📘 Dataset Information</h1>", 
                unsafe_allow_html=True)
    
    with st.expander("📊 About the Dataset"):
        st.write("""
        The Placement Dataset contains information about students' academic performance, 
        skills, internships, workshops, and other factors that influence placement.
        
        *Dataset Features*
        - Degree Percentage
        - Internships
        - Projects
        - Soft Skills
        - Aptitude Score
        - Extracurricular Activities
        - Training
        - SSC & HSC Percentage
        
        This data is used to predict whether a student will be placed.
        """)