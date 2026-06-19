import streamlit as st
import joblib
import re
from PyPDF2 import PdfReader
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import docx
import io

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    model = joblib.load("resume_classifier.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, tfidf, label_encoder

try:
    model, tfidf, label_encoder = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="✨",
    layout="wide"
)

# ==========================
# Adaptive Premium Theme (Safe CSS)
# ==========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Safely apply font to the main app container only, avoiding the top menu */
    .block-container {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Title Gradient */
    h1 {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
        padding-bottom: 10px;
    }

    /* Beautiful Text Areas */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1.5px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }

    /* Premium Buttons - Primary Only */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }
    button[data-testid="baseButton-primary"] p {
        color: white !important;
        font-size: 1.1rem;
    }

    /* KPI Metric Cards */
    div[data-testid="metric-container"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] label {
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }
    div[data-testid="metric-container"] div {
        font-weight: 800;
        font-size: 2rem;
    }

    /* Badges */
    .success-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 6px 16px;
        border-radius: 20px;
        margin: 5px;
        display: inline-block;
        border: 1px solid rgba(16, 185, 129, 0.3);
        font-weight: 600;
    }
    .error-badge {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 6px 16px;
        border-radius: 20px;
        margin: 5px;
        display: inline-block;
        border: 1px solid rgba(239, 68, 68, 0.3);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# Helper Functions
# ==========================
def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

import pandas as pd
import os

def load_skills_db():
    if os.path.exists("skills.csv"):
        df = pd.read_csv("skills.csv")
        return df['Skill'].tolist()
    return ["Python", "Java", "SQL", "Machine Learning"] # Fallback

def extract_skills(text, skills_db):
    found = []
    text_lower = text.lower()
    for skill in skills_db:
        # Use regex word boundaries `(?<!\w)` and `(?!\w)` to safely match skills like C++ and C#
        pattern = r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found

# ==========================
# Header
# ==========================
st.title("✨ AI Resume Screening & Recruitment Assistant")
st.markdown("##### Automate your hiring pipeline with ML-powered candidate analysis, job matching, and skill gap identification.")
st.divider()

# ==========================
# Inputs Layout
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"], label_visibility="collapsed")
    
    extracted_text = ""
    if uploaded_file:
        if uploaded_file.name.endswith('.pdf'):
            pdf = PdfReader(uploaded_file)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"

    resume_text = st.text_area(
        "Resume Text (Edit text before analysis if needed)",
        value=extracted_text.strip(),
        height=250
    )

with col2:
    st.subheader("🎯 Job Requirements")
    st.write("") # Spacer
    job_description = st.text_area(
        "Paste Job Description",
        height=325,
        placeholder="Enter the job description, required skills, and responsibilities here..."
    )

# ==========================
# Button & Analysis
# ==========================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Analyze Candidates", type="primary", use_container_width=True):

    if resume_text.strip() == "":
        st.warning("Please enter or upload resume text.")
    else:
        with st.spinner("Running Analysis..."):
            # ==========================
            # Backend Analytics (Background Calculation Only)
            # ==========================
            resume_word_count = len(resume_text.split())

            # ==========================
            # Resume Classification
            # ==========================
            cleaned_resume = clean_resume(resume_text)
            resume_vector = tfidf.transform([cleaned_resume])
            prediction = model.predict(resume_vector)
            predicted_category = label_encoder.inverse_transform(prediction)[0]

            # ==========================
            # Dynamic Skill Extraction
            # ==========================
            skills_db = load_skills_db()
            found_skills = extract_skills(resume_text, skills_db)

            # ==========================
            # Job Matching 
            # ==========================
            similarity_score = 0
            skill_score = 0
            rating = "No Rating"
            missing_skills = []
            job_skills = []

            if job_description.strip() != "":
                cleaned_job = clean_resume(job_description)
                job_vector = tfidf.transform([cleaned_job])
                
                # TF-IDF text similarity
                similarity_score = min((cosine_similarity(resume_vector, job_vector)[0][0]) * 400, 100.0)

                # Dynamically extract REQUIRED skills from the Job Description
                job_skills = extract_skills(job_description, skills_db)
                
                # Improved Skill Match Logic (Matched / Required * 100)
                if len(job_skills) > 0:
                    matched_skills = set(found_skills) & set(job_skills)
                    skill_score = (len(matched_skills) / len(job_skills)) * 100
                else:
                    skill_score = 100.0 if found_skills else 0.0
                    
                missing_skills = list(set(job_skills) - set(found_skills))

            # Improved ATS Scoring Formula (30% Text / 70% Skill)
            overall_fit = (similarity_score * 0.3) + (skill_score * 0.7)

            # Candidate Rating Logic based directly on Overall ATS Score
            if overall_fit >= 90:
                rating = "Outstanding Candidate"
            elif overall_fit >= 75:
                rating = "Excellent Candidate"
            elif overall_fit >= 60:
                rating = "Good Candidate"
            elif overall_fit >= 40:
                rating = "Average Candidate"
            else:
                rating = "Needs Improvement"

            # Compute extra analytics variables
            total_found_skills = len(found_skills)
            total_required_skills = len(job_skills)
            total_missing_skills = len(missing_skills)
            completeness_score = min((total_found_skills / max(total_required_skills, 1)) * 100, 100)

            # ==========================
            # Results Dashboard
            # ==========================
            st.divider()
            st.markdown("<h2>📊 Analysis Dashboard</h2>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Top KPI Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted Category", predicted_category)
            c2.metric("Overall Fit Score", f"{overall_fit:.1f}%")
            c3.metric("TF-IDF Job Match", f"{similarity_score:.2f}%")
            c4.metric("Skill Match Score", f"{skill_score:.0f}%")

            # Match Analysis Charts
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3>📈 Fit Breakdown</h3>", unsafe_allow_html=True)
            col_gauge, col_prog = st.columns([1, 1.2])
            
            with col_gauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=overall_fit,
                    title={"text": "Overall Candidate Fit"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "rgba(0,0,0,0)"},
                        "borderwidth": 2,
                        "bordercolor": "gray",
                        "steps": [
                            {"range": [0, 40], "color": "#ef4444"},
                            {"range": [40, 70], "color": "#f59e0b"},
                            {"range": [70, 100], "color": "#10b981"}
                        ],
                        "threshold": {
                            "line": {"color": "#6366f1", "width": 5},
                            "thickness": 0.85,
                            "value": overall_fit
                        }
                    }
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                
            with col_prog:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.write(f"**TF-IDF Text Similarity ({similarity_score:.2f}%)**")
                st.progress(min(similarity_score / 100, 1.0))
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Hard Skills Match ({skill_score:.0f}%)**")
                st.progress(min(skill_score / 100, 1.0))
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(f"**Recommendation:**")
                
                if similarity_score >= 60 or overall_fit >= 75:
                    st.success(f"🌟 **{rating}** - Highly recommended for interview.")
                elif similarity_score >= 40 or overall_fit >= 50:
                    st.warning(f"⚠️ **{rating}** - Consider reviewing experience level or missing skills.")
                else:
                    st.error(f"❌ **{rating}** - Does not meet minimum requirements.")

            # Skills Analysis
            st.divider()
            st.markdown("<h3>🛠️ Skills Analysis</h3>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            col6, col7 = st.columns(2)

            with col6:
                st.markdown("#### ✅ Skills Found in Resume")
                if found_skills:
                    skills_html = "".join([f"<span class='success-badge'>✓ {s.title()}</span>" for s in found_skills])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.info("No relevant skills detected.")

            with col7:
                st.markdown("#### ❌ Missing Requirements")
                if missing_skills:
                    skills_html = "".join([f"<span class='error-badge'>✗ {s.title()}</span>" for s in missing_skills])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span class='success-badge'>🎉 Candidate possesses all required skills!</span>", unsafe_allow_html=True)