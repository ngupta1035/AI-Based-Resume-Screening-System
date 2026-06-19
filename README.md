# 📄 AI-Powered Resume Screening & Recruitment Assistant

## Overview

The AI-Powered Resume Screening & Recruitment Assistant is a Machine Learning and Natural Language Processing (NLP) based application designed to automate the resume screening process for recruiters and hiring teams.

The system analyzes candidate resumes, predicts the candidate's professional domain, evaluates compatibility with a given job description, extracts relevant skills, identifies missing skills, and generates an overall ATS-style fit score.

This project aims to reduce manual screening efforts and improve hiring efficiency by providing intelligent candidate evaluation and recommendation capabilities.

---

## Features

### Resume Classification

* Automatically predicts the candidate's professional category.
* Uses Machine Learning classification models trained on real-world resume datasets.
* Supports categories such as:

  * Information Technology
  * Finance
  * HR
  * Healthcare
  * Engineering
  * Sales
  * Business Development
  * Banking
  * Consultant
  * Designer
  * And more.

### Job Description Matching

* Compares resume content with job requirements.
* Uses TF-IDF Vectorization and Cosine Similarity.
* Generates a Job Match Score indicating textual relevance between the resume and job description.

### Dynamic Skill Extraction

* Automatically extracts technical and professional skills from resumes and job descriptions.
* Works across multiple domains.
* Detects programming, cloud, analytics, AI, database, and business skills.

### Skill Gap Analysis

* Identifies:

  * Skills Found
  * Required Skills
  * Missing Skills
* Helps recruiters quickly understand candidate suitability.

### ATS Compatibility Score

* Combines:

  * Text Similarity Score
  * Skill Match Score
* Produces a final ATS-style candidate fit percentage.

### Candidate Recommendation Engine

Provides intelligent hiring recommendations:

* Outstanding Candidate
* Excellent Candidate
* Good Candidate
* Average Candidate
* Needs Improvement

### Resume Upload Support

Supports:

* PDF Resumes
* DOCX Resumes
* Manual Resume Text Input

### Interactive Dashboard

Built using Streamlit with:

* KPI Cards
* Skill Analysis
* Match Score Visualization
* Gauge Charts
* Candidate Insights

---

## Machine Learning Workflow

### Step 1: Data Collection

Resume dataset containing thousands of categorized resumes.

### Step 2: Text Preprocessing

* Lowercasing
* Special character removal
* URL removal
* Whitespace normalization

### Step 3: Feature Engineering

TF-IDF Vectorization converts textual resume data into numerical vectors.

### Step 4: Resume Classification

Machine Learning model predicts candidate category.

### Step 5: Job Matching

Cosine Similarity measures the similarity between:

* Resume
* Job Description

### Step 6: Skill Extraction

Required skills are extracted and compared.

### Step 7: ATS Score Generation

Overall candidate suitability is calculated using weighted scoring.

---

## Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-Learn

### Data Processing

* Pandas
* NumPy

### NLP

* TF-IDF Vectorization
* Cosine Similarity

### Frontend

* Streamlit

### Visualization

* Plotly

### File Processing

* PyPDF2
* python-docx

### Model Serialization

* Joblib

---

## Project Structure

```text
AI-Resume-Screening-System/
│
├── app.py
├── Resume_Screening.ipynb
│
├── resume_classifier.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
│
├── skills.csv
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── dashboard.png
    ├── analysis.png
    └── results.png
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Resume-Screening-System.git

cd AI-Resume-Screening-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Example Workflow

### Input

Resume:

* Python
* SQL
* Machine Learning
* TensorFlow

Job Description:

* Python
* SQL
* AWS
* Docker
* Machine Learning

### Output

Predicted Category:

```text
INFORMATION TECHNOLOGY
```

Job Match Score:

```text
82%
```

Skill Match Score:

```text
75%
```

Missing Skills:

```text
AWS
Docker
```

Candidate Recommendation:

```text
Excellent Candidate
```

---

## Future Enhancements

* Multi-Resume Ranking
* Resume Comparison Dashboard
* Recruiter Analytics Panel
* PDF Report Generation
* Interview Recommendation Engine
* Skill Recommendation System
* Cloud Deployment
* Advanced NLP-based Skill Extraction
* Generative AI Resume Insights

---

## Applications

* Recruitment Agencies
* HR Departments
* Campus Placements
* Talent Acquisition Teams
* Job Portals
* Startup Hiring Platforms

---

## Results

The system successfully:

* Classifies resumes into professional categories.
* Matches resumes against job descriptions.
* Extracts candidate skills automatically.
* Identifies missing requirements.
* Generates ATS-style candidate evaluation scores.

This helps recruiters reduce manual screening effort and improve hiring decision-making.

---

## Author

Narayani Gupta

Bachelor of Engineering – Artificial Intelligence & Data Science

Machine Learning | Data Science | NLP | Analytics
