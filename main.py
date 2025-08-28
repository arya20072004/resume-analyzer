# main_app.py

import streamlit as st
import pandas as pd
import PyPDF2
import io
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import collections

# --- Pre-computation and Setup ---
# We use a function to avoid re-downloading
def download_nltk_data():
    """Downloads the necessary NLTK models if they are not already present."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Call the function to ensure data is downloaded before the app runs
download_nltk_data()

# --- Core Functions ---

def simple_sent_tokenize(text):
    """
    A simple regex-based sentence tokenizer to avoid NLTK's 'punkt' dependency.
    Splits text based on periods, question marks, and exclamation marks followed by a space.
    """
    return re.split(r'(?<=[.?!])\s+', text)

def extract_text_from_pdf(file):
    """
    Extracts text from an uploaded PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {file.name}. Error: {e}")
        return None

def preprocess_text(text):
    """
    Cleans and preprocesses the input text for similarity comparison.
    """
    if not text:
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower().strip()
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return " ".join(lemmatized_tokens)

def calculate_similarity(resume_text, job_desc_text):
    """
    Calculates the cosine similarity between a resume and a job description.
    """
    if not resume_text or not job_desc_text:
        return 0.0
    text_corpus = [resume_text, job_desc_text]
    tfidf_vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_corpus)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        return 0.0

def extract_skills_from_text(text, skills_list):
    """
    Extracts skills from text by matching against a user-defined list.
    """
    found_skills = set()
    processed_text = text.lower()
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', processed_text):
            found_skills.add(skill.strip().title())
    return list(found_skills)

def analyze_keyword_context(resume_text, keywords):
    """
    Finds sentences in the resume that contain the specified keywords.
    """
    context = collections.defaultdict(list)
    sentences = simple_sent_tokenize(resume_text)
    for keyword in keywords:
        for sentence in sentences:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', sentence.lower()):
                context[keyword.title()].append(sentence.strip())
    return dict(context)

def estimate_years_of_experience(resume_text, keywords):
    """
    Estimates years of experience for each keyword by looking for patterns
    like '5 years', '3+ years', '18 months' etc., near the keyword.
    """
    experience = {}
    window = 100  # Search within a 100-character window around the keyword

    for keyword in keywords:
        # Find all occurrences of the keyword
        for match in re.finditer(r'\b' + re.escape(keyword.lower()) + r'\b', resume_text.lower()):
            start, end = match.span()
            # Define the search window around the keyword
            search_area = resume_text[max(0, start - window):min(len(resume_text), end + window)]
            
            # Regex to find numbers (digits or words) followed by 'year' or 'month'
            exp_match = re.search(
                r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[\s\+]+(year|month)s?',
                search_area,
                re.IGNORECASE
            )
            
            if exp_match:
                number_str = exp_match.group(1)
                time_unit = exp_match.group(2).lower()
                
                word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
                               'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
                
                try:
                    number = int(number_str)
                except ValueError:
                    number = word_to_num.get(number_str.lower(), 0)

                if 'month' in time_unit:
                    number /= 12.0
                
                # Store the highest found value for each skill
                if keyword.title() not in experience or number > float(experience[keyword.title()].split()[0]):
                     experience[keyword.title()] = f"{number:.1f} years"

    return experience

# --- Streamlit UI ---

st.set_page_config(page_title="Resume Ranker AI", layout="wide", page_icon="✨")

# --- Custom CSS for a Modern Dark UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; color: #e0e0e0; }
    .stApp { background-color: #121212; }
    h1 { color: #ffffff; font-weight: 700; text-shadow: 2px 2px 4px #00000050; }
    [data-testid="stSidebar"] { background-color: #1e1e1e; border-right: 1px solid #333; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #ffffff; }
    .stButton>button {
        background-image: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white; border-radius: 8px; border: none; padding: 12px 28px;
        font-size: 16px; font-weight: 600; width: 100%; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3); }
    .stButton>button:focus { box-shadow: 0 0 0 3px #2575fc80; }
    .stButton>button[kind="secondary"] { background-image: none; background-color: #444; }
    .stButton>button[kind="secondary"]:hover { background-color: #555; }
    .results-card {
        background-color: #1e1e1e; border-radius: 12px; padding: 25px; margin-bottom: 20px;
        border: 1px solid #333; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .results-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.4); }
    .card-header { display: flex; align-items: center; margin-bottom: 15px; }
    .card-header h3 { margin: 0; font-size: 1.5rem; color: #ffffff; }
    .card-header .rank {
        background-image: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white; border-radius: 50%; width: 40px; height: 40px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; font-weight: bold; margin-right: 15px; flex-shrink: 0;
    }
    .score-bar-container { width: 100%; background-color: #333; border-radius: 5px; height: 20px; margin-top: 10px; overflow: hidden; }
    .score-bar { height: 100%; border-radius: 5px; text-align: right; color: white; padding-right: 8px; line-height: 20px; font-weight: bold; transition: width 0.5s ease-in-out; }
    .stExpander { border: 1px solid #333; border-radius: 8px; background-color: #2a2a2a; }
    .stTextArea textarea { background-color: #2a2a2a; color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# --- App State Management ---
if 'results' not in st.session_state:
    st.session_state.results = None

def clear_session():
    st.session_state.results = None

# --- Header ---
st.title("✨ Resume Ranker AI")
st.markdown("### The intelligent way to screen and rank candidates.")

# --- Sidebar for Uploads ---
with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown("Upload a job description and resumes to begin.")
    job_description_file = st.file_uploader("1. Upload Job Description", type=["pdf", "txt"])
    uploaded_resumes = st.file_uploader("2. Upload Resumes", type="pdf", accept_multiple_files=True)
    
    st.header("🔧 Customize Keywords")
    default_skills = ('python, java, c++, javascript, sql, git, docker, kubernetes, '
                      'aws, azure, gcp, react, angular, vue, node.js, django, flask, '
                      'machine learning, deep learning, tensorflow, pytorch, scikit-learn, '
                      'pandas, numpy, data analysis, natural language processing, nlp, '
                      'computer vision, agile, scrum, project management')
    custom_skills_text = st.text_area("Keywords to analyze (comma-separated):", value=default_skills, height=150)
    custom_skills_list = [skill.strip() for skill in custom_skills_text.split(',') if skill.strip()]

    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("Rank Resumes")
    with col2:
        clear_button = st.button("Clear All", on_click=clear_session, type="secondary")

# --- Main Content Area ---
job_desc_text = ""
if job_description_file:
    if job_description_file.type == "application/pdf":
        job_desc_text = extract_text_from_pdf(job_description_file)
    else:
        job_desc_text = job_description_file.read().decode("utf-8")
    
    with st.expander("📄 View Job Description", expanded=False):
        st.text_area("", value=job_desc_text, height=200, disabled=True)

if process_button and job_desc_text and uploaded_resumes:
    with st.spinner("Analyzing and ranking resumes..."):
        preprocessed_jd = preprocess_text(job_desc_text)
        results = []
        for resume in uploaded_resumes:
            resume_text = extract_text_from_pdf(resume)
            if resume_text:
                preprocessed_resume = preprocess_text(resume_text)
                similarity_score = calculate_similarity(preprocessed_resume, preprocessed_jd)
                skills = extract_skills_from_text(resume_text, custom_skills_list)
                keyword_context = analyze_keyword_context(resume_text, skills)
                estimated_experience = estimate_years_of_experience(resume_text, skills)
                
                results.append({
                    "Filename": resume.name,
                    "Similarity Score": round(similarity_score * 100, 2),
                    "Skills": ", ".join(skills) if skills else "No matching keywords found",
                    "Resume Text": resume_text,
                    "Context": keyword_context,
                    "Experience": estimated_experience
                })
        
        if results:
            ranked_df = pd.DataFrame(results)
            ranked_df = ranked_df.sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)
            st.session_state.results = ranked_df

if st.session_state.results is not None:
    ranked_df = st.session_state.results
    st.header("🏆 Ranked Candidates")
    st.markdown(f"Showing the top **{len(ranked_df)}** candidates based on their match score.")

    for index, row in ranked_df.iterrows():
        score = row['Similarity Score']
        bar_color = ("linear-gradient(90deg, #1d976c 0%, #93f9b9 100%)" if score >= 75 else
                     "linear-gradient(90deg, #ff8008 0%, #ffc837 100%)" if score >= 50 else
                     "linear-gradient(90deg, #c31432 0%, #240b36 100%)")

        st.markdown(f"""
        <div class="results-card">
            <div class="card-header">
                <div class="rank">{index + 1}</div>
                <h3>{row['Filename']}</h3>
            </div>
            <p><strong>Matching Keywords:</strong> {row['Skills']}</p>
            <strong>Match Score: {score}%</strong>
            <div class="score-bar-container">
                <div class="score-bar" style="width: {score}%; background-image: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for the expanders
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.expander("📜 View Full Resume Text"):
                st.text_area("", value=row['Resume Text'], height=300, disabled=True, key=f"resume_{index}")
        with col2:
            with st.expander("🔍 View Keyword Context"):
                context_data = row.get('Context', {})
                if not context_data:
                    st.info("No context found for the specified keywords.")
                else:
                    for keyword, sentences in context_data.items():
                        st.markdown(f"**{keyword}**")
                        for sentence in sentences:
                            st.markdown(f"- *{sentence}*")
        with col3:
            with st.expander("📈 View Estimated Experience"):
                # --- FIX IS HERE ---
                # Use .get() for safe access to prevent KeyError
                experience_data = row.get('Experience', {})
                if not experience_data:
                    st.info("No specific years of experience found for keywords.")
                else:
                    for skill, exp in experience_data.items():
                        st.markdown(f"- **{skill}:** {exp}")


    st.header("📄 Export Results")
    export_df = ranked_df.copy()
    # Flatten the experience dictionary for better CSV export
    experience_df = export_df['Experience'].apply(pd.Series)
    export_df = pd.concat([export_df.drop(columns=['Resume Text', 'Context', 'Experience']), experience_df], axis=1)
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download as CSV", data=csv, file_name='ranked_resumes_with_experience.csv', mime='text/csv')

elif process_button:
    st.warning("⚠️ Please upload a job description and at least one resume to proceed.")
else:
    st.info("Upload your files and customize keywords in the sidebar, then click 'Rank Resumes' to begin.")
