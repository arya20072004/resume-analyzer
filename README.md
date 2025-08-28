# 🤖 Resume Ranker AI

**Resume Ranker AI** is an AI-powered resume screening and ranking tool that analyzes multiple resumes against a given job description. It extracts key skills, estimates years of experience, and ranks candidates using **NLP techniques** and **machine learning**.  

---

## 🚀 Key Features

- **Resume Analysis**: Upload multiple resumes in PDF format for automated screening.  
- **Job Description Parsing**: Supports JD upload in PDF or TXT formats.  
- **Ranking Algorithm**:
  - Uses **TF-IDF Vectorization** and **Cosine Similarity** to score resumes.  
- **Skill Extraction**:
  - Detects predefined or user-specified keywords in resumes.  
- **Experience Estimation**:
  - Estimates years of experience per skill using text pattern recognition.  
- **Exportable Results**:
  - Download ranked results in CSV format.  
- **Interactive Streamlit UI**:
  - Easy-to-use web interface for recruiters and hiring managers.  

---

## 📂 Project Structure

```
.
├── README.md                # Project documentation
├── requirements.txt         # Required Python packages
├── main.py                  # Main Streamlit application
└── data/                    # (Optional) Folder for sample resumes/JDs
```

---

## 🤖 Core Functionalities

| Function                          | Description                                                        |
|----------------------------------|--------------------------------------------------------------------|
| `extract_text_from_pdf()`        | Extracts raw text from uploaded PDF resumes.                      |
| `preprocess_text()`              | Cleans and tokenizes text for NLP analysis.                       |
| `calculate_similarity()`         | Computes similarity between JD and resumes using TF-IDF & cosine. |
| `extract_skills_from_text()`     | Detects predefined/custom skills in text.                         |
| `estimate_years_of_experience()` | Finds and estimates experience years for each detected skill.     |
| `analyze_keyword_context()`      | Highlights sentences containing relevant keywords.                |

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder-name>
   ```

2. **Create a virtual environment (Optional):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♀️ Usage

Run the following command to start the Streamlit application:

```bash
streamlit run main.py
```

Once the app is running, open the local URL provided by Streamlit (e.g., `http://localhost:8501`) to access the tool.  

**Steps:**
1. Upload a **Job Description** (PDF or TXT).  
2. Upload one or more **Resumes** (PDF only).  
3. Optionally add custom keywords in the sidebar.  
4. Click **Rank Resumes** to generate rankings, scores, and skill analysis.  
5. Download the results as a CSV file.  

---

## 📌 Notes

- Supports **PDF resumes** only (TXT support can be added).  
- Skills are customizable through the sidebar input.  
- This project is designed for **HR automation, recruitment optimization**, and **data science portfolio demonstration**.  

---

## 📂 requirements.txt

```txt
streamlit
pandas
scikit-learn
PyPDF2
nltk
wordcloud
matplotlib
```

---

## ✍️ Authors & Contributors

| Name              | GitHub Profile                         |
|-------------------|----------------------------------------|
| **Arya Borhade**  | [@arya202004]((https://github.com/arya20072004))  |

---
