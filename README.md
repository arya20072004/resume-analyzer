✨ Resume Ranker AI
An intelligent Streamlit application designed to screen, analyze, and rank candidate resumes against a given job description. This tool helps recruiters and hiring managers efficiently identify the most qualified candidates by leveraging NLP for similarity scoring, keyword extraction, and experience estimation.

(Suggestion: Replace the placeholder above with a screenshot of your running application!)

🚀 Features
Similarity Scoring: Calculates a cosine similarity score between each resume and the job description to quantify the match.

Custom Keyword Analysis: Extracts a user-defined list of skills and keywords from resumes.

Experience Estimation: Intelligently estimates the years of experience for each specified keyword (e.g., "Python: 5.0 years").

Contact Information Extraction: Pulls email addresses and phone numbers from resumes.

Interactive UI: A modern, dark-themed, and user-friendly interface built with Streamlit.

Detailed Views: Provides expandable sections to view the full resume text, keyword context, and a resume word cloud.

Data Export: Allows you to export the ranked list of candidates to a CSV file.

🛠️ Installation & Setup
To run this application locally, follow these steps:

Clone the repository:

git clone https://github.com/YOUR_USERNAME/resume-ranker-ai.git
cd resume-ranker-ai

Create and activate a virtual environment (recommended):

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

▶️ How to Run
Once the setup is complete, you can run the Streamlit application with the following command:

streamlit run main_app.py

Your web browser should automatically open to the application's URL.

📋 How to Use
Upload Job Description: Use the sidebar to upload a PDF or TXT file containing the job description.

Upload Resumes: Upload one or more candidate resumes in PDF format.

Customize Keywords (Optional): Modify the comma-separated list of keywords you want to analyze.

Rank Resumes: Click the "Rank Resumes" button to start the analysis.

View Results: The main area will display a ranked list of candidates. You can expand each candidate's card to view more details.

Export Data: Scroll to the bottom to download the results as a CSV file.
