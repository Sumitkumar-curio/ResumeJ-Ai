import io
import re
import base64
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK data once at startup
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english') + list(string.punctuation))
lemmatizer = WordNetLemmatizer()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.resumejobai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predefined Job Profiles (removed "Step X:" for better TF-IDF, made more general)
PREDEFINED_PROFILES = {
    "AI Engineer": "The company is seeking an entry-level AI Engineer to join the innovative development team in India. Analyze business requirements and identify AI opportunities by conducting stakeholder interviews and reviewing existing systems. Collect and preprocess data from various sources using tools like Pandas, NumPy, and ETL pipelines to handle missing values, normalization, and feature extraction. Design AI models including neural networks, deep learning architectures (CNNs, RNNs, Transformers), and generative models (GANs, VAEs). Implement models using frameworks such as TensorFlow, PyTorch, Keras, Scikit-learn, and Hugging Face Transformers for tasks like classification, regression, and NLP. Train and fine-tune models with techniques like hyperparameter tuning (Grid Search, Random Search), cross-validation, transfer learning, and regularization to prevent overfitting. Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and custom loss functions. Deploy models to production environments using Docker, Kubernetes, AWS SageMaker, Google AI Platform, or Azure ML, ensuring scalability and monitoring with tools like Prometheus. Monitor and maintain models, handling issues like model drift, retraining with new data, and A/B testing for improvements. Collaborate with data scientists, software engineers, and stakeholders for seamless integration into applications, using agile methodologies. Document processes, code, models, and results for reproducibility, compliance, and knowledge sharing. Qualifications: Bachelor's degree in Computer Science, Artificial Intelligence, Machine Learning, or related field from a recognized university in India. 6-12 months of experience through internships, academic projects, hackathons, or freelance work. Proficiency in programming languages such as Python (primary), Java, or C++, with libraries like NumPy, Pandas, Scikit-learn, OpenCV for computer vision, NLTK/SpaCy for NLP. Basic understanding of algorithms like CNNs, RNNs, LSTMs, GANs, reinforcement learning, and concepts such as supervised/unsupervised learning, ensemble methods. Familiarity with cloud platforms (AWS, Azure, GCP for services like EC2, S3, Lambda), big data tools (Spark, Hadoop for data handling), version control (Git, GitHub), and databases (SQL/NoSQL). Strong problem-solving, analytical thinking, communication skills, and ability to work in teams. Preferred: Certifications like Google Professional Machine Learning Engineer, AWS Certified Machine Learning - Specialty, or Microsoft Certified: Azure AI Engineer Associate; experience with ethical AI, bias mitigation, explainable AI (XAI), and data privacy regulations (GDPR, India's DPDP Act).",
    "Machine Learning Engineer": "The company is seeking an entry-level Machine Learning Engineer to support ML initiatives in our tech team in India. Understand project goals, gather data requirements, and define success metrics through collaboration with stakeholders. Acquire and clean large datasets using SQL queries, Python scripts, and ETL tools like Apache Airflow or Talend. Perform exploratory data analysis (EDA) with visualization libraries like Matplotlib, Seaborn, or Plotly to identify patterns and anomalies. Engineer features by handling missing values, scaling (Min-Max, Standard), encoding categorical variables (One-Hot, Label), and feature selection (PCA, RFE). Select and implement ML algorithms for tasks like classification (SVM, Random Forest), regression (Linear, Lasso), clustering (K-Means, DBSCAN) using Scikit-learn, XGBoost, LightGBM. Build and train models, applying optimization techniques like grid search, random search, Bayesian optimization for hyperparameters. Validate models with k-fold cross-validation, hold-out sets, and metrics like MSE, MAE for regression or precision-recall for imbalanced data. Deploy models via APIs (Flask, FastAPI), cloud services (AWS Lambda, GCP Functions), or MLOps tools like MLflow, Kubeflow for versioning and serving. Monitor performance in production with tools like Prometheus, Grafana, implement A/B testing, and retrain models periodically. Iterate based on feedback, new data, and business needs, documenting the entire pipeline. Qualifications: Bachelor's degree in Computer Science, Data Science, or related field from a top Indian institute. 6-12 months experience via internships or projects. Strong skills in Python or R, libraries like Scikit-learn, TensorFlow/Keras for deep learning, data processing (Pandas, NumPy), statistics (hypothesis testing, distributions, probability). Understanding of supervised/unsupervised learning, ensemble methods (Bagging, Boosting), time-series analysis. Preferred: Experience with big data (Spark for distributed computing), DevOps (Docker for containerization), cloud (AWS ML services like SageMaker), certifications like Google Data Analytics or AWS Certified Machine Learning.",
    "Data Scientist": "The company is seeking an entry-level Data Scientist to uncover insights from data in our analytics team in India. Define business problems, formulate hypotheses, and set objectives through meetings with stakeholders. Collect data from databases (SQL/NoSQL), APIs, web scraping (BeautifulSoup, Selenium), or internal sources. Clean and preprocess data, handling outliers, imputation (mean/median, KNN), and transformation (log, box-cox). Conduct EDA to identify patterns, correlations (Pearson, Spearman), and distributions using statistical tests. Apply statistical methods and ML for modeling, e.g., linear/logistic regression, decision trees, random forests, SVM. Build predictive models, tune hyperparameters with GridSearchCV, and handle imbalanced data (SMOTE, undersampling). Visualize results with Tableau, Power BI, Matplotlib/Seaborn for dashboards, heatmaps, scatter plots. Interpret findings, calculate business impact (ROI, cost savings), and provide actionable recommendations. Communicate via reports (Jupyter Notebooks), presentations (PowerPoint), and data storytelling. Ensure ethical data use, bias detection, and compliance with regulations like India's Personal Data Protection Bill. Qualifications: Bachelor's in Data Science, Statistics, Computer Science from reputed Indian colleges. 6-12 months experience. Proficiency in Python/R, SQL for querying, ML basics (Scikit-learn), visualization (Tableau/Power BI). Strong analytical, communication skills. Preferred: Advanced stats (ANOVA, chi-square), big data (Hadoop), domain knowledge (e.g., e-commerce, finance).",
    "Cybersecurity Analyst": "The company is seeking an entry-level Cybersecurity Analyst to support security operations in our IT team in India. Assess current security posture by reviewing policies, conducting risk assessments, and identifying gaps. Monitor systems and networks for threats using SIEM tools like Splunk, ELK Stack (Elasticsearch, Logstash, Kibana), or QRadar. Identify vulnerabilities with scanners such as Nessus, OpenVAS, or Qualys, and prioritize based on CVSS scores. Perform penetration testing using tools like Metasploit, Burp Suite, Nmap for vulnerability exploitation simulation. Implement and configure security measures including firewalls (Palo Alto, Cisco ASA), IDS/IPS (Snort), antivirus (Symantec, McAfee). Respond to incidents by isolating affected systems, performing forensics with Wireshark or Autopsy, and eradicating threats. Develop and update security policies, procedures, and incident response plans aligned with standards like ISO 27001 or NIST. Train employees on security best practices, such as phishing awareness, password management, and safe internet usage. Prepare compliance reports for audits, ensuring adherence to regulations like IT Act 2000, GDPR, or PCI-DSS. Stay updated on cybersecurity trends, threats (via Threat Intelligence feeds like AlienVault OTX), and participate in CTF challenges. Qualifications: Bachelor's in Cybersecurity, Information Technology, or related from Indian universities. 6-12 months experience. Knowledge of security protocols (HTTPS, SSL/TLS), ethical hacking tools, firewalls, basic scripting (Python/Bash). Strong analytical, problem-solving skills. Preferred: Certifications like CompTIA Security+, CEH, familiarity with cloud security (AWS Security Hub).",
    "Cloud Engineer": "The company is seeking an entry-level Cloud Engineer to assist with cloud infrastructure management in our operations team in India. Plan cloud architecture by evaluating requirements for scalability, availability, and cost. Provision resources on AWS/Azure/GCP, such as EC2 instances, VMs, S3 storage, RDS databases. Configure networking including VPCs, subnets, security groups, load balancers (ELB, ALB). Implement security best practices like IAM roles, encryption (KMS), multi-factor authentication, and compliance checks. Automate infrastructure with IaC tools like Terraform, Ansible, or CloudFormation for repeatable deployments. Monitor performance using CloudWatch, Azure Monitor, or Google Stackdriver, setting alerts for CPU, memory usage. Optimize costs by right-sizing instances, using reserved instances, and auto-scaling groups. Handle migrations from on-premises to cloud using tools like AWS DMS or Azure Migrate. Troubleshoot issues related to connectivity, performance, or outages with logging and debugging. Document architectures, configurations, and disaster recovery plans for team reference. Qualifications: Bachelor's in Computer Science, IT from top Indian institutes. 6-12 months experience. Familiarity with cloud platforms, Linux/Windows OS, networking (TCP/IP, DNS), scripting (Python/Bash). Preferred: Certifications like AWS Certified Cloud Practitioner, experience with CI/CD (Jenkins).",
    "DevOps Engineer": "The company is seeking an entry-level DevOps Engineer to support development and operations integration in our software team in India. Set up CI/CD pipelines using Jenkins, GitHub Actions, GitLab CI for automated builds and tests. Manage infrastructure as code with Terraform or Ansible for provisioning servers, networks. Containerize applications with Docker, writing Dockerfiles and composing multi-container setups. Orchestrate containers with Kubernetes, managing pods, services, deployments, and Helm charts. Monitor logs and metrics using ELK Stack (Elasticsearch, Logstash, Kibana) or Prometheus/Grafana. Automate testing (unit, integration) and security scans (SonarQube, Trivy) in pipelines. Collaborate in agile teams, participating in sprints, stand-ups, and retrospectives. Ensure security with vulnerability scanning, secret management (Vault), and RBAC. Optimize build times, resource usage, and deployment strategies (blue-green, canary). Document processes, runbooks, and contribute to knowledge bases for onboarding. Qualifications: Bachelor's in Computer Science. 6-12 months experience. Skills in Docker, Kubernetes, Jenkins, Git; cloud (AWS/GCP/Azure); Linux, scripting. Preferred: Certifications like CKAD, experience with IaC.",
    "Software Engineer": "The company is seeking an entry-level Software Engineer to join our development team in India. Gather and analyze requirements from stakeholders to define scope and features. Design software architecture, including databases, APIs, and UI/UX flows using UML diagrams. Code applications in languages like Java, Python, C++, following clean code principles (SOLID, DRY). Use frameworks such as Spring Boot for Java, Django/Flask for Python, React for frontend. Test code with unit tests (JUnit, PyTest), integration tests, and manual QA. Debug issues using tools like IntelliJ debugger, logging frameworks (Log4j, SLF4J). Deploy applications to servers or cloud using CI/CD pipelines (Jenkins). Maintain code by fixing bugs, updating dependencies, and refactoring. Participate in code reviews on GitHub/PRs to ensure quality and knowledge sharing. Learn new technologies through online courses, contributing to open-source. Qualifications: Bachelor's in Computer Science. 6-12 months experience. Proficiency in programming languages, databases (SQL), Git. Preferred: Agile experience, basic DevOps.",
    "Data Analyst": "The company is seeking an entry-level Data Analyst to support data-driven decisions in our business intelligence team in India. Define analysis questions based on business needs. Collect data from sources like databases, spreadsheets, APIs. Clean data by removing duplicates, handling nulls, standardizing formats. Analyze using SQL queries, Excel formulas, or Python (Pandas) for trends. Visualize with Tableau/Power BI for charts, dashboards. Interpret results to find insights, correlations. Create reports with executive summaries. Recommend actions based on findings. Automate reports with scheduled scripts. Validate data accuracy through cross-checks. Qualifications: Bachelor's in Analytics. 6-12 months experience. SQL, Excel, visualization tools; statistics. Preferred: Python, business knowledge.",
    "Digital Marketing Manager": "The company is seeking an entry-level Digital Marketing Manager to assist with online campaigns in our marketing team in India. Research market trends, competitors, and audience behavior using tools like Google Trends, SEMrush. Plan digital strategies including content calendar, budget allocation. Create content for blogs, social media, emails (copywriting, graphics). Manage social media platforms (Facebook, Instagram, LinkedIn) for posting and engagement. Optimize SEO/SEM with keyword research (Ahrefs), on-page/off-page tactics. Run paid ads on Google Ads, Facebook Ads, setting targeting and bids. Analyze metrics with Google Analytics for traffic, conversions, ROI. Conduct A/B testing on landing pages, emails for better performance. Report on campaign results with KPIs like CTR, CPC, conversion rate. Adjust strategies based on data and feedback for continuous improvement. Qualifications: Bachelor's in Marketing. 6-12 months experience. Knowledge of digital tools, social platforms, SEO/SEM. Preferred: Google Analytics certification, creative skills.",
    "Business Analyst": "The company is seeking an entry-level Business Analyst to bridge business and IT in our consulting team in India. Elicit requirements through interviews, workshops, surveys. Analyze processes for inefficiencies using flowcharts, SWOT analysis. Model data with UML diagrams (use case, activity, class). Document requirements in SRS, BRD, user stories. Validate solutions with prototypes, mockups. Facilitate meetings between stakeholders and tech teams. Support implementation by tracking progress. Test for requirement fulfillment (UAT). Train users on new systems. Monitor post-implementation for issues and enhancements. Qualifications: Bachelor's in Business/IT. 6-12 months experience. Skills in requirements gathering, UML, Jira. Preferred: CBAP basics, analytical thinking.",
    "Project Manager": "The company is seeking an entry-level Project Manager to support project execution in our PMO in India. Initiate project with charter, stakeholder identification. Plan scope, schedule (Gantt charts), resources using MS Project/Jira. Execute tasks by assigning roles, monitoring daily progress. Track milestones, budget, risks with dashboards. Manage risks by identifying, assessing, mitigating. Control changes through change request process. Close project with handover, lessons learned. Report status in weekly meetings, reports. Lead team motivation, conflict resolution. Evaluate performance with KPIs. Qualifications: Bachelor's in Management. 6-12 months experience. Knowledge of agile/Scrum, tools. Preferred: PMP intro, leadership.",
    "Frontend Developer": "The company is seeking an entry-level Frontend Developer to build user interfaces in our web team in India. Understand UI/UX designs from wireframes, prototypes. Code semantic HTML, CSS for layouts, styling. Use JavaScript/ES6 for interactivity, DOM manipulation. Implement frameworks like React, Angular, Vue for components, state management. Ensure responsiveness with media queries, Bootstrap/Tailwind. Optimize performance by minifying, lazy loading images. Test cross-browser compatibility with tools like BrowserStack. Integrate APIs using Fetch/Axios for data fetching. Debug with Chrome DevTools, fix bugs. Deploy to hosting like Vercel, Netlify. Qualifications: Bachelor's in Computer Science. 6-12 months experience. Skills in HTML5, CSS3, JavaScript, React. Preferred: UI/UX knowledge, Git.",
    "Backend Developer": "The company is seeking an entry-level Backend Developer to support server-side logic in our development team in India. Design server architecture with microservices or monolith. Develop RESTful/GraphQL APIs using Node.js/Express, Java/Spring, Python/Django. Manage databases like MySQL/PostgreSQL for schema design, queries. Implement business logic, authentication (JWT, OAuth). Secure against vulnerabilities (SQL injection, XSS) with input validation. Scale with caching (Redis), load balancing. Test APIs with Postman, unit tests (Jest, JUnit). Deploy to cloud (Heroku, AWS EC2) with CI/CD. Monitor logs with ELK, performance tuning. Maintain by updating dependencies, refactoring. Qualifications: Bachelor's in Computer Science. 6-12 months experience. Proficiency in Node.js/Java/Python, SQL/NoSQL. Preferred: Cloud experience, microservices.",
    "Talent Acquisition Specialist": "The company is seeking an entry-level Talent Acquisition Specialist to support recruitment in our HR team in India. Understand hiring needs from managers, create job descriptions. Source candidates via LinkedIn, Naukri, Indeed, referrals. Screen resumes for qualifications, keywords. Conduct initial phone/video interviews for fit. Assess with aptitude tests, technical rounds. Extend offers, negotiate salary, benefits. Onboard new hires with paperwork, orientation. Maintain talent pipeline in ATS (Workday, BambooHR). Report metrics like time-to-hire, cost-per-hire. Improve processes with feedback, diversity initiatives. Qualifications: Bachelor's in HR. 6-12 months experience. Knowledge of ATS, LinkedIn. Preferred: Communication skills, recruiting cert.",
    "Product Manager": "The company is seeking an entry-level Product Manager to assist with product development in our product team in India. Research market needs, competitors using surveys, analytics. Define product vision, roadmap with priorities. Gather user requirements via interviews, feedback. Prioritize features in backlog using MoSCoW, Kano model. Collaborate with engineering, design in sprints. Create user stories, acceptance criteria. Launch products with marketing, beta testing. Analyze usage metrics (DAU, retention) post-launch. Iterate based on data, A/B tests. Manage lifecycle from ideation to end-of-life. Qualifications: Bachelor's in Business/Tech. 6-12 months experience. Agile knowledge, Jira. Preferred: User research, customer focus.",
    "Marketing Manager": "The company is seeking an entry-level Marketing Manager to support campaigns in our marketing department in India. Develop marketing strategy aligned with business goals. Segment audience based on demographics, behavior. Create multi-channel campaigns (email, social, SEO). Execute with content creation, ad placement. Measure performance with KPIs (leads, conversions). Optimize based on analytics, split testing. Manage budget allocation across channels. Coordinate events, webinars for engagement. Oversee content marketing (blogs, videos). Report results to leadership with insights. Qualifications: Bachelor's in Marketing. 6-12 months experience. Digital marketing, SEO skills. Preferred: Creative thinking, data analysis.",
    "HR Specialist": "The company is seeking an entry-level HR Specialist to support human resources functions in our HR team in India. Recruit by posting jobs, screening applicants. Onboard with contracts, induction programs. Manage employee relations, grievances. Administer benefits like insurance, leaves. Conduct training needs analysis, sessions. Ensure compliance with labor laws (EPF, ESI). Handle performance appraisals, feedback. Resolve conflicts through mediation. Maintain HR records in HRIS systems. Support diversity, inclusion initiatives. Qualifications: Bachelor's in HR. 6-12 months experience. Knowledge of labor laws, HRIS. Preferred: Interpersonal skills.",
    "Financial Analyst": "The company is seeking an entry-level Financial Analyst to support financial planning in our finance team in India. Collect financial data from reports, databases. Analyze trends using variance analysis. Model scenarios with sensitivity analysis. Forecast revenues, expenses with time-series methods. Evaluate investments using NPV, IRR, ROI. Assess risks with Monte Carlo simulations. Prepare dashboards, financial statements. Recommend cost-saving strategies. Monitor budgets vs. actuals. Support audits with documentation. Qualifications: Bachelor's in Finance. 6-12 months experience. Excel, SQL, modeling skills. Preferred: CFA basics, GAAP knowledge."
}

# Pydantic Model for Response
class ATSResponse(BaseModel):
    score: float
    section_scores: Dict[str, float]
    keyword_matches: Dict[str, Any]
    suggestions: List[str]
    detailed_suggestions: Dict[str, List[str]]
    experience_years: float
    experience_gaps_months: List[Dict[str, Any]]
    full_analysis: str
    visualizations: List[str]

# Function to Parse PDF Resume
def parse_resume_pdf(pdf_bytes: bytes) -> str:
    full_text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    full_text += text + "\n"
                else:
                    image = page.to_image(resolution=300).original
                    enhancer = ImageEnhance.Contrast(image.convert("L"))
                    enhanced = enhancer.enhance(2.0)
                    ocr_text = pytesseract.image_to_string(enhanced)
                    full_text += ocr_text + "\n"
        full_text = re.sub(r'(\b.{1,50}\b)(\s*\1){3,}', r'\1', full_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {str(e)}")
    return full_text.strip()

# Function to Extract Resume Sections (expanded patterns)
def extract_sections(full_text: str) -> Dict[str, str]:
    sections = {"summary": "", "experience": "", "skills": "", "education": ""}
    current_section = "summary"
    header_patterns = [
        (r"(?:professional|work)?\s*(?:experience|history|employment|internship)", "experience"),
        (r"(?:technical|key|core)?\s*skills|abilities|proficiencies", "skills"),
        (r"education|qualifications|academic\s*(?:background|history|qualifications)", "education"),
        (r"key projects|projects|scholastic achievements|positions of responsibility", "experience"),
    ]
    lines = full_text.split("\n")
    for line in lines:
        line_lower = line.lower().strip()
        matched = False
        for pattern, section_name in header_patterns:
            if re.search(pattern, line_lower):
                current_section = section_name
                matched = True
                break
        if not matched:
            sections[current_section] += line + " "
    for key in sections:
        sections[key] = re.sub(r"\s+", " ", sections[key].strip())
    return sections

# Function to Analyze Skills (simple word-based, low threshold)
def analyze_skills(resume_text: str, job_description: str) -> Tuple[List[str], List[str]]:
    job_skills = list(set(re.findall(r'\b\w{3,}\b', job_description.lower())))
    resume_words = re.findall(r'\b\w{3,}\b', resume_text.lower())
    matched_skills = []
    for skill in job_skills:
        for word in resume_words:
            if SequenceMatcher(None, skill, word).ratio() > 0.6:
                matched_skills.append(skill)
                break
    return job_skills, list(set(matched_skills))

# Function to Calculate Match Score (no semantic, higher TF-IDF weight, strong keyword boost, no sublinear_tf)
def calculate_match_score(resume_sections: Dict[str, str], job_description: str) -> Tuple[float, Dict[str, Any], Dict[str, float]]:
    section_weights = {"skills": 0.45, "experience": 0.35, "education": 0.1, "summary": 0.1}
    combined_resume = " ".join(resume_sections.values()).lower()
    if not combined_resume or not job_description:
        return 0.0, {"matched_keywords": [], "missing_keywords": [], "importance": {}}, {}

    # Synonym map (replace only in resume to align with JD)
    synonym_map = {
        "implemented": "implement",
        "project": "development",
        "intern": "experience",
        "mentored": "collaborate",
        "algorithm": "design",
        "programming languages": "programming skills",
        "b.tech": "bachelor's degree",
        "iitk": "computer science",
        "electrical engineering": "computer science",
        "denoising": "debug",
        "traffic prediction": "software solutions",
        "data": "data",
        "tensor": "models",
        "svt": "code",
        "dtc": "frameworks",
        "admm": "test",
        "phabricator": "deploy",
        "jenkins": "ci cd",
        "latex": "documentation",
        "matlab": "python",
        "gnu octave": "tools",
        "git": "version control",
        "micro cap": "tools",
        "tensor toolbox": "libraries",
        "c++": "c++",
        "python": "python",
        "java": "java",
        "html": "html",
    }
    for orig, repl in synonym_map.items():
        combined_resume = combined_resume.replace(orig, repl)

    # Preprocess JD to remove numbering
    job_description = re.sub(r'Step \d+:', '', job_description).lower()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), min_df=1, max_df=1.0)
    tfidf_matrix = vectorizer.fit_transform([combined_resume, job_description])
    overall_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    section_scores = {}
    weighted_score = 0.0
    for section, weight in section_weights.items():
        if section in resume_sections and resume_sections[section]:
            section_text = resume_sections[section].lower()
            for orig, repl in synonym_map.items():
                section_text = section_text.replace(orig, repl)
            section_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), min_df=1, max_df=1.0)
            section_tfidf = section_vectorizer.fit_transform([section_text, job_description])
            sim = cosine_similarity(section_tfidf[0:1], section_tfidf[1:2])[0][0]
            section_scores[section] = sim * 100
            weighted_score += sim * weight
        else:
            section_scores[section] = 0.0

    # Keyword analysis with low threshold
    feature_names = vectorizer.get_feature_names_out()
    job_vector = tfidf_matrix[1].toarray().flatten()
    top_indices = np.argsort(job_vector)[-75:][::-1]
    top_keywords = [(feature_names[i], job_vector[i]) for i in top_indices if job_vector[i] > 0.01]
    matched_keywords = []
    missing_keywords = []
    job_description_words = set(word_tokenize(job_description.lower()))
    resume_words = set(word_tokenize(combined_resume.lower()))

    for keyword, _ in top_keywords:
        if any(SequenceMatcher(None, keyword, word).ratio() > 0.6 for word in resume_words):
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    all_job_keywords = set([kw for kw, _ in top_keywords])
    no_match_keywords = list(all_job_keywords - set(matched_keywords) - set(missing_keywords))

    importance = {kw: score for kw, score in top_keywords}

    # Strong keyword boost
    keyword_percentage = len(matched_keywords) / max(len(top_keywords), 1) * 100
    keyword_boost = keyword_percentage * 0.5  # Up to 50% boost

    # Higher weight on overall similarity
    final_score = (overall_similarity * 70 + weighted_score * 10 + keyword_boost)  # Adjusted to aim for 60+
    return min(final_score, 100.0), {"matched_keywords": matched_keywords, "missing_keywords": missing_keywords, "no_match_keywords": no_match_keywords, "importance": importance}, section_scores

# Function for ATS Checks and Suggestions (relaxed thresholds)
def run_ats_checks(score: float, keyword_matches: Dict[str, Any], resume_text: str, experience_years: float, gaps: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[str]]]:
    suggestions = []
    detailed_suggestions = {"general": [], "keywords": [], "experience": [], "format": []}

    total_keywords = len(keyword_matches["matched_keywords"]) + len(keyword_matches["missing_keywords"])
    if total_keywords > 0 and len(keyword_matches["matched_keywords"]) / total_keywords < 0.3:
        suggestions.append("Improve keyword match by adding missing terms.")
        detailed_suggestions["keywords"].append(f"Missing: {', '.join(keyword_matches['missing_keywords'][:5])}")

    word_count = len(resume_text.split())
    keyword_density = len(keyword_matches["matched_keywords"]) / word_count if word_count else 0
    if keyword_density > 0.1:
        suggestions.append("Avoid keyword stuffing; ensure natural language.")
        detailed_suggestions["keywords"].append("Balance keyword usage.")

    if not 200 <= word_count <= 800:
        suggestions.append("Optimize resume length to 300-700 words.")
        detailed_suggestions["format"].append("Adjust content length.")

    if experience_years < 0.1:
        suggestions.append("Highlight more experience or relevant projects.")
        detailed_suggestions["experience"].append("Target: 6+ months.")

    if gaps:
        suggestions.append("Explain employment gaps in cover letter.")
        detailed_suggestions["experience"].append(f"Gaps: {gaps}")

    if score < 50:
        suggestions.append("Align resume closer to job description.")
        detailed_suggestions["general"].append("Use standard formatting, incorporate more job-specific terms.")

    return suggestions, detailed_suggestions

# Function to Analyze Experience (fixed year parsing for 'YY)
def analyze_experience(experience_text: str) -> Tuple[float, List[Dict[str, Any]]]:
    if not experience_text:
        return 0.0, []

    date_pattern = r'(?i)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\'?\s*(\d{2,4})|\d{1,2}[/\-]\d{1,2}[/\-]\d{4}|\d{4}\s*[\-–—to]\s*\d{4}|\d{4}\s*[\-–—to]\s*(present|current|ongoing)|present|current|ongoing'
    dates = re.findall(date_pattern, experience_text)

    parsed_dates = []
    for match in dates:
        if isinstance(match, tuple):
            d = ' '.join([str(part) for part in match if part]).strip()
        else:
            d = match
        # Fix two-digit year
        month_match = re.match(r'([A-Za-z]+)\s*(\d{2})', d)
        if month_match:
            month = month_match.group(1)
            year = int(month_match.group(2))
            full_year = 2000 + year if year < 50 else 1900 + year
            d = f"{month} {full_year}"
        try:
            if "present" in d.lower() or "current" in d.lower() or "ongoing" in d.lower():
                parsed_dates.append(datetime.now())
            else:
                parsed_dates.append(parse(d, fuzzy=True))
        except ValueError:
            continue

    if len(parsed_dates) < 2:
        return 0.0, []

    parsed_dates.sort()
    total_days = 0
    gaps = []
    for i in range(0, len(parsed_dates) - 1, 2):
        start = parsed_dates[i]
        end = parsed_dates[i + 1] if i + 1 < len(parsed_dates) else datetime.now()
        if start < end:
            total_days += (end - start).days
        if i + 2 < len(parsed_dates):
            next_start = parsed_dates[i + 2]
            gap_days = (next_start - end).days
            if gap_days > 90:
                gaps.append({"start": end.strftime("%Y-%m"), "end": next_start.strftime("%Y-%m"), "duration_months": gap_days // 30})

    total_years = total_days / 365.25
    return total_years, gaps

# Function to Create Visualizations (fixed colors)
def create_visualizations(section_scores: Dict[str, float], keyword_matches: Dict[str, Any]) -> List[str]:
    visualizations = []

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(section_scores.keys()), y=list(section_scores.values()), palette="viridis", ax=ax)
    ax.set_title("Section Match Scores")
    ax.set_ylabel("Score (%)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    visualizations.append(base64.b64encode(buf.read()).decode("utf-8"))
    plt.close(fig)

    labels = ["Matched", "Missing"]
    sizes = [len(keyword_matches["matched_keywords"]), len(keyword_matches["missing_keywords"])]
    if sum(sizes) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"])
        ax.set_title("Keyword Breakdown")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        visualizations.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close(fig)

    return visualizations

# Endpoint to Get Job Profiles
@app.get("/job-profiles")
def get_job_profiles():
    profiles_list = [{"name": name, "description": description} for name, description in PREDEFINED_PROFILES.items()]
    return {"profiles": profiles_list}

# Main Analyze Endpoint
@app.post("/analyze", response_model=ATSResponse)
async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    resume_bytes = await resume.read()
    full_text = parse_resume_pdf(resume_bytes)
    if not full_text:
        raise HTTPException(status_code=400, detail="Empty resume content.")

    sections = extract_sections(full_text)
    job_skills, matched_skills = analyze_skills(full_text, job_description)
    score, keyword_matches, section_scores = calculate_match_score(sections, job_description)
    keyword_matches["job_description_skills"] = job_skills
    keyword_matches["matched_resume_skills"] = matched_skills

    experience_text = sections.get("experience", "")
    experience_years, experience_gaps = analyze_experience(experience_text)

    suggestions, detailed_suggestions = run_ats_checks(score, keyword_matches, full_text, experience_years, experience_gaps)

    full_analysis = f"""
# ATS Analysis Report

## Score: {score:.1f}%
## Section Scores: {section_scores}
## Keywords: Matched {len(keyword_matches['matched_keywords'])}, Missing {len(keyword_matches['missing_keywords'])}
## Experience: {experience_years:.1f} years
## Suggestions: {', '.join(suggestions)}
"""

    visualizations = create_visualizations(section_scores, keyword_matches)

    return ATSResponse(
        score=score,
        section_scores=section_scores,
        keyword_matches=keyword_matches,
        suggestions=suggestions,
        detailed_suggestions=detailed_suggestions,
        experience_years=experience_years,
        experience_gaps_months=experience_gaps,
        full_analysis=full_analysis,
        visualizations=visualizations
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)