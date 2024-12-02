import PyPDF2
import re
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Load SpaCy language model
nlp = spacy.load("en_core_web_sm")

files = "C://Users//himan//PycharmProjects//Resume_Extraction//HIMANK SHARMA - Resume.pdf"
#Creating a function to extract the test from pdf
def extract_text_from_pdf(file):
    with open(files, "rb") as file:
        reader = PyPDF2.PdfReader(files)
        text = ""

        for page in reader.pages:
            text += page.extract_text()
        return text

#1.
#Extracting email addresses and phone numbers
def extract_contact_info(text):
    #Email Regex

    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_pattern,text)

    #Phone number regex
    phone_patter = r'\+?\d{1,4}[\s-]?\(?\d{1,3}\)?[\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,9}'
    phones = re.findall(phone_patter, text)

    return emails, phones

#2.
#Extracting skills
skills_list = [
"Python", "Java", "Machine Learning", "Data Analysis",
    "Project Management", "Communication", "Leadership"
]

#Function to extract skills
def extract_skills(text):
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp(skill.lower()) for skill in skills_list]
    matcher.add("SKILLS", patterns)

    doc = nlp(text.lower())
    matches = matcher(doc)

    extracted_skills = set()
    for match_id, start, end in matches:
        extracted_skills.add(doc[start:end].text.capitalize())

    return list(extracted_skills)


#3.
#Extracting Certifications
certifications_list = [
    "AWS Certified Solutions Architect", "PMP", "Certified Data Scientist",
    "Scrum Master", "Google Cloud Certified", "Azure Fundamentals"
]

# Function to extract certifications
def extract_certifications(text):
    certifications_found = [
        cert for cert in certifications_list if cert.lower() in text.lower()
    ]
    return certifications_found


#4.
# Function to calculate similarity between resume and job description
def calculate_similarity(resume_text, job_description_text):
    documents = [resume_text, job_description_text]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    similarity_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return similarity_score



#Main
# Load resume and job description
resume_path = "resume.pdf"
job_description_path = "job_description.txt"

# Extract resume text
resume_text = extract_text_from_pdf(resume_path)

# Extract contact info, skills, and certifications
emails, phones = extract_contact_info(resume_text)
skills = extract_skills(resume_text)
certifications = extract_certifications(resume_text)

# Load job description text
with open(job_description_path, "r") as file:
    job_description_text = file.read()

# Calculate similarity
similarity_score = calculate_similarity(resume_text, job_description_text)

# Display results
print("Contact Information:")
print("Emails:", emails)
print("Phones:", phones)
print("\nSkills:", skills)
print("\nCertifications:", certifications)
print("\nSimilarity with Job Description:", f"{similarity_score * 100:.2f}%")

