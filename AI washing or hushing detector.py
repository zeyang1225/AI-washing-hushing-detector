import streamlit as st
import pdfplumber
import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
nltk.download('stopwords')

st.title("AI Washing/Hushing Detector")
st.write("Upload your firm's 10-K filings or earnings call transcripts to calculate the discordance between AI disclosure and actual AI investment.")

uploaded_files = st.file_uploader("Upload documents (PDF format)", type=["pdf"], accept_multiple_files=True)


ai_skilled_employees = st.number_input("Enter the number of AI-skilled employees", min_value=0, step=1)
total_employees = st.number_input("Enter the total number of employees from Compustat", min_value=1, step=1)


if uploaded_files and ai_skilled_employees and total_employees:
    st.write("Processing files...")

    def extract_text_from_pdf(pdf_files):
        text = ""
        for pdf_file in pdf_files:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
        return text


    if uploaded_files:
        document_text = extract_text_from_pdf(uploaded_files)
        st.write("Extracted text from uploaded files.")

def extract_text_from_excel(excel_files, text_column="Disclosure Text"):
    text = ""
    for excel_file in excel_files:
        df = pd.read_excel(excel_file)
        if text_column in df.columns:
            text += " ".join(df[text_column].dropna().astype(str).tolist())
        else:
            st.warning(f"Column '{text_column}' not found in the uploaded file.")
    return text


ai_seed_terms = ["machine learning", "neural network", "deep learning", "computer vision",
                 "natural language processing"]


# Placeholder function for Word2Vec expansion (use pretrained model if available)
def expand_keywords(seed_terms):
    model = Word2Vec(sentences=[seed_terms], vector_size=100, window=5, min_count=1, workers=4)
    expanded_terms = set(seed_terms)
    for term in seed_terms:
        try:
            similar_words = model.wv.most_similar(term, topn=5)
            expanded_terms.update([word for word, similarity in similar_words])
        except KeyError:
            continue
    return list(expanded_terms)


expanded_ai_terms = expand_keywords(ai_seed_terms)
st.write("Expanded AI-related terms:", expanded_ai_terms)

# Process uploaded Excel files if available
if uploaded_files:
    document_text = extract_text_from_excel(uploaded_files)
    st.write("Extracted text from uploaded files.")


    # Calculate AI disclosure score using tf-idf
    def calculate_ai_disclosure(text, ai_terms):
        vectorizer = TfidfVectorizer(vocabulary=ai_terms, stop_words=nltk.corpus.stopwords.words('english'))
        tfidf_matrix = vectorizer.fit_transform([text])
        tfidf_scores = tfidf_matrix.sum(axis=1)
        return tfidf_scores.item()


    ai_disclosure_score = calculate_ai_disclosure(document_text, expanded_ai_terms)
    st.write("AI Disclosure Score:", ai_disclosure_score)

# Calculate AI investment score from employee data
if total_employees > 0:
    ai_invest_score = ai_skilled_employees / total_employees
    st.write("AI Investment Score:", ai_invest_score)


# Calculate discordance and determine AI washing/hushing
def calculate_discordance(ai_disclosure, ai_invest):
    return ai_disclosure - ai_invest


if 'ai_disclosure_score' in locals() and 'ai_invest_score' in locals():
    discordance = calculate_discordance(ai_disclosure_score, ai_invest_score)
    st.write("AI Washing/Hushing Indicator (Discordance):", discordance)
    if discordance > 0:
        st.write("The firm may be engaging in AI washing.")
    elif discordance < 0:
        st.write("The firm may be engaging in AI hushing.")
    else:
        st.write("The firm’s AI disclosure matches its investment.")