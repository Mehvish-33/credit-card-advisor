# Full AI-Powered Credit Card Advisor (CLI + Voice + PDF + LangChain + Pinecone)
import streamlit as st
import requests
from langchain.chains import LLMChain, ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.text_splitter import CharacterTextSplitter

from langchain.memory import ConversationBufferWindowMemory
from transformers import pipeline
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
import pyttsx3
import speech_recognition as sr
import os
import pinecone 
from pinecone import Pinecone


# === SETUP ===
load_dotenv()
#pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "users"
urls = [
    "https://select.finology.in/credit-card/sbi-cashback-credit-card",
    "https://select.finology.in/credit-card/tata-neu-infinity-hdfc-credit-card",
    "https://select.finology.in/credit-card/american-express-bank-smart-earn-credit-card",
    "https://select.finology.in/credit-card/axis-bank-myzone-credit-card",
    "https://select.finology.in/credit-card/club-vistara-idfc-first-bank-credit-card",
    "https://select.finology.in/credit-card/idfc-bank-first-select-credit-card",
    "https://select.finology.in/credit-card/sbi-simply-click-credit-card",
    "https://select.finology.in/credit-card/idfc-first-bank-wow-credit-card",
    "https://select.finology.in/credit-card/icici-bank-amazon-pay-credit-card",
    "https://select.finology.in/credit-card/hdfc-bank-infinia-credit-card",
    "https://select.finology.in/credit-card/axis-bank-reserve-credit-card",
    "https://select.finology.in/credit-card/sbi-bank-simply-save-credit-card",
    "https://select.finology.in/credit-card/axis-bank-ace-credit-card",
    "https://select.finology.in/credit-card/american-express-platinum-credit-card",
    "https://select.finology.in/credit-card/idfc-first-bank-wealth-credit-card",
    "https://select.finology.in/credit-card/hdfc-bank-regalia-credit-card",
    "https://select.finology.in/credit-card/axis-bank-atlas-credit-card",
    "https://select.finology.in/credit-card/icici-bank-hpcl-supersaver-credit-card",
    "https://select.finology.in/credit-card/kotak-bank-kaching6e-credit-card",
    "https://select.finology.in/credit-card/hsbc-premier-credit-card",
    "https://select.finology.in/credit-card/idfc-first-bank-millennia-credit-card",
    "https://select.finology.in/credit-card/american-express-membership-rewards-credit-card",
    "https://select.finology.in/credit-card/hdfc-bank-millenia-credit-card",
    "https://select.finology.in/credit-card/axis-bank-indian-oil-credit-card",
    "https://select.finology.in/credit-card/sbi-bank-elite-credit-card",
    "https://select.finology.in/credit-card/rbl-bank-world-safari-credit-card",
    "https://select.finology.in/credit-card/icici-bank-platinum-chip-credit-card",
    "https://select.finology.in/credit-card/standard-chartered-smart-credit-card",
    "https://select.finology.in/credit-card/federal-bank-signet-credit-card",
    "https://select.finology.in/credit-card/indusind-bank-legend-credit-card"
]



#pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#index_name = "credit-card-index"

loader = UnstructuredURLLoader(urls=urls)
docs = CharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(loader.load())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings()
vectorstore = PineconeStore.from_documents(docs, embedding=embedding, index_name=users)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

qa_nlp = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

filter_prompt = PromptTemplate.from_template("""
Act as a credit card expert. Based on user profile and the given credit cards, select the top 5 with:
- Name, Issuer, Annual Fee
- Reward Structure, Eligibility
- Match Score (0–100)
- Apply Link, Top 3 reasons, Estimated Yearly Benefit

USER:
Income: {income}
Habits: {habits}
Benefits: {benefits}
Credit Score: {creditScore}
Existing Cards: {existingCards}

CARDS:
{cards}
""")

human_prompt = PromptTemplate.from_template("""
You're a friendly advisor. Present the 5 best matched credit cards with:
- Name & Issuer
- Fees, Perks, Reward Type
- Eligibility & Benefits
- Yearly Gain
- Match Score
- Apply Link
Wrap with a friendly conclusion.

DATA:
{filtered_cards}
""")

llm = OpenAI(temperature=0.2)
memory = ConversationBufferWindowMemory(k=5)
filter_chain = LLMChain(prompt=filter_prompt, llm=llm)
response_chain = LLMChain(prompt=human_prompt, llm=llm)
convo = ConversationChain(llm=llm, memory=memory)

def save_pdf_report(title, summary):
    pdf = canvas.Canvas("credit_card_recommendations.pdf")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(30, 800, title)
    for i, line in enumerate(summary.split("\n")):
        pdf.drawString(30, 780 - 15*i, line[:110])
    pdf.save()

def voice_input(prompt_text="Speak now..."):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt_text)
        speak(prompt_text)
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return input("Could not recognize voice. Type manually: ")

def generate_card_recommendation(user_profile):
    query = f"Top credit cards for income {user_profile['income']}, spending on {user_profile['habits']}"
    retrieved_docs = retriever.invoke(query)
    card_snippets = "\n".join([doc.page_content for doc in retrieved_docs])

    filter_input = {
        "income": user_profile["income"],
        "habits": ", ".join(user_profile["habits"]),
        "benefits": ", ".join(user_profile["benefits"]),
        "creditScore": user_profile.get("creditScore", "Unknown"),
        "existingCards": ", ".join(user_profile.get("existingCards", [])),
        "cards": card_snippets
    }

    filtered = filter_chain.run(filter_input)
    summary = qa_nlp(filtered, max_length=800, min_length=300, do_sample=False)[0]['summary_text']
    final_response = response_chain.run({"filtered_cards": summary})
    save_pdf_report("Your Credit Card Recommendations", final_response)
    speak(final_response)
    return final_response

if __name__ == "__main__":
    print("Welcome to Credit Card Advisor AI 💳")
    speak("Welcome to Credit Card Advisor AI")
    income = voice_input("Speak your annual income (e.g., Rs 10 lakh/year): ")
    habits = voice_input("Spending categories (travel, groceries, fuel, etc.): ").split(",")
    benefits = voice_input("Preferred benefits (cashback, lounge access, etc.): ").split(",")
    credit_score = voice_input("Your credit score (optional): ")
    existing_cards = voice_input("Any existing cards (comma-separated): ").split(",")

    user_profile = {
        "income": income.strip(),
        "habits": [h.strip() for h in habits],
        "benefits": [b.strip() for b in benefits],
        "creditScore": credit_score.strip(),
        "existingCards": [c.strip() for c in existing_cards if c.strip() != ""]
    }

    response = generate_card_recommendation(user_profile)
    print("\n===== RECOMMENDED CARDS =====\n")
    print(response)
    speak("A PDF summary has been generated. Thank you!")
    
    
    
    #FRONTEND



st.set_page_config(page_title="Credit Card Advisor", layout="centered")

st.title("💳 Intelligent Credit Card Advisor")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login"):
    res = requests.post("http://localhost:8000/login", json={"email": email, "password": password})
    if res.ok:
        st.success("Logged in!")
    else:
        st.error(res.json()["detail"])

with st.form("card-form"):
    income = st.text_input("Annual Income (e.g., ₹15L/year)")
    habits = st.text_input("Spending Categories (comma-separated)")
    benefits = st.text_input("Preferred Benefits (cashback, lounge access, etc.)")
    credit_score = st.text_input("Credit Score")
    existing_cards = st.text_input("Current Cards")

    submit = st.form_submit_button("Get Recommendations")

if submit:
    profile = {
        "income": income,
        "habits": habits.split(","),
        "benefits": benefits.split(","),
        "creditScore": credit_score,
        "existingCards": existing_cards.split(",")
    }
    recommendation = generate_card_recommendation(profile)
    st.success("Top 5 Credit Cards:")
    st.write(recommendation)
    with open("credit_card_recommendations.pdf", "rb") as f:
        st.download_button("📄 Download PDF", f, file_name="credit_card_recommendations.pdf")

