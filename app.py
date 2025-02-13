import streamlit as st
import nitk
from transformers import pipeline 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import pandas as pd

# Initialize an empty DataFrame for logging
logs = pd.DataFrame(columns=["User Query", "Bot Response"])

def log_query(user_query, bot_response):
    global logs
    new_entry = pd.DataFrame([[user_query, bot_response]], columns=["User Query", "Bot Response"])
    logs = pd.concat([logs, new_entry], ignore_index=True)






chatbot = pipeline("text-generation", model="distilgpt2")
question_answering = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased")
summarization = pipeline("summarization", model="facebook/bart-large-cnn")
translation = pipeline("translation_en_to_fr", model="t5-base")



def main():
    st.title("Healthcare Assistant Chatbot")
    
    user_input = st.text_input("How can I assist you today?")
    
    if st.button("Submit"):
        if user_input:
            st.write("User: " + user_input)
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: " + response)
        else:
            st.write("Please enter a message to get a response.")

def healthcare_chatbot(user_input):
    # Dummy response function
    return "This is a response to: " + user_input
def healthcare_chatbot(user_input):
    if "symptom" in user_input:
        return "Please consult a doctor for accurate advice."
    elif "corona" in user_input:
        return "fever,cough,lungs infection,breathing problem"
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in user_input:
        return "It's important to take prescribed medicines regularly. If you have concerns, consult your doctor."
    elif "summarize:" in user_input.lower():  
    # Extract the text after "summarize:"  
     text_to_summarize = user_input.split("summarize:", 1)[-1].strip()
    
    if text_to_summarize:
        summary = summarization(text_to_summarize, max_length=100, min_length=50, do_sample=False)
        return summary[0]["summary_text"]
    else:
        return "Please provide text to summarize after 'summarize:'."

        log_query(user_input, response)

    return response
   
    st.subheader("Available NLP Models for Query Processing")

nlp_models = [
    "DistilGPT-2 (Text Generation)",
    "DistilBERT (Question Answering)",
    "DistilBERT (Sentiment Analysis)",
    "BART (Summarization)",
    "T5 (Translation)"
]

for model in nlp_models:
    st.write(f"- {model}")

if st.button("View Logs"):
    if logs.empty:
        st.write("No queries logged yet.")
    else:
        st.dataframe(logs)


if __name__ == "__main__":
    main()

