import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, BartTokenizer, BartForConditionalGeneration, pipeline
import fitz  # PyMuPDF for PDF procesig

# ---- Load Pre-trained Models ---- #
@st.cache_resource
def load_models():
    qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    qa_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    return qa_model, qa_tokenizer, summarization_model, summarization_tokenizer, ner_pipeline

qa_model, qa_tokenizer, summarization_model, summarization_tokenizer, ner_pipeline = load_models()

# ---- Load Cleaned Datasets ---- #
@st.cache_data
def load_cleaned_datasets():
    squad_df = pd.read_csv("cleaned_squad.csv")
    cnn_df = pd.read_csv("cleaned_cnn_dailymail.csv")
    return squad_df, cnn_df

squad_df, cnn_df = load_cleaned_datasets()

# ---- Streamlit UI ---- #
st.title("📖 NLP Model Deployment: Question Answering, Summarization & NER")
st.write("Perform Question Answering, Document Summarization, and Named Entity Recognition using pre-trained NLP models.")

# ---- Sidebar for Task Selection ---- #
task_choice = st.sidebar.selectbox("Select Task", ("Question Answering", "Document Summarization", "Named Entity Recognition"))

# ---- Question Answering Task ---- #
if task_choice == "Question Answering":
    st.subheader("🤔 Question Answering Task")
    
    input_method = st.radio("Choose input method:", ("Select from Dataset", "Enter Manually"))
    
    if input_method == "Select from Dataset":
        selected_paragraph = st.selectbox("Select a paragraph:", squad_df["context"].unique())
    else:
        selected_paragraph = st.text_area("Enter your paragraph:")
    
    # Display selected paragraph
    st.write("### Selected Paragraph:")
    st.write(selected_paragraph)
    
    # User inputs question
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_question and selected_paragraph:
            inputs = qa_tokenizer(user_question, selected_paragraph, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = qa_model(**inputs)

            # Extract start and end tokens
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = qa_tokenizer.convert_tokens_to_string(
                qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
            )

            st.write(f"### 🏆 Predicted Answer: {answer}")
        else:
            st.warning("⚠️ Please enter a question and a paragraph.")

# ---- Document Summarization Task ---- #
elif task_choice == "Document Summarization":
    st.subheader("📜 Document Summarization Task")
    
    # User selects how to provide input
    input_option = st.radio("Choose input method:", ("Select from Dataset", "Enter Manually", "Upload PDF"))

    if input_option == "Select from Dataset":
        selected_article = st.selectbox("Select a paragraph:", cnn_df["article"].unique())
        
        # Display selected paragraph
        st.write("### Selected Paragraph:")
        st.write(selected_article)
        
    elif input_option == "Enter Manually":
        selected_article = st.text_area("Enter your text:")
    
    elif input_option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        
        if uploaded_file is not None:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            selected_article = " ".join([page.get_text("text") for page in doc])
            st.write("### Extracted Text from PDF:")
            st.write(selected_article)
    
    if st.button("Summarize"):
        if selected_article:
            inputs = summarization_tokenizer(selected_article, return_tensors="pt", max_length=1024, truncation=True)
            
            with torch.no_grad():
                summary_ids = summarization_model.generate(
                    **inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4
                )
            
            summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.write(f"### ✨ Generated Summary: {summary}")
        else:
            st.warning("⚠️ Please provide text to summarize.")

# ---- Named Entity Recognition (NER) ---- #
elif task_choice == "Named Entity Recognition":
    st.subheader("🔍 Named Entity Recognition")
    user_text = st.text_area("Enter text for NER analysis:")
    
    if st.button("Analyze"):
        if user_text:
            ner_results = ner_pipeline(user_text)
            for entity in ner_results:
                st.write(f"**Entity:** {entity['word']} | **Type:** {entity['entity']} | **Confidence:** {entity['score']:.2f}")
        else:
            st.warning("⚠️ Please enter text for analysis.")

st.write("---")
st.write("🚀 Developed with ❤️ using Streamlit")
