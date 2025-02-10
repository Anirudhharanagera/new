import streamlit as st
import pandas as pd
import pickle
import gzip

# Function to load the trained model
def load_model():
    with gzip.open("model.pkl.gz", "rb") as f:
        model = pickle.load(f)
    return model

# Load the trained model
model = load_model()

# Streamlit UI
st.title("NLP Model Deployment")
st.write("This app allows users to perform Question Answering and Document Summarization using the trained model.")

# Load cleaned datasets
@st.cache_data
def load_cleaned_datasets():
    squad_df = pd.read_csv("cleaned_squad.csv")
    cnn_df = pd.read_csv("cleaned_cnn_dailymail.csv")
    return squad_df, cnn_df

squad_df, cnn_df = load_cleaned_datasets()

# Sidebar - Task selection
task_choice = st.sidebar.selectbox("Select Task", ("Question Answering", "Document Summarization"))

if task_choice == "Question Answering":
    st.subheader("Question Answering Task")
    
    # User selects a paragraph from the SQuAD dataset
    selected_paragraph = st.selectbox("Select a paragraph:", squad_df["context"].unique())
    
    # Display selected paragraph
    st.write("### Selected Paragraph:")
    st.write(selected_paragraph)
    
    # User inputs question
    user_question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if user_question:
            input_data = pd.DataFrame({"question": [user_question], "context": [selected_paragraph]})
            prediction = model.predict(input_data)
            st.write(f"Predicted Answer: {prediction[0]}")
        else:
            st.warning("Please enter a question.")

elif task_choice == "Document Summarization":
    st.subheader("Document Summarization Task")
    
    # User selects how to provide input
    input_option = st.radio("Choose input method:", ("Select from Dataset", "Enter Manually"))
    
    if input_option == "Select from Dataset":
        selected_article = st.selectbox("Select a paragraph:", cnn_df["article"].unique())
        
        # Display selected paragraph
        st.write("### Selected Paragraph:")
        st.write(selected_article)
        
        if st.button("Summarize"):
            input_data = pd.DataFrame({"article": [selected_article]})
            summary = model.predict(input_data)
            st.write(f"Generated Summary: {summary[0]}")
    
    elif input_option == "Enter Manually":
        user_text = st.text_area("Enter your text:")
        
        if st.button("Summarize"):
            if user_text:
                input_data = pd.DataFrame({"article": [user_text]})
                summary = model.predict(input_data)
                st.write(f"Generated Summary: {summary[0]}")
            else:
                st.warning("Please enter text to summarize.")

st.write("---")
st.write("Developed with ❤️ using Streamlit")
