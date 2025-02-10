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
st.write("This app processes cleaned SQuAD & CNN/DailyMail datasets and makes predictions using the trained model.")

# Load cleaned datasets
@st.cache_data
def load_cleaned_datasets():
    squad_df = pd.read_csv("cleaned_squad.csv")
    cnn_df = pd.read_csv("cleaned_cnn_dailymail.csv")
    return squad_df, cnn_df

squad_df, cnn_df = load_cleaned_datasets()

# Sidebar - Dataset selection
dataset_choice = st.sidebar.selectbox("Select Dataset", ("SQuAD", "CNN/DailyMail"))

if dataset_choice == "SQuAD":
    st.subheader("SQuAD Dataset Sample")
    st.write(squad_df.sample(5))
    
    # User input for prediction
    user_question = st.text_input("Enter a question:")
    user_context = st.text_area("Enter the context:")
    
    if st.button("Predict Answer"):
        if user_question and user_context:
            input_data = pd.DataFrame({"question": [user_question], "context": [user_context]})
            prediction = model.predict(input_data)
            st.write(f"Predicted Answer: {prediction[0]}")
        else:
            st.warning("Please enter both question and context.")

elif dataset_choice == "CNN/DailyMail":
    st.subheader("CNN/DailyMail Dataset Sample")
    st.write(cnn_df.sample(5))
    
    # User input for summarization
    user_article = st.text_area("Enter an article for summarization:")
    
    if st.button("Generate Summary"):
        if user_article:
            input_data = pd.DataFrame({"article": [user_article]})
            summary = model.predict(input_data)
            st.write(f"Generated Summary: {summary[0]}")
        else:
            st.warning("Please enter an article.")

st.write("---")
st.write("Developed with ❤️ using Streamlit")
