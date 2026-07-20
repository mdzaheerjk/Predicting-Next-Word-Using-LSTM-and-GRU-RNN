import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('Models/next_word_lstm.h5')

with open('Models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict multiple next words
def predict_next_words(tokenizer, input_text, num_words, model, max_sequence_len):
    current_text = input_text
    for _ in range(num_words):
        # 1. Convert text to a 1D sequence list: [ [1, 2, 3] ] -> extract [0] to get [1, 2, 3]
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        
        # 2. Pad the 1D list. Passing [token_list] turns it into a valid 2D array: (1, max_sequence_len-1)
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # 3. Predict the probability distribution
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        if not output_word:
            break
            
        current_text += " " + output_word
    return current_text


# ==========================================
# SIDEBAR CONFIGURATION (ALL DATA ONLY)
# ==========================================
with st.sidebar:
    st.header("📋 Training Source Data")
    
    try:
        # Read and clean the text file lines dynamically
        with open('Data/IndiaUS.txt', 'r', encoding='utf-8') as file:
            raw_lines = file.readlines()
            
        # Clean text line by line to remove extra formatting/whitespace
        cleaned_lines = [line.strip() for line in raw_lines if line.strip()]
        cleaned_corpus = "\n\n".join(cleaned_lines)
        
        # Display all data cleanly in a dedicated text container
        st.text_area(
            label="All Source Text Dataset Content:", 
            value=cleaned_corpus, 
            height=600, 
            disabled=True
        )
            
    except FileNotFoundError:
        st.error("Could not find file at 'Data/IndiaUS.txt'. Check your path directory layout.")

# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
st.title("🔮 Next Word Prediction")
st.caption("Powered by an LSTM Deep Learning Network with Early Stopping Regularisation")

# Input field for the starting sentence
input_text = st.text_input("Enter the sequence of Words:")

# Slider to choose word limits dynamically
num_words_to_predict = st.slider(
    label="Number of consecutive words to generate:", 
    min_value=1, 
    max_value=10, 
    value=3
)

# Execution trigger
if st.button("Generate Text Prediction", type="primary"):
    max_sequence_len = model.input_shape[1] + 1  
    
    # Process sequence predictions
    predicted_text = predict_next_words(tokenizer, input_text, num_words_to_predict, model, max_sequence_len)
    
    # Clean UI rendering output
    st.subheader("Results")
    st.success(f"**Generated Sequence:** {predicted_text}")
