import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load a more context-aware chatbot model
chatbot = pipeline("text-generation", model="gpt2")   

# Define a rule-based response function
def healthcare_chatbot(user_input):
    user_input = user_input.lower()  # Normalize input
    
    if "fever" in user_input:
        return ("A fever is a sign that your body is fighting an infection. "
                "Stay hydrated, rest, and monitor your temperature. If it exceeds 102°F (38.8°C) or lasts more than 3 days, consult a doctor.")
    
    elif "cold" in user_input or "cough" in user_input:
        return ("For a cold or cough, drink warm fluids, rest, and use a humidifier. "
                "If symptoms persist for more than a week or worsen, see a healthcare provider.")

    elif "headache" in user_input:
        return ("Headaches can be caused by stress, dehydration, or underlying issues. "
                "Drink water, rest, and avoid bright screens. If the headache is severe or persistent, seek medical advice.")

    elif "appointment" in user_input:
        return ("Would you like me to help schedule an appointment with a doctor? "
                "I can provide details on nearby clinics and available time slots.")

    elif "medication" in user_input or "medicine" in user_input:
        return ("Always take medications as prescribed. If you have side effects or concerns, consult your doctor. "
                "Would you like information on specific medicines?")

    elif "emergency" in user_input or "chest pain" in user_input:
        return ("This sounds serious. If you are experiencing chest pain, difficulty breathing, or any life-threatening symptoms, "
                "call emergency services or go to the nearest hospital immediately.")

    else:
        # Use BioGPT model to generate a response
        response = chatbot(user_input, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']

# Streamlit web app
def main():
    st.title("🩺 Healthcare Assistant Chatbot")
    st.markdown("I can provide health advice, symptom checks, and appointment guidance.")

    user_input = st.text_area("How can I assist you today?", "")
    
    if st.button("Submit"):
        if user_input.strip():
            response = healthcare_chatbot(user_input)
            st.write(f"**User:** {user_input}")
            st.success(f"**Healthcare Assistant:** {response}")
        else:
            st.warning("Please enter a valid query.")

if __name__ == "__main__":
    main()


