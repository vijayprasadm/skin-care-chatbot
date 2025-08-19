import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import speech_recognition as sr
import pyttsx3
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import base64

# Configure Gemini API
genai.configure(api_key="AIzaSyBDH5HjKT6N_sCF3QIGOzuHN2roSCH7TVs")
# Model classes
CONDITION_CLASSES = ["Dark Spots", "Normal", "Puffy Eyes", "Wrinkles"]
SKIN_TYPE_CLASSES = ['Acne', 'Dry', 'Oily']
SKIN_TYPE_CLASSES2 = ["dry", "normal", "oily"]

# Load models
@st.cache_resource
def load_models():
    try:
        condition_model = tf.keras.models.load_model("models/inceptionv3_model_darkspots.h5", compile=False)
        skin_type_model = tf.keras.models.load_model("models/newmobilenetv2_skin_classification_model.keras")
        compare_model = tf.keras.models.load_model("models/ninceptionv3_model_for_oilyskin.h5", compile=False)
        return condition_model, skin_type_model, compare_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

condition_model, skin_type_model, compare_model = load_models()

def preprocess_image(image, model_type="inception"):
    try:
        if model_type != "mobilenet":
            img = Image.open(image).convert("RGB")
            img = img.resize((299, 299)) 
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return preprocess_inception(img_array) 
        else:
            img = load_img(image, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_mobilenet(img_array.reshape(1, 224, 224, 3))
            return img_array
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def get_gemini_response(input_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        response = model.generate_content(input_text)
        return response.text if response else "No response generated."
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        return "Could not get response."

class ImageClassifier:
    def __init__(self, condition_model, skin_type_model, compare_model):
        self.condition_model = condition_model
        self.skin_type_model = skin_type_model
        self.compare_model = compare_model
    
    def predict(self, image):
        condition_img = preprocess_image(image, model_type="inception")
        skin_type_img = preprocess_image(image, model_type="mobilenet")
        compare_model_img = preprocess_image(image, model_type="inception")
        
        if any(img is None for img in [condition_img, skin_type_img, compare_model_img]):
            return None, None, None
        
        with st.spinner("🔍 Analyzing skin..."):
            condition_pred = self.condition_model.predict(condition_img, verbose=0)[0]
            skin_type_pred = self.skin_type_model.predict(skin_type_img, verbose=0)
            compare_model_pred = self.compare_model.predict(compare_model_img, verbose=0)
            
            condition_result = {CONDITION_CLASSES[i]: float(condition_pred[i]) for i in range(len(CONDITION_CLASSES))}
            
            compare_predicted_index = np.argmax(compare_model_pred[0])
            compare_predicted_class = SKIN_TYPE_CLASSES2[compare_predicted_index]

            if compare_predicted_class == "normal":
                compare_result = "Normal skin"
            else:
                compare_result = {SKIN_TYPE_CLASSES2[i]: float(compare_model_pred[0, i]) for i in range(len(SKIN_TYPE_CLASSES2))}
            
            top_2_indices = np.argsort(skin_type_pred[0])[-2:]
            top_2_indices = top_2_indices[::-1]

            predicted_class_index = top_2_indices[0]
            second_predicted_index = top_2_indices[1]

            predicted_class = SKIN_TYPE_CLASSES[predicted_class_index]
            second_predicted_class = SKIN_TYPE_CLASSES[second_predicted_index]

            if predicted_class == "acne":
                skin_type_result = f"acne + {second_predicted_class}"
            else:
                skin_type_result = predicted_class
            
            return condition_result, skin_type_result, compare_result

if None not in [condition_model, skin_type_model, compare_model]:
    classifier = ImageClassifier(condition_model, skin_type_model, compare_model)
else:
    classifier = None
    st.error("Failed to load one or more models. Please check the model files.")

# UI Design
st.title("🧖 AI Skin Analysis Chatbot")
st.caption("Upload an image and get AI-based skin analysis.")

# Initialize session state for image tracking
if 'last_image' not in st.session_state:
    st.session_state.last_image = None

image_file = st.file_uploader("Upload a skin photo", type=["png", "jpg", "jpeg"])

if image_file and classifier:
    # Check if image has changed
    if st.session_state.last_image != image_file.name:
        st.session_state.last_image = image_file.name
        # Clear chat history and summary when new image is uploaded
        if 'history' in st.session_state:
            del st.session_state.history
        if 'summary' in st.session_state:
            del st.session_state.summary
        if 'chat_history' in st.session_state:
            del st.session_state.chat_history
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image_file, caption="Your Skin Photo", use_container_width=True)
    
    condition_probs, skin_type_probs, compare_skin_probs = classifier.predict(image_file)
    
    if condition_probs and skin_type_probs:
        with col2:
            st.subheader("📊 Skin Condition Probability")
            for condition, prob in condition_probs.items():
                st.progress(prob)
                st.write(f"**{condition}**: {prob:.1%}")

            st.subheader("📊 Skin Type")
            FaceType = skin_type_probs
            if compare_skin_probs != "Normal skin tone":
                st.write(f"**{skin_type_probs}**")
                FaceType = skin_type_probs
            else:
                st.write(f"**Normal Skin**")
                FaceType = "Normal Skin"
        
        st.session_state.last_analysis = {
            "condition": condition_probs, 
            "skin_type": skin_type_probs,
            "compare_type": compare_skin_probs
        }
        
        if "summary" not in st.session_state:
            with st.spinner("💡 Generating summary..."):
                prompt = f"Skin condition results: {condition_probs}. Skin type results: {skin_type_probs}. Provide a short skincare summary."
                st.session_state.summary = get_gemini_response(prompt)

        if st.session_state.summary:
            st.markdown(f"**Summary:** {st.session_state.summary}")

# Voice recognition functions
def recognize_speech():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5
    recognizer.dynamic_energy_threshold = True
    with sr.Microphone() as source:
        global SHOWED
        if not SHOWED:
            st.info("Listening... (say 'stop' or remain silent to end)")
            SHOWED = True
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=None)
            text = recognizer.recognize_google(audio)
            if text.lower() == 'stop':
                return "stop", False
            return text, True
        except sr.WaitTimeoutError:
            return "", False
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand.", False
        except sr.RequestError:
            return "Speech service is unavailable.", False

def speak(text):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 180)
        engine.say(text)
        engine.runAndWait()
    except Exception as ex:
        print(f"Error: {ex}")

def get_download_link(file_path, text):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{file_path}">{text}</a>'
    return href

SHOWED = False

def create_pdf_report(analysis_data, image_file, history_type="text"):
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt=f"Skin Analysis Report ({history_type.capitalize()} Chat)", ln=1, align='C')
            pdf.ln(10)
            
            # Date
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 10, txt=f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
            pdf.ln(5)
            
            # Image
            if image_file:
                try:
                    with open("temp_image.jpg", "wb") as f:
                        f.write(image_file.getbuffer())
                    pdf.image("temp_image.jpg", w=50, h=50)
                    os.remove("temp_image.jpg")
                except:
                    pass
            
            # Analysis Results
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Analysis Results", ln=1)
            pdf.set_font("Arial", '', 12)
            
            for condition, prob in analysis_data['condition'].items():
                pdf.cell(200, 10, txt=f"- {condition}: {prob:.1%}", ln=1)
            pdf.cell(200, 10, txt=f"Skin Type: {analysis_data['skin_type']}", ln=1)
            
            # Summary
            if 'summary' in st.session_state:
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Summary", ln=1)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, txt=st.session_state.summary)
            
            report_path = f"skin_analysis_{history_type}_report.pdf"
            pdf.output(report_path)
            return report_path


@st.dialog("Talk with AI", width="large")
def voicechat():
    if "show_dialog" not in st.session_state:
        st.session_state.show_dialog = True

    st.image("loader.gif", use_container_width=True)
    speak("Hi")

    if st.button("Close"):
        st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_container = st.empty()

    while True:
        user_input, state = recognize_speech()
        
        if user_input.lower() == "stop" or user_input.strip() == "":
            st.write("Voice chat ended.")
            break
        
        if state:
            condition_probs = st.session_state.last_analysis['condition']
            skin_type_probs = st.session_state.last_analysis['skin_type']
            user_inputs = f"""
    Respond to this in English sentences with local flavor words:
    Question: {user_input}
    Skin Data: {condition_probs}, {skin_type_probs}
    
    Rules:
    1. Roman letters only
    2. Max 3 sentences
    3. Make it short
    4. Structure: Findings + Simple Advice
    5. Speak like friendly dermatologist
    """
            ai_response = get_gemini_response(user_inputs)
            st.session_state.chat_history.append(f"You: {user_input}")
            st.session_state.chat_history.append(f"AI: {ai_response}")

            speak(ai_response)

        with chat_container.container():
            with st.expander("View Chat History", expanded=True):
                for message in st.session_state.chat_history:
                    st.write(message)



# Chat interface
if 'last_analysis' in st.session_state:
    if "history" in st.session_state and st.session_state.history:
        st.subheader("📊 Conversation")
        for q, r in st.session_state.history:
            st.write(f"**User:** {q}")
            st.write(f"**Assistant:** {r}")

    col1, col2, col3, col4 = st.columns([5, 1, 1, 1], gap="small")  # Added gap between columns

    with col1:
        prompt = st.chat_input("Ask about your skin analysis...", key="chat_input")

    with col2:
        voice_button = st.button("🎙️", 
                            key="voice_button",
                            help="Start voice conversation")

    with col3:
        download_button = st.button("💾",  # Changed to floppy disk icon
                                help="Download chat history as text file")

    with col4:
        report_button = st.button("📊",  # Changed to chart icon
                                help="Generate PDF analysis report")
    
    if prompt:
        condition_probs = st.session_state.last_analysis['condition']
        skin_type_probs = st.session_state.last_analysis['skin_type']

        with st.spinner("💡 Reasoning the solution..."):
            response_prompt = f"Skin condition probabilities: {condition_probs}. Skin type probabilities: {skin_type_probs}. User question: {prompt}. You are a very skilled assistant in this field. Provide a well-structured and moderate-length response."
            response = get_gemini_response(response_prompt)

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append((prompt, response))
        
        st.rerun()

    if voice_button:
        voicechat()
        st.session_state.chat_history=[]

    if download_button:
        if "history" in st.session_state:
            chat_text = "\n".join([f"User: {q}\nAssistant: {r}" for q, r in st.session_state.history])
            
            # Encode the file content
            b64 = base64.b64encode(chat_text.encode()).decode()
            
            # Generate an invisible download link
            href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt"></a>'
            
            # Automatically trigger download using a dummy button click
            st.markdown(href, unsafe_allow_html=True)

            if "history" in st.session_state:
                chat_text = "\n".join([f"User: {q}\nAssistant: {r}" for q, r in st.session_state.history])
                st.download_button(label="Download Chat", data=chat_text, file_name="chat_history.txt", mime="text/plain")
                try:
                    chat_text = "\n".join([f"User: {q}\nAssistant: {r}" for q, r in st.session_state.history])
                    with open("text_chat_history.txt", "w") as f:
                        f.write(chat_text)
                    
                    # Auto-download the file
                    with open("text_chat_history.txt", "rb") as f:
                        data = f.read()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:file/txt;base64,{b64}" download="text_chat_history.txt" id="auto_download"></a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Add JavaScript to trigger the download automatically
                    st.components.v1.html(
                        """
                        <script>
                        document.getElementById('auto_download').click();
                        </script>
                        """
                    )
                except:pass


    if report_button:
        
        # Add this in your download buttons section
        if 'last_analysis' in st.session_state:
            # ... (other download buttons)
                # Create the PDF report
            report_path = create_pdf_report(
                st.session_state.last_analysis, 
                image_file, 
                "text"
            )
            
            # Read the generated PDF
            with open(report_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Create download button that auto-triggers
            st.download_button(
                label="⬇️ Click if download didn't start automatically",
                data=pdf_bytes,
                file_name="skin_analysis_report.pdf",
                mime="application/pdf",
                key="auto_download_pdf"
            )
            
            # Auto-trigger the download using JavaScript
            st.markdown("""
            <script>
            // Wait for the download button to be rendered
            setTimeout(function() {
                // Get the download button by its key
                const downloadButton = parent.document.querySelector(
                    'div[data-testid="stDownloadButton"] button'
                );
                // Click the button automatically
                if (downloadButton) {
                    downloadButton.click();
                }
            }, 500);
            </script>
            """, unsafe_allow_html=True)
            
        