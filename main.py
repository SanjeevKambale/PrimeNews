import streamlit as st
import requests
import re
import joblib



# --------------- CONFIG ---------------- #
st.set_page_config(page_title="PrimeNews - Fake News Predictor", layout="centered")



# Embed background image as base64 for reliability
import base64
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpg;base64,{encoded}');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .block-container {{
            padding-top: 2rem;
        }}
        .title-left {{
            font-size: 40px;
            text-align: left;
            margin-bottom: 30px;
            font-weight: bold;
            color: #4A90E2;
            margin-left: 0;
            margin-top: 0.5em;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path
set_bg_from_local("bg_news_img.jpg")






# --------------- FAKE NEWS PREDICTION ONLY ---------------- #
st.markdown('<div class="title-left">🔮 PrimeNews - Fake News Predictor</div>', unsafe_allow_html=True)


# Use pre-trained models and vectorizer for fast loading
import joblib


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

@st.cache_resource
def load_models():
    vectorizer = joblib.load(r'C:\MyProject\PrimeNews\Models\vectorizer.joblib')
    model_lr = joblib.load(r'C:\MyProject\PrimeNews\Models\lr_model.joblib')
    model_dtc = joblib.load(r'C:\MyProject\PrimeNews\Models\dtc_model.joblib')
    return vectorizer, model_lr, model_dtc

vectorizer, model_lr, model_dtc = load_models()


st.subheader("Enter the news article text below:")
if 'custom_text' not in st.session_state:
    st.session_state.custom_text = ""

def clear_custom_text():
    st.session_state.custom_text = ""

user_input = st.text_area("Paste news article here", value=st.session_state.custom_text, key="custom_text")
col1, col2 = st.columns([1,1])
with col1:
    predict_clicked = st.button("Predict")
with col2:
    refresh_clicked = st.button("Refresh Input", on_click=clear_custom_text)

if predict_clicked:
    if user_input:
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        prediction_lr = model_lr.predict(vec_input)[0]
        prediction_dtc = model_dtc.predict(vec_input)[0]
        if prediction_lr == 1 and prediction_dtc == 1:
            st.success("✅ It might be TRUE. BOTH models predict it's TRUE")
        elif prediction_lr == 1 and prediction_dtc == 0:
            st.success("✅ It might be TRUE. LR_MODEL predicts it's TRUE")
        elif prediction_lr == 0 and prediction_dtc == 1:
            st.success("✅ It might be TRUE. DTC_MODEL predicts it's TRUE")
        else:
            st.error("❌ It might be FAKE. Please verify it.")
    else:
        st.warning("Please enter some text to analyze.")

#-----------------------------Footer------------------------------------------
st.markdown("""
    <style>
    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 2em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="footer">© 2025 PrimeNews • Built by PrimeDev with ❤️ using Streamlit</div>', unsafe_allow_html=True)