import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
from PIL import Image

# ------------------- Model Loaders (cached) -------------------
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    # Use flan-t5-small for speed and better long-output capability
    return pipeline("text2text-generation", model="google/flan-t5-small")

# ------------------- Core Functions -------------------
def img2text(pil_image):
    model = load_caption_model()
    result = model(pil_image)
    caption = result[0]['generated_text']
    return caption

def text2story(caption):
    model = load_story_model()
    prompt = f"Write a long children's story (at least 150 words) based on this description: {caption}"
    output = model(
        prompt,
        max_new_tokens=300,
        min_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        early_stopping=False,
        no_repeat_ngram_size=3
    )
    story = output[0]['generated_text'].strip()
    # If still too short (less than 50 words), generate a continuation
    if len(story.split()) < 50:
        continuation = model(
            story + " Then,",
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8
        )[0]['generated_text'].strip()
        story = story + " " + continuation
    return story

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ------------------- Streamlit UI (no icons) -------------------
st.set_page_config(page_title="Storytime for Kids", page_icon=None)
st.title("Picture-to-Story for Kids")
st.write("Upload any picture, and I'll create a long children's story with audio!")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption='Your picture', use_column_width=True)

    with st.spinner("Looking at the picture..."):
        caption = img2text(pil_image)
    st.success(f"Caption: {caption}")

    with st.spinner("Writing a long story (min. 150 words)..."):
        story = text2story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to listen to the story!")
