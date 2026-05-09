import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
from PIL import Image

# ------------------- Model Loaders (cached) -------------------
@st.cache_resource
def load_caption_model():
    """Image captioning model (BLIP)."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    """GPT-2 model for text generation."""
    return pipeline("text-generation", model="gpt2")

# ------------------- Core Functions -------------------
def img2text(pil_image):
    model = load_caption_model()
    result = model(pil_image)
    return result[0]['generated_text']

def text2story(caption):
    model = load_story_model()
    # Strong, explicit prompt for a children's story
    prompt = (
        f"Write a short, fun children's story (about 150 words) based on this description: "
        f"'{caption}'. The story should have a beginning, middle, and end. "
        f"Use simple words for kids. Start the story now:\n\n"
    )
    output = model(
        prompt,
        max_new_tokens=250,          # enough for ~180-200 words
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.1,
        pad_token_id=50256           # GPT-2's eos token
    )
    story = output[0]['generated_text']
    # Remove the prompt from the beginning if present
    if story.startswith(prompt):
        story = story[len(prompt):]
    # Clean up starting whitespace or incomplete sentences
    story = story.strip()
    # Ensure story is not extremely short (less than 30 words) -> fallback
    if len(story.split()) < 30:
        story = f"{caption}. The children were very happy and played all day. They made new friends and learned to share. Everyone went home with a big smile on their face. What a wonderful day it was!"
    return story

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ------------------- Streamlit UI (no icons) -------------------
st.set_page_config(page_title="Storytime for Kids", page_icon=None)
st.title("Picture-to-Story for Kids")
st.write("Upload a picture and get a complete children's story with audio.")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption='Your picture', use_column_width=True)

    with st.spinner("Looking at the picture..."):
        caption = img2text(pil_image)
    st.success(f"Caption: {caption}")

    with st.spinner("Writing a complete story (about 150-200 words)..."):
        story = text2story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to listen to the story.")
