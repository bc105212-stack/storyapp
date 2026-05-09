import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
from PIL import Image

# ------------------- Model Loaders (cached) -------------------
@st.cache_resource
def load_caption_model():
    """Load image-to-text pipeline from Hugging Face."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    """Load a text-to-text generation model that follows instructions (Flan-T5)."""
    return pipeline("text2text-generation", model="google/flan-t5-base")

# ------------------- Core Functions -------------------
def img2text(pil_image):
    """Generate a caption from a PIL Image object."""
    model = load_caption_model()
    result = model(pil_image)
    caption = result[0]['generated_text']
    return caption

def text2story(caption):
    """Expand the caption into a coherent children's story using Flan-T5."""
    model = load_story_model()
    # Explicit instruction to generate a short story for kids
    prompt = f"Write a short children's story (about 150 words) based on this description: {caption}"
    # Generate longer output (up to 250 new tokens, which yields ~150-200 words)
    output = model(prompt, max_new_tokens=250, do_sample=False, temperature=0.7)
    story = output[0]['generated_text']
    # Clean up any leading/trailing whitespace
    story = story.strip()
    return story

def text2audio(story_text):
    """Convert story text to MP3 audio using gTTS."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ------------------- Streamlit UI (no icons) -------------------
st.set_page_config(page_title="Storytime for Kids", page_icon=None)
st.title("Picture-to-Story for Kids")
st.write("Upload any picture, and I'll create a short children's story with audio!")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption='Your picture', use_column_width=True)

    with st.spinner("Looking at the picture..."):
        caption = img2text(pil_image)
    st.success(f"Caption: {caption}")

    with st.spinner("Writing a coherent story (about 150-200 words)..."):
        story = text2story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to listen to the story!")
