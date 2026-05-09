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
    """Load text-generation pipeline (GPT-2 for short stories)."""
    return pipeline("text-generation", model="gpt2")

# ------------------- Core Functions -------------------
def img2text(image_file):
    """Generate a caption from the uploaded image."""
    model = load_caption_model()
    # pipeline can accept a PIL Image directly
    result = model(image_file)
    caption = result[0]['generated_text']
    return caption

def text2story(caption):
    """Expand the caption into a 50-100 word children's story."""
    model = load_story_model()
    prompt = f"Once upon a time, {caption}. "
    # Generate ~80 new tokens (each token ≈ 0.75 words in English)
    output = model(prompt, max_new_tokens=80, do_sample=True, temperature=0.8)
    story = output[0]['generated_text']
    # Remove the prompt if it appears at the beginning
    if story.startswith(prompt):
        story = story[len(prompt):]
    # Enforce maximum 100 words
    words = story.split()
    if len(words) > 100:
        story = ' '.join(words[:100]) + '...'
    return story

def text2audio(story_text):
    """Convert story text to MP3 audio using gTTS."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    # Save to a temporary file and return its path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Storytime for Kids", page_icon="📖")
st.title("📸 Picture-to-Story for Kids")
st.write("Upload any picture, and I'll create a short 50‑100 word story with audio!")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Your picture', use_column_width=True)

    # Step 1: Image captioning
    with st.spinner("👀 Looking at the picture..."):
        caption = img2text(uploaded_file)
    st.success(f"Caption: {caption}")

    # Step 2: Story generation
    with st.spinner("✍️ Writing a story (50‑100 words)..."):
        story = text2story(caption)
    st.subheader("📖 Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    # Step 3: Text-to-speech
    with st.spinner("🔊 Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to listen to the story!")
