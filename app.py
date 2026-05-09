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
    # Use distilgpt2 with repetition penalty and no repeat ngram to avoid loops
    return pipeline("text-generation", model="distilgpt2")

# ------------------- Core Functions -------------------
def img2text(pil_image):
    model = load_caption_model()
    result = model(pil_image)
    return result[0]['generated_text']

def text2story(caption):
    model = load_story_model()
    # Clean caption: remove extra spaces and ensure it's a proper sentence
    caption = caption.strip().capitalize()
    prompt = f"Write a short children's story about {caption}. "
    output = model(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=50256
    )
    story = output[0]['generated_text']
    # Remove prompt if repeated
    if story.startswith(prompt):
        story = story[len(prompt):]
    # Cut at first occurrence of "The end" or period+newline for cleanliness
    # Also limit to first 200 words
    words = story.split()
    if len(words) > 150:
        story = ' '.join(words[:150]) + "..."
    # Ensure story ends with a period
    if not story.endswith(('.', '!', '?')):
        story += '.'
    return story

def text2audio(story_text):
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

    with st.spinner("Writing a story..."):
        story = text2story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to listen to the story!")
