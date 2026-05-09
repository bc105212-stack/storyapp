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
    # Use a model specifically fine-tuned for story generation
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

def img2text(pil_image):
    model = load_caption_model()
    result = model(pil_image)
    return result[0]['generated_text']

def text2story(caption):
    model = load_story_model()
    prompt = f"Write a children's story based on: {caption}\n\nStory:"
    output = model(
        prompt,
        max_new_tokens=200,          # Enough for 200-250 words
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3
    )
    story = output[0]['generated_text']
    # Remove the prompt part if it remains
    if story.startswith(prompt):
        story = story[len(prompt):]
    story = story.strip()
    # If still too short, we can add a fallback, but usually it's fine
    return story

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Storytime for Kids", page_icon=None)
st.title("Picture-to-Story for Kids")
st.write("Upload a picture, and get a complete children's story (200-300 words) with audio.")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption='Your picture', use_column_width=True)

    with st.spinner("Looking at the picture..."):
        caption = img2text(pil_image)
    st.success(f"Caption: {caption}")

    with st.spinner("Writing a complete story (this may take 10-15 seconds)..."):
        story = text2story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button to listen to the story.")
