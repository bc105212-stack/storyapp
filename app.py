import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
from PIL import Image

@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    return pipeline("text-generation", model="gpt2")

def img2text(pil_image):
    model = load_caption_model()
    result = model(pil_image)
    caption = result[0]['generated_text']
    return caption

def text2story(caption):
    model = load_story_model()
    prompt = f"Once upon a time, {caption}. Write a short and complete story for kids in 50 to 100 words:"
    output = model(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2,
        early_stopping=True,
        pad_token_id=50256
    )
    story = output[0]['generated_text']
    if story.startswith(prompt):
        story = story[len(prompt):]
    story = story.strip()
    if story and story[-1] not in '.!?':
        story += '.'
    words = story.split()
    if len(words) > 100:
        story = ' '.join(words[:100])
        if story[-1] not in '.!?':
            story += '...'
    return story

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

st.set_page_config(page_title="Storytime for Kids", page_icon=None)
st.title("Picture-to-Story for Kids")
st.write("Upload any picture, and I'll create a short 50-100 word story with audio!")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption='Your picture', use_column_width=True)

    with st.spinner("Looking at the picture..."):
        caption = img2text(pil_image)
    st.success(f"Caption: {caption}")

    with st.spinner("Writing a story (50-100 words)..."):
        story = text2story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to listen to the story!")
