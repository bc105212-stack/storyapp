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
    return pipeline("text-generation", model="gpt2", device=-1)  # cpu

# ------------------- Core Functions -------------------
def img2text(pil_image):
    model = load_caption_model()
    result = model(pil_image)
    caption = result[0]['generated_text']
    return caption

def text2story(caption):
    model = load_story_model()
    prompt = f"Once upon a time, {caption}. "
    output = model(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=50256,
        eos_token_id=50256
    )
    story = output[0]['generated_text']
    # Remove prompt if repeated
    if story.startswith(prompt):
        story = story[len(prompt):]
    # Stop at a period or newline to avoid cut off mid-sentence? not needed
    # Remove excessive repetition by truncating after repeated phrases
    # Simple heuristic: if same sentence repeated, cut after first occurrence
    lines = story.split('.')
    unique_lines = []
    for line in lines:
        if line.strip() and line.strip() not in unique_lines:
            unique_lines.append(line.strip())
    if unique_lines:
        story = '. '.join(unique_lines[:8]) + '.'  # limit to 8 sentences
    # Ensure story ends with a period
    if not story.endswith('.'):
        story += '.'
    return story

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ------------------- Streamlit UI -------------------
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
