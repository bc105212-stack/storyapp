import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import re
from PIL import Image

# ------------------- Model Loaders -------------------
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    # Using standard GPT-2 (small but better than distilgpt2)
    # If memory is tight, change to "distilgpt2"
    return pipeline("text-generation", model="gpt2")

# ------------------- Clean & Filter -------------------
def clean_story(raw_story):
    """Remove first-person ramblings, placeholders, and unnatural text."""
    # Remove anything before the first real sentence (sometimes model outputs "____")
    raw_story = re.sub(r'^[^A-Za-z]*', '', raw_story)
    
    # Remove lines that contain blacklisted phrases
    blacklist = [
        r'I\s+', r'my\s+', r'opinion', r'Wikipedia', r'_____', r'___',
        r'personal', r'you see', r'let me', r'I think', r'I didn\'t know',
        r'future generations', r'more information'
    ]
    for pattern in blacklist:
        raw_story = re.sub(pattern, '', raw_story, flags=re.IGNORECASE)
    
    # Remove parentheticals
    raw_story = re.sub(r'\([^)]*\)', '', raw_story)
    # Remove extra spaces and newlines
    raw_story = re.sub(r'\s+', ' ', raw_story).strip()
    return raw_story

def truncate_to_sentence(text, max_words=180):
    """Truncate text to the last complete sentence within max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = ' '.join(words[:max_words])
    # Find last sentence boundary (. ! ?)
    match = re.search(r'(.*[.!?])\s', truncated)
    if match:
        return match.group(1)
    else:
        return truncated

def text2story(caption):
    model = load_story_model()
    
    # Natural story prompt
    prompt = f"Once upon a time, {caption}. One day, "
    
    output = model(
        prompt,
        max_new_tokens=160,          # Longer generation for completeness
        do_sample=True,
        temperature=0.65,             # Less random
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=50256
    )
    
    story = output[0]['generated_text']
    
    # Remove the prompt from the beginning
    if story.startswith(prompt):
        story = story[len(prompt):]
    else:
        # Fallback: remove the first sentence if it contains the prompt
        story = re.sub(r'^Once upon a time,.*?\.\s*', '', story)
    
    # Clean up weird content
    story = clean_story(story)
    
    # Ensure the story ends with a complete sentence
    story = truncate_to_sentence(story, max_words=180)
    
    # Force ending punctuation
    if story and story[-1] not in '.!?':
        story += '.'
    
    # If story is still too short (<30 words), add a generic happy ending
    if len(story.split()) < 30 and len(story) > 0:
        story += " It was a wonderful day, and everyone felt happy."
    
    return story.strip()

def img2text(pil_image):
    model = load_caption_model()
    result = model(pil_image)
    return result[0]['generated_text']

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ------------------- Streamlit UI (no icons) -------------------
st.set_page_config(page_title="Storytime for Kids", page_icon=None)
st.title("Picture-to-Story for Kids")
st.write("Upload any picture, and I'll create a fun, complete story with audio!")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption='Your picture', use_column_width=True)

    with st.spinner("Looking at the picture..."):
        caption = img2text(pil_image)
    st.success(f"Caption: {caption}")

    with st.spinner("Writing a complete story..."):
        story = text2story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words (story is complete)")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to hear the story read aloud.")
