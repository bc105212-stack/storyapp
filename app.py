import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import re
from PIL import Image

# ------------------- Model Loaders (cached) -------------------
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    return pipeline("text-generation", model="gpt2")

# ------------------- Core Functions -------------------
def img2text(pil_image):
    model = load_caption_model()
    result = model(pil_image)
    caption = result[0]['generated_text']
    return caption

def text2story(caption):
    """
    Generate a complete 50-100 word story from a caption.
    Improved generation: longer output, end-of-sentence trimming.
    """
    model = load_story_model()
    
    # Better prompt that encourages a natural story arc
    prompt = f"Once upon a time, {caption}. "
    
    # Generate longer output (up to 150 tokens), then trim to whole sentences near 100 words
    output = model(
        prompt,
        max_new_tokens=150,          # Generate enough tokens
        do_sample=True,
        temperature=0.8,
        top_p=0.9,                   # Nucleus sampling for coherence
        repetition_penalty=1.2       # Avoid repetitive loops
    )
    
    raw_story = output[0]['generated_text']
    
    # Remove the prompt if it appears at the beginning
    if raw_story.startswith(prompt):
        raw_story = raw_story[len(prompt):]
    
    # Truncate to the last complete sentence within ~100 words
    words = raw_story.split()
    if len(words) > 100:
        # Find the last sentence boundary (., !, ?) within the first 100 words
        truncated = ' '.join(words[:100])
        # Find the last punctuation that ends a sentence
        match = re.search(r'(.*[.!?])\s', truncated)
        if match:
            raw_story = match.group(1)
        else:
            raw_story = truncated  # Fallback: just truncate without cutting words
    
    # Ensure story ends with a proper punctuation
    if raw_story and raw_story[-1] not in '.!?':
        raw_story += '.'
    
    return raw_story.strip()

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# ------------------- Streamlit UI (no icons) -------------------
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
