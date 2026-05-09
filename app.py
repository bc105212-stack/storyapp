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
    # Specialized story generation model (fine-tuned on stories)
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

# ------------------- Story Generation with Retry -------------------
def is_bad_story(text):
    """Detect if the story contains gibberish or first-person meta comments."""
    bad_phrases = [
        "i didn't know", "i think", "in my opinion", "wikipedia",
        "this is a test", "you see what happened", "future generations"
    ]
    lower = text.lower()
    for phrase in bad_phrases:
        if phrase in lower:
            return True
    # Also if story ends with punctuation but has no space after period? Not needed.
    # Detect if story is very short or starts with a comma
    if len(text.split()) < 10:
        return True
    if text.startswith(',') or text.startswith('.'):
        return True
    return False

def generate_clean_story(caption, retries=2):
    """Generate story and retry if output is bad."""
    model = load_story_model()
    # Strong prompt that asks for a short children's story
    prompt = f"Write a short story for children about {caption}\nStory: "
    
    for attempt in range(retries + 1):
        # Adjust randomness: lower temperature on retry
        temp = 0.6 if attempt == 0 else 0.4
        output = model(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=50256
        )
        story = output[0]['generated_text']
        # Remove prompt if present
        if story.startswith(prompt):
            story = story[len(prompt):]
        # Clean up line breaks and extra spaces
        story = re.sub(r'\s+', ' ', story).strip()
        
        # Ensure ending punctuation
        if story and story[-1] not in '.!?':
            story += '.'
        
        # Check quality
        if not is_bad_story(story):
            # Truncate to last sentence within 100 words
            words = story.split()
            if len(words) > 100:
                truncated = ' '.join(words[:100])
                match = re.search(r'(.*[.!?])\s', truncated)
                if match:
                    story = match.group(1)
            return story
    
    # If all attempts fail, craft a safe default story
    return f"Once upon a time, {caption}. It was a beautiful day full of joy and laughter. The end."

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
st.write("Upload any picture, and I'll create a short 50-100 word story with audio!")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption='Your picture', use_column_width=True)

    with st.spinner("Looking at the picture..."):
        caption = img2text(pil_image)
    st.success(f"Caption: {caption}")

    with st.spinner("Writing a coherent story (50-100 words)..."):
        story = generate_clean_story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to listen to the story!")
