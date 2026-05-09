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
    # Using distilgpt2 - more stable and less prone to nonsense
    return pipeline("text-generation", model="distilgpt2")

# ------------------- Helper: Clean Story -------------------
def clean_story(raw_story):
    """Remove unwanted phrases and ensure story is coherent."""
    # Remove anything after a line break or unnatural long pauses
    raw_story = raw_story.split('\n')[0]
    # Remove parenthetical expressions (like (I mean...))
    raw_story = re.sub(r'\([^)]*\)', '', raw_story)
    # List of blacklisted phrases (adult/unrelated)
    blacklist = [
        r'Wikipedia', r'opinion', r'personal opinion', r'you see what happened',
        r'future generations', r'more information', r'Well there is always hope'
    ]
    for pattern in blacklist:
        raw_story = re.sub(pattern, '', raw_story, flags=re.IGNORECASE)
    # Trim extra spaces
    raw_story = re.sub(r'\s+', ' ', raw_story).strip()
    return raw_story

def text2story(caption):
    """Generate a clean 50-100 word story from the image caption."""
    model = load_story_model()
    
    # Explicit prompt that asks for a short children's story
    prompt = f"Caption: {caption}\nWrite a short children's story (50-100 words):\nStory: "
    
    output = model(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        pad_token_id=50256  # eos token for distilgpt2
    )
    
    story = output[0]['generated_text']
    
    # Remove the prompt from the beginning if present
    if story.startswith(prompt):
        story = story[len(prompt):]
    
    # Clean up unwanted content
    story = clean_story(story)
    
    # Ensure we end at a complete sentence within ~100 words
    words = story.split()
    if len(words) > 100:
        truncated = ' '.join(words[:100])
        # Find last sentence boundary
        match = re.search(r'(.*[.!?])\s', truncated)
        if match:
            story = match.group(1)
        else:
            story = truncated
    
    # Force ending punctuation
    if story and story[-1] not in '.!?':
        story += '.'
    
    # If story is strangely short (<20 words), append a generic conclusion
    if len(words) < 20 and len(story) > 0:
        story += " It was a happy day for everyone."
    
    return story

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
        story = text2story(caption)
    st.subheader("Your Story")
    st.write(story)
    word_count = len(story.split())
    st.caption(f"Word count: {word_count} words")

    with st.spinner("Creating voice audio..."):
        audio_path = text2audio(story)
    st.audio(audio_path, format='audio/mp3')
    st.info("Click the play button above to listen to the story!")
