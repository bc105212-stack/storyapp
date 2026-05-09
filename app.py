import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile

st.set_page_config(page_title="儿童图片故事工厂", page_icon="📖")
st.title("📸 图片故事生成器（适合3-10岁）")
st.write("上传一张图片，我会为你编一个有趣的故事，并且读给你听！")

# 使用更小、下载更快的模型
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="ydshieh/vit-gpt2-coco-en")

@st.cache_resource
def load_story_model():
    return pipeline("text-generation", model="distilgpt2")

caption_model = load_caption_model()
story_model = load_story_model()

def img2text(image):
    result = caption_model(image)
    return result[0]['generated_text']

def text2story(prompt_text):
    full_prompt = f"Once upon a time, {prompt_text}. Then,"
    output = story_model(full_prompt, max_new_tokens=120, do_sample=True, temperature=0.8)
    story = output[0]['generated_text']
    if story.startswith(full_prompt):
        story = story[len(full_prompt):].strip()
    words = story.split()
    if len(words) < 50:
        story += " They all lived happily ever after. The end."
    elif len(words) > 100:
        story = " ".join(words[:100])
    return story

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        temp_path = f.name
    tts.save(temp_path)
    return temp_path

uploaded_file = st.file_uploader("点击上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="你上传的图片", use_column_width=True)
    
    with st.spinner("观察图片中..."):
        caption = img2text(image)
        st.info(f"我看到：{caption}")
    
    with st.spinner("编故事中..."):
        story = text2story(caption)
        st.success("故事来啦：")
        st.write(story)
        st.caption(f"字数：{len(story.split())} 词")
    
    with st.spinner("生成语音中..."):
        audio_file = text2audio(story)
        st.audio(audio_file, format="audio/mp3")
else:
    st.info("请先上传一张图片")