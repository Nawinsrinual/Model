import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# =======================
# CSS พื้นหลัง + ปุ่มอัปโหลดชัด
# =======================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.postimg.cc/c4TXtncG/fall-nature-background-with-leaves-picjumbo-com.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #222222;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}

.main-header {
    font-size: 32px;
    color: #222222;
    text-align: center;
}

/* ปรับปุ่ม Browse file และ Take photo */
[data-testid="stFileUploader"] button,
[data-testid="stCameraInput"] button {
    background: linear-gradient(90deg, #f48c06, #f4b41a) !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    border: none !important;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.2) !important;
    transition: 0.3s;
}

[data-testid="stFileUploader"] button:hover,
[data-testid="stCameraInput"] button:hover {
    background: linear-gradient(90deg, #f4a261, #e76f51) !important;
    transform: scale(1.05);
}

/* ตารางเมนู */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 16px;
    font-family: 'Arial', sans-serif;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    border-radius: 10px;
    overflow: hidden;
}

th, td {
    padding: 12px 15px;
    text-align: left;
    color: #222222;
}

th {
    background: linear-gradient(90deg, #f4b41a, #f48c06);
    color: white;
    font-size: 18px;
}

tr:nth-child(even) {
    background-color: rgba(255,255,255,0.8);
}

tr:nth-child(odd) {
    background-color: rgba(255,255,255,0.6);
}

tr:hover {
    background-color: rgba(208,240,253,0.8);
    transition: 0.3s;
}

td {
    border-bottom: 1px solid #ddd;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =======================
# ส่วนหัวของหน้าเว็บ
# =======================
st.image("01.jpg", use_container_width=True)
st.header('🍛 Image Classification Model')

# =======================
# แสดงเมนูที่โมเดลสามารถทำนายได้
# =======================
st.subheader("เมนูที่ทำนายได้")
menu_list = [
    'Khaoklukkapi (ข้าวคลุกกะปิ)', 'Fish Cake (ทอดมันปลา)', 'Green Curry (แกงเขียวหวาน)', 'Kai Look Keaw (ไข่ลูกเขย)', 
    'Khao Mok Gai (ข้าวหมกไก่)', 'Khao Mun Gai (ข้าวมันไก่)', 'Kung Ob Wunsen (กุ้งอบวุ้นเส้น)', 'Kung Pao (กุ้งเผา)', 
    'Moo Satay (หมูสะเต๊ะ)', 'Pad Thai (ผัดไทย)', 'Pad Krapao (ผัดกะเพรา)', 'Pad Pak Bung (ผัดผักบุ้ง)', 
    'Palo (พะโล้)', 'Som Tum (ส้มตำ)', 'Tom Jued (ต้มจืด)', 'Tom Jued Mara (ต้มจืดมะระ)', 
    'Tom Kha Kai (ต้มข่าไก่)', 'Tom Yum Kung (ต้มยำกุ้ง)'
]

menu_table = """
<table>
<tr><th>English Name</th><th>Thai Name</th></tr>
""" + "".join(f"<tr><td>{menu.split(' (')[0]}</td><td>{menu.split(' (')[1].replace(')','')}</td></tr>" for menu in menu_list) + "</table>"

st.markdown(menu_table, unsafe_allow_html=True)

# =======================
# โหลดโมเดลและกำหนดหมวดหมู่
# =======================
model = load_model('thai_food_model.keras')
data_cat = [
    'khaoklukkapi', 'fishcake', 'greencurry', 'khailookkeaw', 'khaomokkai', 'khaomunkai',
    'kungobwunsen', 'kungpao', 'moosatay', 'padthai', 'padkrapao', 'padpakbung',
    'palo', 'somtum', 'tomjude', 'tomjudmara', 'tomkhakai', 'tomyumkung'
]
img_height = 224
img_width = 224

# =======================
# ลิงก์ข้อมูลเมนู
# =======================
menu_links = {
    "khaoklukkapi": ("ข้าวคลุกกะปิ", "https://www.wongnai.com/recipes/rice-seasoned-with-shrimp-paste-recipe"),
    "fishcake": ("ทอดมันปลา", "https://www.wongnai.com/recipes/fish-cake"),
    "greencurry": ("แกงเขียวหวาน", "https://www.wongnai.com/recipes/green-curry"),
    "khailookkeaw": ("ไข่ลูกเขย", "https://www.wongnai.com/recipes/deep-fried-duck-egg-with-sweet-sour-sauce"),
    "khaomokkai": ("ข้าวหมกไก่", "https://www.wongnai.com/recipes/khao-mok-gai"),
    "khaomunkai": ("ข้าวมันไก่", "https://www.wongnai.com/recipes/hainanese-chicken-rice"),
    "kungobwunsen": ("กุ้งอบวุ้นเส้น", "https://www.wongnai.com/recipes/shrimp-potted-with-vermicelli"),
    "kungpao": ("กุ้งเผา", "https://www.wongnai.com/recipes/ugc/e17dade0126a438ea3470adf31af8c2f"),
    "moosatay": ("หมูสะเต๊ะ", "https://www.wongnai.com/recipes/pork-satay"),
    "padthai": ("ผัดไทย", "https://www.wongnai.com/recipes/thai-fried-noodles-with-shrimp"),
    "padkrapao": ("ผัดกะเพรา", "https://cooking.kapook.com/view246668.html"),
    "padpakbung": ("ผัดผักบุ้ง", "https://www.wongnai.com/recipes/ugc/638b2833769c4a9f9b92ddd1e530b807"),
    "palo": ("พะโล้", "https://www.wongnai.com/recipes/ugc/816e86a9a705491e91606b71b1c8c665"),
    "somtum": ("ส้มตำ", "https://www.wongnai.com/food-tips/somtum-series"),
    "tomjude": ("ต้มจืด", "https://www.wongnai.com/recipes/ugc/c082fa555e704ef98dc825f73dafd46a"),
    "tomjudmara": ("ต้มจืดมะระ", "https://www.sanook.com/women/176917/"),
    "tomkhakai": ("ต้มข่าไก่", "https://www.wongnai.com/recipes/tom-khai-gai"),
    "tomyumkung": ("ต้มยำกุ้ง", "https://www.wongnai.com/recipes/ugc/06c21e83ed584ed4996a129af5c3231c"),
}

# =======================
# ตัวเลือกการอัปโหลดและถ่ายภาพ
# =======================
uploaded_file = st.file_uploader("อัปโหลดรูปภาพจากไฟล์...", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("หรือถ่ายภาพจากกล้อง")
image_file = uploaded_file if uploaded_file is not None else camera_file

# =======================
# ประมวลผลและทำนาย
# =======================
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    image = image.resize((img_height, img_width))
    img_arr = np.expand_dims(np.array(image), axis=0)

    predict = model.predict(img_arr)
    score = tf.nn.softmax(predict)
    predicted_class = data_cat[np.argmax(score)]

    if predicted_class in menu_links:
        thai_name, url = menu_links[predicted_class]
        st.markdown(f'### 🥘 **Predicted Menu (ผลการทำนาย):** {predicted_class} ({thai_name})')
        st.markdown(f"[🔗 ข้อมูลเพิ่มเติมเกี่ยวกับเมนู {thai_name}]({url})")
    else:
        st.warning("⚠️ ไม่มีข้อมูลลิงก์สำหรับเมนูนี้")
else:
    st.warning("⚠️ กรุณาอัปโหลดรูปภาพหรือถ่ายภาพเพื่อเริ่มต้นการทำนาย")
