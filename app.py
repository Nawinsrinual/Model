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
    "khaoklukkapi": ("ข้าวคลุกกะปิ", "https://projectclassification.com/%e0%b8%82%e0%b9%89%e0%b8%b2%e0%b8%a7%e0%b8%84%e0%b8%a5%e0%b8%b8%e0%b8%81%e0%b8%81%e0%b8%b0%e0%b8%9b%e0%b8%b4-rice-seasoned-with-shrimp-paste-khaoklukkapi/"),
    "fishcake": ("ทอดมันปลา", "https://projectclassification.com/%e0%b8%97%e0%b8%ad%e0%b8%94%e0%b8%a1%e0%b8%b1%e0%b8%99%e0%b8%9b%e0%b8%a5%e0%b8%b2-thai-fish-cake-fishcake/"),
    "greencurry": ("แกงเขียวหวาน", "https://projectclassification.com/%e0%b9%81%e0%b8%81%e0%b8%87%e0%b9%80%e0%b8%82%e0%b8%b5%e0%b8%a2%e0%b8%a7%e0%b8%ab%e0%b8%a7%e0%b8%b2%e0%b8%99-green-curry-greencurry-2/"),
    "khailookkeaw": ("ไข่ลูกเขย", "https://projectclassification.com/%e0%b9%84%e0%b8%82%e0%b9%88%e0%b8%a5%e0%b8%b9%e0%b8%81%e0%b9%80%e0%b8%82%e0%b8%a2-son-in-law-eggs-kailoogkeay/"),
    "khaomokkai": ("ข้าวหมกไก่", "https://projectclassification.com/%e0%b8%82%e0%b9%89%e0%b8%b2%e0%b8%a7%e0%b8%ab%e0%b8%a1%e0%b8%81%e0%b9%84%e0%b8%81%e0%b9%88-thai-chicken-biryani-khaomokkai/"),
    "khaomunkai": ("ข้าวมันไก่", "https://projectclassification.com/%e0%b8%82%e0%b9%89%e0%b8%b2%e0%b8%a7%e0%b8%a1%e0%b8%b1%e0%b8%99%e0%b9%84%e0%b8%81%e0%b9%88-hainanese-chicken-rice-khoamunkai/"),
    "kungobwunsen": ("กุ้งอบวุ้นเส้น", "https://projectclassification.com/%e0%b8%81%e0%b8%b8%e0%b9%89%e0%b8%87%e0%b8%ad%e0%b8%9a%e0%b8%a7%e0%b8%b8%e0%b9%89%e0%b8%99%e0%b9%80%e0%b8%aa%e0%b9%89%e0%b8%99-shrimp-potted-with-vermicelli-kungopwunsen/"),
    "kungpao": ("กุ้งเผา", "https://projectclassification.com/%e0%b8%81%e0%b8%b8%e0%b9%89%e0%b8%87%e0%b9%80%e0%b8%9c%e0%b8%b2-grilled-river-prawns-kungpao/"),
    "moosatay": ("หมูสะเต๊ะ", "https://projectclassification.com/%e0%b8%ab%e0%b8%a1%e0%b8%b9%e0%b8%aa%e0%b8%b0%e0%b9%80%e0%b8%95%e0%b9%8a%e0%b8%b0-pork-satay-moosatae/"),
    "padthai": ("ผัดไทย", "https://projectclassification.com/%e0%b8%9c%e0%b8%b1%e0%b8%94%e0%b9%84%e0%b8%97%e0%b8%a2-pad-thai-padthai/"),
    "padkrapao": ("ผัดกะเพรา", "https://projectclassification.com/%e0%b8%9c%e0%b8%b1%e0%b8%94%e0%b8%81%e0%b8%b0%e0%b9%80%e0%b8%9e%e0%b8%a3%e0%b8%b2-stir-fried-basil-padkrapao/"),
    "padpakbung": ("ผัดผักบุ้ง", "https://projectclassification.com/%e0%b8%9c%e0%b8%b1%e0%b8%94%e0%b8%9c%e0%b8%b1%e0%b8%81%e0%b8%9a%e0%b8%b8%e0%b9%89%e0%b8%87%e0%b9%84%e0%b8%9f%e0%b9%81%e0%b8%94%e0%b8%87-stir-fried-morning-glory-padpukbung/"),
    "palo": ("พะโล้", "https://projectclassification.com/%e0%b8%9e%e0%b8%b0%e0%b9%82%e0%b8%a5%e0%b9%89-five-spice-egg-and-pork-stew-palo/"),
    "somtum": ("ส้มตำ", "https://projectclassification.com/%e0%b8%aa%e0%b9%89%e0%b8%a1%e0%b8%95%e0%b8%b3-papaya-salad-somtum/"),
    "tomjude": ("ต้มจืด", "https://projectclassification.com/%e0%b8%95%e0%b9%89%e0%b8%a1%e0%b8%88%e0%b8%b7%e0%b8%94-clear-soup-tomjued/"),
    "tomjudmara": ("ต้มจืดมะระ", "https://projectclassification.com/%e0%b8%95%e0%b9%89%e0%b8%a1%e0%b8%88%e0%b8%b7%e0%b8%94%e0%b8%a1%e0%b8%b0%e0%b8%a3%e0%b8%b0-bitter-melon-soup-tomjuedmara/"),
    "tomkhakai": ("ต้มข่าไก่", "https://projectclassification.com/%e0%b8%95%e0%b9%89%e0%b8%a1%e0%b8%82%e0%b9%88%e0%b8%b2%e0%b9%84%e0%b8%81%e0%b9%88-galangal-chicken-soup-tomkhakai/"),
    "tomyumkung": ("ต้มยำกุ้ง", "https://projectclassification.com/%e0%b8%95%e0%b9%89%e0%b8%a1%e0%b8%a2%e0%b8%b3%e0%b8%81%e0%b8%b8%e0%b9%89%e0%b8%87-tom-yum-goong-tomyumkung/"),
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
