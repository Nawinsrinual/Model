import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# ส่วนหัว
st.image("01.jpg", use_container_width=True)  # ใช้ use_container_width แทน use_column_width
st.header('🍛 Image Classification Model')
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        color: #4CAF50;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# แสดงเมนูที่โมเดลสามารถทำนายได้ในรูปแบบตาราง
st.subheader("เมนูที่ทำนายได้")

# รายการเมนู
menu_list = [
    'Khaoklukkapi (ข้าวคลุกกะปิ)', 'Fishcake (ทอดมัน)', 'Greencurry (แกงเขียวหวาน)', 'Kailoogkeay (ไข่ลูกเขย)', 
    'Khaomokkai (ข้าวหมกไก่)', 'Khoamunkai (ข้าวมันไก่)', 'Kungopwunsen (กุ้งอบวุ้นเส้น)', 'Kungpao (กุ้งเผา)', 
    'Moosatae (หมูสะเต๊ะ)', 'PadThai (ผัดไทย)', 'Padkrapao (ผัดกระเพรา)', 'Padpukbung (ผัดผักบุ้ง)', 
    'Palo (พะโล้)', 'Somtum (ส้มตำ)', 'Tomjued (ต้มจืด)', 'Tomjuedmara (ต้มจืดมะละ)', 
    'Tomkhakai (ต้มข่าไก่)', 'Tomyumkung (ต้มยำกุ้ง)'
]

# จัดข้อมูลเป็น HTML ตาราง
menu_table = """
<style>
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 16px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f4b41a;
        color: white;
    }
    tr:nth-child(even) {
        background-color: rgb(22, 178, 240);
    }
</style>
<table>
    <tr><th>Menu</th></tr>
""" + "".join(f"<tr><td>{menu}</td></tr>" for menu in menu_list) + "</table>"

st.markdown(menu_table, unsafe_allow_html=True)

# โหลดโมเดล
model = load_model('Image_classify2.keras')

# หมวดหมู่
data_cat = ['Khaoklukkapi', 'fishcake', 'greencurry', 'kailoogkeay', 'khaomokkai', 'khoamunkai', 
            'kungopwunsen', 'kungpao', 'moosatae', 'padThai', 'padkrapao', 'padpukbung', 
            'palo', 'somtum', 'tomjued', 'tomjuedmara', 'tomkhakai', 'tomyumkung']

# ขนาดของภาพ
img_height = 224
img_width = 224

# ลิงก์ข้อมูลเมนู
menu_links = {
    "Khaoklukkapi": "https://www.wongnai.com/recipes/rice-seasoned-with-shrimp-paste-recipe",
    "fishcake": "https://www.wongnai.com/recipes/fish-cake",
    "greencurry": "https://www.wongnai.com/recipes/green-curry",
    "kailoogkeay": "https://www.wongnai.com/recipes/deep-fried-duck-egg-with-sweet-sour-sauce",
    "khaomokkai": "https://www.wongnai.com/recipes/khao-mok-gai",
    "khoamunkai": "https://www.wongnai.com/recipes/hainanese-chicken-rice",
    "kungopwunsen": "https://www.wongnai.com/recipes/shrimp-potted-with-vermicelli",
    "kungpao": "https://www.wongnai.com/recipes/ugc/e17dade0126a438ea3470adf31af8c2f",
    "moosatae": "https://www.wongnai.com/recipes/pork-satay",
    "padThai": "https://www.wongnai.com/recipes/thai-fried-noodles-with-shrimp",
    "padkrapao": "https://cooking.kapook.com/view246668.html",
    "padpukbung": "https://www.wongnai.com/recipes/ugc/638b2833769c4a9f9b92ddd1e530b807",
    "palo": "https://www.wongnai.com/recipes/ugc/816e86a9a705491e91606b71b1c8c665",
    "somtum": "https://www.wongnai.com/food-tips/somtum-series",
    "tomjued": "https://www.wongnai.com/recipes/ugc/c082fa555e704ef98dc825f73dafd46a",
    "tomjuedmara": "https://www.sanook.com/women/176917/",
    "tomkhakai": "https://www.wongnai.com/recipes/tom-khai-gai",
    "tomyumkung": "https://www.wongnai.com/recipes/ugc/06c21e83ed584ed4996a129af5c3231c",
}

# ตัวเลือกการอัปโหลดภาพ
uploaded_file = st.file_uploader("อัปโหลดรูปภาพจากไฟล์...", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("หรือถ่ายภาพจากกล้อง")

# เลือกรูปภาพที่ใช้งาน (ถ้ามี)
image_file = uploaded_file if uploaded_file is not None else camera_file

# ถ้ามีการอัปโหลดหรือถ่ายภาพ
if image_file is not None:
    # แสดงภาพ
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)  # ใช้ use_container_width แทน use_column_width

    # ประมวลผลภาพ
    image = image.resize((img_height, img_width))
    img_arr = np.array(image)
    img_arr = np.expand_dims(img_arr, axis=0)

    # ทำนายผล
    predict = model.predict(img_arr)

    # ได้คะแนน
    score = tf.nn.softmax(predict)

    # แสดงผลการทำนาย
    predicted_class = data_cat[np.argmax(score)]

    # แสดงผล
    st.markdown(f'### 🥘 **Predicted Menu ( ผลการทำนาย คือ ) :** {predicted_class}')
    
    # แสดงลิงก์ข้อมูลเมนู
    if predicted_class in menu_links:
        menu_url = menu_links[predicted_class]
        st.markdown(f"[🔗 ข้อมูลเพิ่มเติมเกี่ยวกับเมนู {predicted_class}]( {menu_url} )")
    else:
        st.warning("⚠️ ไม่มีข้อมูลลิงก์สำหรับเมนูนี้")
else:
    st.warning("⚠️ กรุณาอัปโหลดรูปภาพหรือถ่ายภาพเพื่อเริ่มต้นการทำนาย")
