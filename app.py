import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Header
st.header('Image Classification Model')

# แสดงเมนูที่โมเดลสามารถทำนายได้ในแนวนอน
st.subheader("เมนูที่ทำนายได้")

# รายการเมนู
menu_list = [
    'Khaoklukkapi (ข้าวคลุกกะปิ)', 'Fishcake (ทอดมัน)', 'Greencurry (แกงเขียวหวาน)', 'Kailoogkeay (ไข่ลูกเขย)', 'Khaomokkai (ข้าวหมกไก่)', 'Khoamunkai (ข้าวมันไก่)',
    'Kungopwunsen (กุ้งอบวุ้นเส้น)', 'Kungpao (กุ้งเผา)', 'Moosatae (หมูสะเต๊ะ)', 'PadThai (ผัดไทย)', 'Padkrapao (ผัดกระเพรา)', 'Padpukbung (ผัดผักบุ้ง)', 'Palo (พะโล้)',
    'Somtum (ส้มตำ)', 'Tomjued (ต้มจืด)', 'Tomjuedmara (ต้มจืดมะละ)', 'Tomkhakai (ต้มข่าไก่)', 'Tomyumkung (ต้มยำกุ้ง)'
]

# สร้างคอลัมน์ 3 คอลัมน์เพื่อแบ่งเมนู
num_columns = 3
columns = st.columns(num_columns)

# ใช้การวนลูปเพื่อแสดงเมนูในแต่ละคอลัมน์
for i, menu in enumerate(menu_list):
    col = columns[i % num_columns]  # เลือกคอลัมน์ตามลำดับ
    col.write(menu)

# โหลดโมเดล
model = load_model('Image_classify2.keras')

# หมวดหมู่
data_cat = ['Khaoklukkapi', 'fishcake', 'greencurry', 'kailoogkeay', 'khaomokkai', 'khoamunkai', 'kungopwunsen', 'kungpao', 'moosatae', 'padThai', 'padkrapao', 'padpukbung', 'palo', 'somtum', 'tomjued', 'tomjuedmara', 'tomkhakai', 'tomyumkung']

# ขนาดของภาพ
img_height = 224
img_width = 224

# การแจ้งเตือนเกี่ยวกับการใช้งานกล้อง
st.info("⚠️ กรุณาอนุญาตการเข้าถึงกล้องหากต้องการถ่ายภาพ และใช้งานแอปนี้ผ่าน HTTPS หรือ localhost เพื่อความปลอดภัย")

# ตัวเลือกการอัปโหลดภาพ
uploaded_file = st.file_uploader("อัปโหลดรูปภาพจากไฟล์...", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("หรือถ่ายภาพจากกล้อง")

# เลือกรูปภาพที่ใช้งาน (ถ้ามี)
image_file = uploaded_file if uploaded_file is not None else camera_file

# ถ้ามีการอัปโหลดหรือถ่ายภาพ
if image_file is not None:
    # แสดงภาพ
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

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
    accuracy = np.max(score) * 100

    # แสดงผล
    st.write(f'**Predicted Menu:** {predicted_class}')
    st.write(f'**Prediction Accuracy:** {accuracy:.2f}%')

# หากไม่มีการอัปโหลดหรือถ่ายภาพ
else:
    st.warning("⚠️ กรุณาอัปโหลดรูปภาพหรือถ่ายภาพเพื่อเริ่มต้นการทำนาย")
