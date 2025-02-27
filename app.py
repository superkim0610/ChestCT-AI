import streamlit as st
from PIL import Image
import numpy as np
import model

# Streamlit 앱 제목
st.title("Lung Cancer Prediction from Chest CT with CNN")

# 이미지 업로드
uploaded_file = st.file_uploader("Upload Chest CT Image", type=["jpg", "jpeg", "png", "bmp", "dcm"])

# 업로드된 이미지가 있으면 처리
if uploaded_file is not None:
    # 이미지 보여주기
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest CT', use_column_width=True)

    # 'Predict' 버튼 생성
    if st.button("Predict"):
        # 이미지 배열로 변환 (모델 입력 형식에 맞게 전처리 필요)
        
        prediction = model.predict_img(image)

        # 결과 출력
        st.subheader("Prediction Results:")
        labels = ['normal', 'adenocarcinoma', 'large cell carcinoma', 'squamous cell carcinoma']
        for i in range(len(labels)):
            st.write(f"{labels[i]}: {prediction[0, i] * 100:.2f}%")
            
        # for label, probability in predictions.items():
        #     st.write(f"{label}: {probability * 100:.2f}%")
        # st.write(str(prediction))