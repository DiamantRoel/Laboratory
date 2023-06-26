
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('model_1.h5')
feature = ['측정연령수',
           '장애등급명_완전 마비',
           '필요진단_하체',
           '수축기혈압(최고)mmHg', 
           '이완기혈압(최저)mmHg', 
           'BMI', 
           '체지방율']

diagnosis = {
    0: '제자리 걷기',
    1: '앉아서 다리 모으기',
    2: '앉아서 다리 벌리기',
    3: '앉아서 다리 펴기',
    4: '앉아서 다리 밀기',
    5: '앉아서 다리 굽히기',
    6: '하늘자전거타기',
    7: '앉아서 뒤로 당기기',
    8: '발 닿기',
    9: '몸통 들어올리기',
    10: '앉아서 당겨 내리기',
    11: '앉아서 모으기',
    12: '앉아서 위로 밀기',
    13: '바벨 들어올리기',
    14: '앉아서 밀기',
    15: '거꾸로 누워서 밀기',
    16: '비스듬히 누워서 밀기',
    17: '허리 굽혀 덤벨 들기',
    18: '원판 들어올리기',
    19: '서서 어깨 들어올리기',
    20: '파워클린',
    21: '앉았다 일어서기',
    22: '바벨 들어 팔꿈치 굽히기',
    23: '손목 펴기',
    24: '옆구리늘리기',
    25: '턱걸이',
    26: '덤벨 옆으로 들어올리기',
    27: '엎드려서 균형잡기',
    28: '앉아서 팔꿈치 굽히기',
    29: '윗몸 말아 올리기',
    30: '한발 앞으로 내밀고 앉았다 일어서기',
    31: '앉아서 몸통 움츠리기',
    32: '서서 균형잡으며 몸통 회전하기',
    33: '바벨 끌어당기기',
    34: '누워서 밀기',
    35: '허리 굽혀 덤벨 뒤로 들기',
    36: '짝 운동',
    37: '뒤꿈치 들기',
    38: '몸통 옆으로 굽히기',
    39: '매달려서 다리 들기',
    40: '누워서 머리 위로 팔꿈치 펴기',
    41: '의자 앞에서 앉았다 일어서기',
    42: '엎드려서 다리 차올리기'
}

diagnosis_2 = {
    '제자리 걷기': '하체',
    '앉아서 다리 모으기': '하체',
    '앉아서 다리 벌리기': '하체',
    '앉아서 다리 펴기': '하체',
    '앉아서 다리 밀기': '하체',
    '앉아서 다리 굽히기': '하체',
    '하늘자전거타기': '하체',
    '앉아서 뒤로 당기기': '상체',
    '발 닿기': '하체',
    '몸통 들어올리기': '상체',
    '앉아서 당겨 내리기': '상체',
    '앉아서 모으기': '하체',
    '앉아서 위로 밀기': '상체',
    '바벨 들어올리기': '상체',
    '앉아서 밀기': '상체',
    '거꾸로 누워서 밀기': '상체',
    '비스듬히 누워서 밀기': '상체',
    '허리 굽혀 덤벨 들기': '상체',
    '원판 들어올리기': '상체',
    '서서 어깨 들어올리기': '상체',
    '파워클린': '상체',
    '앉았다 일어서기': '하체',
    '바벨 들어 팔꿈치 굽히기': '상체',
    '손목 펴기': '상체',
    '옆구리늘리기': '상체',
    '턱걸이': '상체',
    '덤벨 옆으로 들어올리기': '상체',
    '엎드려서 균형잡기': '상체',
    '앉아서 팔꿈치 굽히기': '상체',
    '윗몸 말아 올리기': '상체',
    '한발 앞으로 내밀고 앉았다 일어서기': '하체',
    '앉아서 몸통 움츠리기': '상체',
    '서서 균형잡으며 몸통 회전하기': '상체',
    '바벨 끌어당기기': '상체',
    '누워서 밀기': '상체',
    '허리 굽혀 덤벨 뒤로 들기': '상체',
    '짝 운동': '상체',
    '뒤꿈치 들기': '하체',
    '몸통 옆으로 굽히기': '상체',
    '매달려서 다리 들기': '상체',
    '누워서 머리 위로 팔꿈치 펴기': '상체',
    '의자 앞에서 앉았다 일어서기': '하체',
    '엎드려서 다리 차올리기': '하체'
}

diagnosis_3 = {
    '제자리 걷기': '불완전마비',
    '앉아서 다리 모으기': '완전마비',
    '앉아서 다리 벌리기': '완전마비',
    '앉아서 다리 펴기': '완전마비',
    '앉아서 다리 밀기': '완전마비',
    '앉아서 다리 굽히기': '완전마비',
    '하늘자전거타기': '완전마비',
    '앉아서 뒤로 당기기': '완전마비',
    '발 닿기': '완전마비',
    '몸통 들어올리기': '완전마비',
    '앉아서 당겨 내리기': '완전마비',
    '앉아서 모으기': '완전마비',
    '앉아서 위로 밀기': '완전마비',
    '바벨 들어올리기': '불완전마비',
    '앉아서 밀기': '완전마비',
    '거꾸로 누워서 밀기': '완전마비',
    '비스듬히 누워서 밀기': '완전마비',
    '허리 굽혀 덤벨 들기': '불완전마비',
    '원판 들어올리기': '불완전마비',
    '서서 어깨 들어올리기': '완전마비',
    '파워클린': '불완전마비',
    '앉았다 일어서기': '완전마비',
    '바벨 들어 팔꿈치 굽히기': '완전마비',
    '손목 펴기': '완전마비',
    '옆구리늘리기': '완전마비',
    '턱걸이': '불완전마비',
    '덤벨 옆으로 들어올리기': '불완전마비',
    '엎드려서 균형잡기': '완전마비',
    '앉아서 팔꿈치 굽히기': '완전마비',
    '윗몸 말아 올리기': '완전마비',
    '한발 앞으로 내밀고 앉았다 일어서기': '완전마비',
    '앉아서 몸통 움츠리기': '완전마비',
    '서서 균형잡으며 몸통 회전하기': '완전마비',
    '바벨 끌어당기기': '불완전마비',
    '누워서 밀기': '완전마비',
    '허리 굽혀 덤벨 뒤로 들기': '불완전마비',
    '짝 운동': '불완전마비',
    '뒤꿈치 들기': '완전마비',
    '몸통 옆으로 굽히기': '완전마비',
    '매달려서 다리 들기': '완전마비',
    '누워서 머리 위로 팔꿈치 펴기': '완전마비',
    '의자 앞에서 앉았다 일어서기': '완전마비',
    '엎드려서 다리 차올리기': '불완전마비'
}

st.title('운동 처방 모델')

user_input = {}
col1, col2, col3 = st.columns(3)
user_input['측정연령수'] = col1.slider('나이', min_value=0, max_value=100, value=20, step=1)
user_input['장애등급명_완전 마비'] = col2.selectbox('장애등급명', ['완전마비', '불완전마비'])
user_input['필요진단_하체'] = col3.selectbox('필요진단', ['상체', '하체'])

col4, col5 = st.columns(2)
user_input['수축기혈압(최고)mmHg'] = col4.slider('최고혈압 / mmHg', min_value=0, max_value=200, value=120, step=1)
user_input['이완기혈압(최저)mmHg'] = col5.slider('최저혈압 / mmHg', min_value=0, max_value=200, value=80, step=1)

col6, col7 = st.columns(2)
user_input['BMI'] = col6.number_input('BMI', value=20.0)
user_input['체지방율'] = col7.number_input('체지방율', value=20.0)

if user_input['장애등급명_완전 마비'] == '완전마비':
    user_input['장애등급명_완전 마비'] = 1
else:
    user_input['장애등급명_완전 마비'] = 0

if user_input['필요진단_하체'] == '하체':
    user_input['필요진단_하체'] = 1
else:
    user_input['필요진단_하체'] = 0

model_input = np.array([
    user_input['측정연령수'],
    user_input['장애등급명_완전 마비'],
    user_input['필요진단_하체'],
    user_input['수축기혈압(최고)mmHg'],
    user_input['이완기혈압(최저)mmHg'],
    user_input['BMI'],
    user_input['체지방율']
])

model_input = model_input.reshape(1, -1)

if st.button('처방'):
    prediction = model.predict(model_input)
    top_diagnoses_indices = np.argsort(prediction[0])[::-1]
    top_diagnoses = [diagnosis[i] for i in top_diagnoses_indices]

    if user_input['필요진단_하체'] == 1:
        if user_input['장애등급명_완전 마비'] == 1:
            top_diagnoses = [d for d in top_diagnoses if diagnosis_2[d] == '하체' and diagnosis_3[d] == '완전마비']
        elif user_input['장애등급명_완전 마비'] == 0:
            top_diagnoses = [d for d in top_diagnoses if diagnosis_2[d] == '하체' and diagnosis_3[d] == '불완전마비']
    elif user_input['필요진단_하체'] == 0:
        if user_input['장애등급명_완전 마비'] == 1:
            top_diagnoses = [d for d in top_diagnoses if diagnosis_2[d] == '상체' and diagnosis_3[d] == '완전마비']
        elif user_input['장애등급명_완전 마비'] == 0:
            top_diagnoses = [d for d in top_diagnoses if diagnosis_2[d] == '상체' and diagnosis_3[d] == '불완전마비']

    top_diagnoses = top_diagnoses[:3] if len(top_diagnoses) >= 3 else top_diagnoses

    for i, diagnosis in enumerate(top_diagnoses):
        st.write(f'{diagnosis}')
