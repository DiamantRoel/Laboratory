import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import webbrowser

model = load_model('model.h5')
feature = ['측정연령수', '수축기혈압(최고)mmHg', '이완기혈압(최저)mmHg', 'BMI', '체지방율', '악력']

diagnosis = {
    0: '제자리 걷기',
    1: '바벨 들어 팔꿈치 굽히기',
    2: '발 닿기',
    3: '앉아서 다리 모으기',
    4: '앉아서 다리 벌리기',
    5: '앉아서 다리 펴기',
    6: '앉아서 다리 밀기',
    7: '하늘 자전거타기',
    8: '발 닿기',
    9: '앉아서 다리 굽히기',
    10: '앉아서 뒤로 당기기',
    11: '몸통 들어올리기',
    12: '앉아서 당겨 내리기',
    13: '앉아서 모으기',
    14: '앉아서 위로 밀기',
    15: '바벨들어올리기',
    16: '앉아서 밀기',
    17: '누워서 밀기',
    18: '거꾸로 누워서 밀기',
    19: '비스듬히 누워서 밀기',
    20: '허리 굽혀 덤벨 들기',
    21: '원판들어올리기',
    22: '서서 어깨 들어올리기',
    23: '파워클린',
    24: '바벨 끌어당기기',
    25: '턱걸이',
    26: '옆구리늘리기',
    27: '앉았다 일어서기',
    28: '바벨 들어 팔',
    29: '손목 펴기',
    30: '굽히기',
    31: '엎드려서 균형잡기',
    32: '한발 앞으로 내밀고 앉았다 일어서기',
    33: '덤벨 옆으로 들어올리기',
    34: '앉아서 팔꿈치 굽히기',
    35: '펴기',
    36: '윗몸 말아 올리기',
    37: '앉아서 몸통 움츠리기',
    38: '서서 균형잡으며 몸통 회전하기',
    39: '의자 앞에서 앉았다 일어서기',
    40: '허리 굽혀 덤벨 뒤로 들기',
    41: '계단 올라갔다 내려오기',
    42: '뒤꿈치 들기',
    43: '몸통 옆으로 굽히기',
    44: '누워서 머리 위로 팔꿈치 펴기',
    45: '매달려서 다리 들기',
    46: '계단 뛰어 오르기',
    47: '짝 운동',
    48: '엎드려서 다리 차올리기'
}

def open_previous_page():
    url = "https://gorgeous-dolphin-5dace2.netlify.app/course.html" 
    webbrowser.open_new_tab(url)

col1, col2, col3 = st.columns([3, 1, 1])
col1.title('운동 처방 모델')
col3.button('이전', on_click=open_previous_page)

col4, col5, col6 = st.columns(3)
user_input = {}
user_input['측정연령수'] = col4.number_input('나이', min_value=0, step=1, value=0)
user_input['수축기혈압(최고)mmHg'] = col5.number_input('최고혈압 / mmHg', min_value=0.0)
user_input['이완기혈압(최저)mmHg'] = col6.number_input('최저혈압 / mmHg', min_value=0.0)

col7, col8, col9 = st.columns(3)
user_input['BMI'] = col7.number_input('BMI', min_value=0.0)
user_input['체지방율'] = col8.number_input('체지방율', min_value=0.0)
user_input['악력'] = col9.number_input('악력', min_value=0.0)

if st.button('처방'):
    input_features = [user_input[feature] for feature in feature]
    input_array = np.array([input_features])
    predictions = model.predict(input_array)

    top_indices = np.argsort(predictions, axis=1)[:, -3:][0]
    top_predictions = predictions[:, top_indices][0]

    for i, prediction in enumerate(top_predictions):
        exercise_name = diagnosis[top_indices[i]]
        st.write(f'{i+1}: {exercise_name}')