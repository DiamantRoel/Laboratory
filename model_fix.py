
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import webbrowser

model = load_model('model.h5')
feature = ['측정연령수', '장애등급명', '수축기혈압(최고)mmHg', '이완기혈압(최저)mmHg', 'BMI', '체지방율']

diagnosis = {
    0: '제자리 걷기',
    1: '앉아서 다리 모으기',
    2: '앉아서 다리 벌리기',
    3: '바벨들어올리기',
    4: '앉아서 다리 밀기',
    5: '앉아서 다리 펴기',
    6: '하늘자전거타기',
    7: '앉아서 다리 굽히기',
    8: '손목 펴기',
    9: '원판들어올리기',
    10: '앉았다 일어서기',
    11: '앉아서 위로 밀기',
    12: '파워클린',
    13: '비스듬히 누워서 밀기',
    14: '앉아서 밀기',
    15: '허리 굽혀 덤벨 들기',
    16: '앉았다 일어서기',
    17: '거꾸로 누워서 밀기',
    18: '앉아서 몸통 움츠리기',
    19: '옆구리늘리기',
    20: '덤벨 옆으로 들어올리기',
    21: '허리 굽혀 덤벨 뒤로 들기',
    22: '발 닿기',
    23: '서서 어깨 들어올리기',
    24: '서서 균형잡으며 몸통 회전하기',
    25: '엎드려서 균형잡기',
    26: '앉아서 팔꿈치 굽히기',
    27: '턱걸이',
    28: '윗몸 말아 올리기',
    29: '누워서 밀기',
    30: '한발 앞으로 내밀고 앉았다 일어서기',
    31: '앉아서 뒤로 당기기',
    32: '몸통 옆으로 굽히기',
    33: '뒤꿈치 들기',
    34: '발 끌어당기기',
    35: '앉아서 팔꿈치 굽히기',
    36: '바벨 들어 팔꿈치 굽히기',
    37: '누워서 머리 위로 팔꿈치 펴기',
    38: '의자 앞에서 앉았다 일어서기',
    39: '매달려서 다리 들기',
    40: '앉아서 당겨 내리기',
    41: '짝 운동',
    42: '엎드려서 다리 차올리기'
}

st.title('운동 처방 모델')

user_input = {}
col1, col2 = st.columns(2)
user_input['측정연령수'] = col1.slider('나이', min_value=0, max_value=100, value=20, step=1)
user_input['장애등급명'] = col2.radio('장애등급명', options=[0, 1], format_func=lambda x: '불완전마비' if x == 0 else '완전마비', index=0)

col3, col4 = st.columns(2)
user_input['수축기혈압(최고)mmHg'] = col3.slider('최저혈압 / mmHg', min_value=0, max_value=200, value=120, step=1)
user_input['이완기혈압(최저)mmHg'] = col4.slider('최저혈압 / mmHg', min_value=0, max_value=200, value=80, step=1)

col5, col6 = st.columns(2)
user_input['BMI'] = col5.number_input('BMI', min_value=0.0)
user_input['체지방율'] = col6.number_input('체지방율', min_value=0.0)

if st.button('처방'):
    input_features = [user_input[feature] for feature in feature]
    if user_input['장애등급명'] == 0:
        input_features[1] = 0.0
    else:
        input_features[1] = 1.0
    input_array = np.array([input_features])
    predictions = model.predict(input_array)

    top_indices = np.argsort(predictions, axis=1)[:, -3:][0]
    top_predictions = predictions[:, top_indices][0]

    for i, prediction in enumerate(top_predictions):
        exercise_name = diagnosis[top_indices[i]]
        st.write(f'{i+1}: {exercise_name}')
