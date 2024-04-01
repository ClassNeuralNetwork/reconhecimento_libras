import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import h5py
import mediapipe as mp

mp_hands = mp.solutions.hands

with h5py.File('reconhecimento_libras/modelo/model.h5', 'r', driver='core') as f:
    model = load_model(f, compile=False)


label_to_text = {0: 'bus', 1: 'bank', 2: 'car', 3: 'formation', 4: 'hospital', 5: 'I', 6: 'man', 7: 'motorcycle', 8: 'my', 9: 'supermarket', 10: 'we', 11: 'woman', 12: 'you', 13: 'you (plural)', 14: 'your'}


def predict_object(hand_roi):
    
    resized_hand = cv2.resize(hand_roi, (120, 213))
    
    
    normalized_hand = resized_hand.astype('float32') / 255.0
    
    
    predicted_class = model.predict(np.expand_dims(normalized_hand, axis=0)).argmax()
    predicted_object = label_to_text[predicted_class]
    
    return predicted_object


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands()
    
    def coordinates(self, hand_landmarks, img):
        offset = 20
        
        x_min = max(0, min([landmark.x for landmark in hand_landmarks.landmark]) * img.shape[1] - offset)
        x_max = min(img.shape[1], max([landmark.x for landmark in hand_landmarks.landmark]) * img.shape[1] + offset)
        y_min = max(0, min([landmark.y for landmark in hand_landmarks.landmark]) * img.shape[0] - offset)
        y_max = min(img.shape[0], max([landmark.y for landmark in hand_landmarks.landmark]) * img.shape[0] + offset)
        
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        
        hand_roi = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        predicted_object = predict_object(hand_roi)
        
        text_x = int(x_min)
        text_y = max(0, int(y_min) - 10)

        img = cv2.putText(img, predicted_object,(text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            fist_hand = results.multi_hand_landmarks[0]
            self.coordinates(fist_hand, img)
            
            # Check if there is a second hand
            if len(results.multi_hand_landmarks) > 1:
                second_hand = results.multi_hand_landmarks[1]
                self.coordinates(second_hand, img)
            
        return img

st.sidebar.image("https://www.mjvinnovation.com/wp-content/uploads/2021/07/mjv_blogpost_redes_neurais_ilustracao_cerebro-01-1024x1020.png")

st.sidebar.title('Reconhecimento de :red[Sinais] :ok_hand:')


st.sidebar.info("""
## Reconhecimento de Mãos - Projeto

Este projeto visa desenvolver um programa capaz de utilizar uma rede neural treinada para detectar mãos em tempo real por meio da câmera do usuário. Usando um modelo de rede neural CNN. 
""")


webrtc_streamer(key="hand-recognition-1", video_processor_factory=VideoTransformer)


st.sidebar.write("""
## Integrantes

- Lorrayne
- Libhinny
- Samira
- Ytalo
""")
