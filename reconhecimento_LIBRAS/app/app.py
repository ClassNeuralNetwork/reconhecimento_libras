import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import h5py

with h5py.File('model.h5', 'r', driver='core') as f:
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
        pass
        
    def transform(self, frame):
        
        frame_bgr = np.array(frame.to_image())

        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
     
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                hand_points = np.array([[lmk.x, lmk.y] for lmk in hand_landmarks.landmark]).astype(np.float32)
                xmin, ymin = np.min(hand_points, axis=0)
                xmax, ymax = np.max(hand_points, axis=0)
                xmin, ymin, xmax, ymax = int(xmin * frame_rgb.shape[1]), int(ymin * frame_rgb.shape[0]), \
                                         int(xmax * frame_rgb.shape[1]), int(ymax * frame_rgb.shape[0])
                
                
                hand_roi = frame_rgb[ymin:ymax, xmin:xmax]
                
                predicted_object = predict_object(hand_roi)
                
                
                frame_rgb = cv2.putText(frame_rgb, predicted_object, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                print(predicted_object)
        
        return frame_rgb

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
