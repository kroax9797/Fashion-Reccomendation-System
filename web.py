import streamlit as st
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import cv2 


#Importing Resnet50 model
model = ResNet50(weights='imagenet' , include_top=False , input_shape=(224,224,3))
model.trainable = False 
model = tensorflow.keras.Sequential([
    model ,
    GlobalMaxPooling2D()
])

#function to save image into uploads folder (mahiti nahi save ka karat aahe image te)
def save_img(uploaded_file):
    try:
        with open(os.path.join('uploads' , uploaded_file.name) , 'wb') as f:
            f.write(uploaded_file.getbuffer())
            st.write("Upload Success")
            display_image = Image.open(uploaded_file)
            return display_image
            # st.image(display_image , width=256)
    except:
        st.warning("Upload failed !")

#function to extract features from an image 
def extract_features(img_path , model):
    img = image.load_img(img_path , target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img , axis = 0)
    img = preprocess_input(img)
    result = model.predict(img).flatten()
    result = result / norm(result)
    return result  


def reccomend(features , feature_list):
    neighbours = NearestNeighbors(n_neighbors=25,algorithm='brute',metric = 'euclidean')
    neighbours.fit(feature_list)
    distances , indices = neighbours.kneighbors([features])
    return distances , indices 


filenames = []
with open ('filenames.pkl' , 'rb') as f:
    filenames = pickle.load(f)

feature_list = []
filename = ['embeddings0-20.pkl' , 'embeddings20-30.pkl' ,'embeddings30-44.441.pkl']

for file in filename :
    with open(file , 'rb') as f : 
        data = pickle.load(f)
        for i in range(len(data)):
            feature_list.append(data[i])


#webpage design over here :
st.title("Fashion Reccomender System")

#upload file : 
uploaded_file = st.file_uploader("Choose an image :")
if uploaded_file is not None :
    reference_image = save_img(uploaded_file)
    features = extract_features(os.path.join('uploads',uploaded_file.name),model)
    similarity , indices = reccomend(features,feature_list)

    similarity = np.array(similarity)
    similarity = 1/similarity
    similarity = similarity/norm(similarity)

    col0 , col1 , col2 , col3 , col4 , col5 = st.columns(6)
    
    j = 0

    with col0:
        st.image(reference_image , use_column_width=True)
        st.write("Uploaded Image")
    
    j = 0
    with col1 : 
        
        for i in range(0 , 5 , 5):
            img = Image.open(filenames[indices[0][i+j]])
            # img = img.resize((256,256))
            st.image(img , use_column_width=True)
            st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")

    j = 1
    with col2 :
        for i in range(0 , 5 , 5):
            img = Image.open(filenames[indices[0][i+j]])
            # img = img.resize((256,256))
            st.image(img , use_column_width=True)
            st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")

    j = 2
    with col3 : 
        for i in range(0 , 5 , 5):
            img = Image.open(filenames[indices[0][i+j]])
            # img = img.resize((256,256))
            st.image(img , use_column_width=True)
            st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")

    j = 3
    with col4 : 
        for i in range(0 , 5 , 5):
            img = Image.open(filenames[indices[0][i+j]])
            # img = img.resize((256,256))
            st.image(img , use_column_width=True)
            st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")
    j = 4
    with col5 : 
        for i in range(0 , 5 , 5):
            img = Image.open(filenames[indices[0][i+j]])
            # img = img.resize((256,256))
            st.image(img , use_column_width=True)
            st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")

    with st.expander("Open to see more reccomendations"):
        col1 , col2 , col3 , col4 , col5 = st.columns(5)

        j = 0
        with col1 : 
            
            for i in range(5 , 25 , 5):
                img = Image.open(filenames[indices[0][i+j]])
                # img = img.resize((256,256))
                st.image(img , use_column_width=True)
                st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")

        j = 1
        with col2 :
            for i in range(5 , 25 , 5):
                img = Image.open(filenames[indices[0][i+j]])
                # img = img.resize((256,256))
                st.image(img , use_column_width=True)
                st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")

        j = 2
        with col3 : 
            for i in range(5 , 25 , 5):
                img = Image.open(filenames[indices[0][i+j]])
                # img = img.resize((256,256))
                st.image(img , use_column_width=True)
                st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")

        j = 3
        with col4 : 
            for i in range(5 , 25 , 5):
                img = Image.open(filenames[indices[0][i+j]])
                # img = img.resize((256,256))
                st.image(img , use_column_width=True)
                st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")
        j = 4
        with col5 : 
            for i in range(5 , 25 , 5):
                img = Image.open(filenames[indices[0][i+j]])
                # img = img.resize((256,256))
                st.image(img , use_column_width=True)
                st.write(f"Similarity Index : {round(similarity[0][i+j]*100 , 2)}")


