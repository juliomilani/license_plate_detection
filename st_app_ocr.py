# Playground para mexer nos filtros e testar o OCR:
# Spoiler: Melhor não por filtros
import cv2 as cv
import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io,color,filters, measure,exposure,transform,util,morphology,measure
import matplotlib.pyplot as plt
import altair as alt
import pytesseract
import string

class Params:
    def __init__(self):
        self.img_width = 300
        self.thresh_blocksize = st.sidebar.slider('thresh_blocksize',min_value = 0,max_value=35,value=11)
        self.thresh_C = st.sidebar.slider('thresh_C',min_value = 0,max_value=35,value=2)
        if(self.thresh_blocksize%2==0):
            self.thresh_blocksize+=1
        if st.sidebar.checkbox('Filtros:'):
            self.kernel = st.sidebar.slider('Kernal Gauss',min_value = 0,max_value=100,value=5)
            if(self.kernel%2==0):
                self.kernel+=1
            self.dia_bil = st.sidebar.slider('Diameter Bilateral',min_value = 0,max_value=255,value=8)
            self.sigcol_bil = st.sidebar.slider('SigmaColor Bilateral',min_value = 0,max_value=255,value=50)
            self.sigspa_bil = st.sidebar.slider('SigmaSpace Bilateral',min_value = 0,max_value=255,value=50)
        else:
            self.kernel = 5
            self.dia_bil = 8
            self.sigcol_bil = 50
            self.sigspa_bil = 50 

params = Params()
def main():
    st.title('ALPR OCR playground:')
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg",'png','jpeg','gif'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        upl_img_0 = cv.imdecode(file_bytes,1)
        upl_img_0 = rescale(upl_img_0)
        upl_img_1 = foo(upl_img_0)
        st.image([upl_img_0,upl_img_1], width=params.img_width)

    # Images in folder part:
    imgs_path = get_all_imgs_paths()
    for i,img_path in enumerate(imgs_path):
        st.write('--------------------------------')
        st.write("Img",i,os.path.abspath(imgs_path[i]),":")
        img_0 = cv.imread(img_path,0)
        img_0 = rescale(img_0)
        img_1 = foo(img_0)


def get_all_imgs_paths():
    folder_0 = "teste_imgs/placas/"
    imgs_filenames = os.listdir(folder_0)
    imgs_path = []
    for img_filename in imgs_filenames:
        img_path = os.path.join(folder_0,img_filename)
        imgs_path.append(img_path)
    st.write("Number of images: ", len(imgs_path))
    return imgs_path

def rescale(img_in):
    new_width = 500
    scale_ratio = new_width / img_in.shape[1]
    new_height = int(img_in.shape[0] * scale_ratio)
    dim = (new_width, new_height)
    img_out = cv.resize(img_in.copy(), dim, interpolation = cv.INTER_AREA)
    return img_out


def img2str(image):
    plate = pytesseract.image_to_string(image,config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789')
    plate = plate.strip(string.ascii_lowercase)
    plate = [char for char in plate if char.isalnum()]
    plate = ''.join(plate)
    return plate


def get_background(image):
    kernel_size = 50
    background = morphology.closing(image,morphology.square(kernel_size))
    
    return background

def remove_background(image):
    background = get_background(image)
    new_image = image + np.invert(background.astype(np.int))
    return new_image

def image_closing_n_opening(new_image):
    kernel_size = 10
    # new_image = morphology.opening(new_image,morphology.square(kernel_size))
    new_image = morphology.closing(new_image,morphology.square(kernel_size))
    # new_image = morphology.dilation(new_image,morphology.square(kernel_size))
    return new_image

def foo(img_in):
    st.text(img2str(img_in))
    st.image(img_in, width=params.img_width)

    with st.echo():
        img_out = rescale(img_in)
    st.text(img2str(img_out))
    st.image(img_out, width=params.img_width) 

    with st.echo():
        img_out = cv.GaussianBlur(img_in.copy(),(params.kernel,params.kernel),cv.BORDER_DEFAULT)
    st.text(img2str(img_out))
    st.image(img_out, width=params.img_width) 

    with st.echo():
        img_out = cv.bilateralFilter(img_out, params.dia_bil, params.sigcol_bil, params.sigspa_bil)
    st.text(img2str(img_out))
    st.image(img_out, width=params.img_width) 

    with st.echo():
        img_out = cv.equalizeHist(img_out)
    st.text(img2str(img_out))
    st.image(img_out, width=params.img_width) 

    # with st.echo():
    #     img_out = cv.adaptiveThreshold(img_out,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,params.thresh_blocksize,params.thresh_C)
    # st.text(img2str(img_out))
    # st.image(img_out, width=params.img_width) 

    # with st.echo():
    #     back = remove_background(img_in)
    # st.text(img2str(img_out))
    # st.image(img_out, width=params.img_width) 

    # with st.echo():
    #     img_out = image_closing_n_opening(img_in)
    # st.text(img2str(img_out))
    # st.image(img_out, width=params.img_width) 

    # img_out = cv.GaussianBlur(img_in.copy(),(params.kernel,params.kernel),cv.BORDER_DEFAULT)
    # st.text(img2str(img_out))
    # img_out = cv.bilateralFilter(img_out, params.dia_bil, params.sigcol_bil, params.sigspa_bil)
    # st.text(img2str(img_out))
    # img_out = cv.Canny(img_out,params.t1can,params.t2can)
    # st.text(img2str(img_out))
    return img_out


if __name__ == '__main__':
    main()
