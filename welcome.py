import shutil
import os
import sys
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from visualization import main, simple_preprocess_st1, preprocess_file_collation

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def mri_model(path):
    path_lst = []
    for i in range(32):
        png_name = path + '/' + 'img' + str((i + 1) * 3 - 3) + '.png'
        path_lst.append(png_name)

    arr_lst = []
    for i in range(32):
        img = Image.open(path_lst[i]).convert('L')
        arr_lst.append(img)

    final_arr = np.stack(arr_lst)

    volume = final_arr
    r, c = volume[0].shape

    # Define frames

    nb_frames = 32

    fig = go.Figure(
        frames=[go.Frame(
            data=go.Surface(z=(3.1 - k * 0.1) * np.ones((r, c)),
                            surfacecolor=volume[31 - k],
                            cmin=0,
                            cmax=255
                            ),
            name=str(k)  # you need to name the frame for the animation to behave properly
        )
            for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=3.1 * np.ones((r, c)),
        surfacecolor=np.zeros_like(volume[31]),
        colorscale="gray",
        cmin=0, cmax=255,
        colorbar=dict(thickness=20, ticklen=4)
    ))

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title='Slices in MRI data',
        width=600,
        height=600,
        scene=dict(
            zaxis=dict(range=[-0.1, 6.8],
                       autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    st.plotly_chart(fig)


def try_sample():
    st.header("Show MRI 3D model")
    mri_model('images')

    st.header("Show Segmentation")
    suc1 = st.slider('slices', 1, 32, 20, 1)
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    num = suc1 * 3 - 3
    num2 = suc1 * 3 - 1
    image = Image.open('images/img' + str(num) + '.png')
    image2 = Image.open('images/img' + str(num2) + '.png')

    ax1.imshow(image)
    ax1.set(xlabel="Input Image")
    ax2.imshow(image2)
    ax2.set(xlabel="Prediction Image")

    st.pyplot(fig1)


def output():
    st.header("Show MRI 3D model")
    mri_model('input')

    st.header("Show Segmentation")
    suc1 = st.slider('slices', 1, 32, 20, 1)
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    num = suc1 * 3 - 3
    num2 = suc1 * 3 - 1
    image = Image.open('input/img' + str(num) + '.png')
    image2 = Image.open('input/img' + str(num2) + '.png')

    ax1.imshow(image)
    ax1.set(xlabel="Input Image")
    ax2.imshow(image2)
    ax2.set(xlabel="Prediction Image")

    st.pyplot(fig1)


with st.sidebar:
    page = option_menu(menu_title='Menu',
                       menu_icon="robot",
                       options=["Welcome!",
                                "Try example",
                                "Start from npy file",
                                "Start from labeled MRI file",
                                "Start from original WMH challenge file"],
                       icons=["house-door",
                              "chat-dots",
                              "key",
                              "tag",
                              "building"],
                       default_index=0
                       )

st.title('WMH Segmentation Challenge')

if page == "Welcome!":
    st.header('Welcome!')

    st.markdown("![Alt Text](https://media.giphy.com/media/2fEvoZ9tajMxq/giphy.gif)")
    st.write(
        """


        """
    )

    st.subheader("Introduction")
    st.write("""
        Hello! This project builds a neural network model that can accurately segment WMH from MRI images. It 
        is based on the WMH Segmentation Challenge and is led by the Faculty of Engineering at the University of 
        Edinburgh. The project has been published, and the paper and raw data can be viewed through the link below. 
            
        *[Article: White matter hyperintensity and stroke lesion segmentation and differentiation using convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S2213158217303273)
            
        *[WMH Segmentation Challenge Data](https://wmh.isi.uu.nl/)
    
        White matter hyperintensities (WMH) are a feature of sporadic small vessel disease also frequently observed 
        in magnetic resonance images (MRI) of healthy elderly subjects. The accurate assessment of WMH burden is of 
        crucial importance for epidemiological studies to determine association between WMHs, cognitive and clinical 
        data; their causes, and the effects of new treatments in randomized trials. The manual delineation of WMHs is 
        a very tedious, costly and time consuming process, that needs to be carried out by an expert annotator (e.g. 
        a trained image analyst or radiologist). The problem of WMH delineation is further complicated by the fact 
        that other pathological features (i.e. stroke lesions) often also appear as hyperintense regions. Recently, 
        several automated methods aiming to tackle the challenges of WMH segmentation have been proposed. Most of 
        these methods have been specifically developed to segment WMH in MRI but cannot differentiate between WMHs 
        and strokes. Other methods, capable of distinguishing between different pathologies in brain MRI, 
        are not designed with simultaneous WMH and stroke segmentation in mind. Therefore, a task specific, reliable, 
        fully automated method that can segment and differentiate between these two pathological manifestations on 
        MRI has not yet been fully identified. In this work we propose to use a convolutional neural network (CNN) 
        that is able to segment hyperintensities and differentiate between WMHs and stroke lesions. Specifically, 
        we aim to distinguish between WMH pathologies from those caused by stroke lesions due to either cortical, 
        large or small subcortical infarcts. The proposed fully convolutional CNN architecture, called uResNet, 
        that comprised an analysis path, that gradually learns low and high level features, followed by a synthesis 
        path, that gradually combines and up-samples the low and high level features into a class likelihood semantic 
        segmentation. Quantitatively, the proposed CNN architecture is shown to outperform other well established and 
        state-of-the-art algorithms in terms of overlap with manual expert annotations. Clinically, the extracted WMH 
        volumes were found to correlate better with the Fazekas visual rating score than competing methods or the 
        expert-annotated volumes. Additionally, a comparison of the associations found between clinical risk-factors 
        and the WMH volumes generated by the proposed method, was found to be in line with the associations found 
        with the expert-annotated volumes. 

        * This application currently have five pages:
            * Welcome!
            * Try example
            * Start from npy file
            * Start from labeled MRI file
            * Start from original WMH challenge file

        To learn how to use this model, please read the How to use it section below.
        """
             )

    st.subheader("How to use it")
    st.write(
        """
        Please find the data structure corresponding to the file you want to upload according to the navigation 
        bar on the right. Different data structures may cause different running times. Incorrectly uploaded files may 
        result in errors. 
        
        If you want to see the prediction performance of the model, you can directly click on the 
        Try example page. 
        """
    )

    st.subheader("Contacts")
    st.write(
        """
        * If you have any questions or suggestions about the project, you can send us an email through the following contact information, and we will reply as soon as possible after receiving the email.

           [Project Github](https://github.com/BenjaminPhi5/Trustworthai-MRI-WMH)

           [Project Instructor: Dr. Chen Qin](https://www.eng.ed.ac.uk/about/people/dr-chen-qin)

           [Code Builder: Ben Philps](https://www.inf.ed.ac.uk/people/students/Benjamin_Philps.html)

           [Web Maintainer: Jingyu Sun](s2091784@ed.ac.uk)
           
           [Project Agency: University of Edinburgh](https://www.eng.ed.ac.uk/)

        """
    )
elif page == "Try example":
    st.header('Try example')
    st.markdown("![Alt Text](https://media3.giphy.com/media/xT0BKr4MvHdohFTe6s/giphy.gif?cid"
                "=ecf05e47i2seweh432ix10vtjsuhtxvd4atd86z5b0hyj2a2&rid=giphy.gif&ct=g)")
    st.write(
        """
        """
    )

    try_sample()


elif page == "Start from npy file":
    st.header('Start from npy file')
    st.markdown("![Alt Text](https://media.giphy.com/media/XIqCQx02E1U9W/giphy.gif)")
    st.write(
        """
        """
    )

    st.image('web/npy.png', caption='data structure of npy file')
    uploaded_file = st.file_uploader("Choose a npy file", type='zip')

    if st.button('upload'):
        with st.spinner("Loading..."):
            if uploaded_file is not None:
                st.success('upload successful')
                path_in = uploaded_file.name
                st.write('File name: ' + path_in)
                path_in = os.path.splitext(path_in)[0]
                st.write('File type: ' + uploaded_file.type)
                # -*-coding: UTF-8 -*-
                from zipfile import ZipFile

                shutil.rmtree('data')
                os.mkdir('data')
                zFile = ZipFile(uploaded_file)
                # ZipFile.namelist(): 获取ZIP文档内所有文件的名称列表 Get a list of the names of all files within a ZIP archive
                for fileM in zFile.namelist():
                    zFile.extract(fileM, 'data/')
                zFile.close()
                main('data/' + path_in)
            else:
                st.write('error')
    else:
        st.write('please upload a npy file')

    output()


elif page == "Start from labeled MRI file":
    st.header('Start from labeled MRI file')
    st.markdown("![Alt Text](https://media.giphy.com/media/xT9C25UNTwfZuk85WP/giphy-downsized-large.gif)")
    st.write(
        """
        """
    )

    st.image('web/label.png', caption='data structure of labeled MRI file')
    uploaded_file = st.file_uploader("Choose a labeled MRI file", type='zip')

    if st.button('upload'):
        with st.spinner("Loading..."):
            if uploaded_file is not None:
                st.success('upload successful')
                path_in = uploaded_file.name
                st.write('File name: ' + path_in)
                path_in = os.path.splitext(path_in)[0]
                st.write('File type: ' + uploaded_file.type)
                # -*-coding: UTF-8 -*-
                from zipfile import ZipFile

                shutil.rmtree('data')
                os.mkdir('data')
                zFile = ZipFile(uploaded_file)
                # ZipFile.namelist(): 获取ZIP文档内所有文件的名称列表 Get a list of the names of all files within a ZIP archive
                for fileM in zFile.namelist():
                    zFile.extract(fileM, 'data/')
                zFile.close()
                preprocess_file_collation('data/' + path_in, 'data/', 'Singapore')
                preprocess_file_collation('data/' + path_in, 'data/', 'GE3T')
                preprocess_file_collation('data/' + path_in, 'data/', 'Utrecht')
                main('data/')
            else:
                st.write('error')
    else:
        st.write('please upload a labeled MRI file')

    output()

elif page == "Start from original WMH challenge file":
    st.header('Start from original WMH challenge file')
    st.markdown("![Alt Text](https://media.giphy.com/media/WoWm8YzFQJg5i/giphy.gif)")
    st.write(
        """
        """
    )

    st.image('web/structure.png', caption='data structure of original WMH challenge file')
    uploaded_file = st.file_uploader("Choose an original WMH challenge file", type='zip')
    if st.button('upload'):
        with st.spinner("Loading..."):
            if uploaded_file is not None:
                sys.path.append("../../")
                st.success('upload successful')
                path_in = uploaded_file.name
                st.write('File name: ' + path_in)
                path_in = os.path.splitext(path_in)[0]
                st.write('File type: ' + uploaded_file.type)
                # -*-coding: UTF-8 -*-
                from zipfile import ZipFile

                shutil.rmtree('data')
                os.mkdir('data')
                zFile = ZipFile(uploaded_file)
                # ZipFile.namelist(): 获取ZIP文档内所有文件的名称列表 Get a list of the names of all files within a ZIP archive
                for fileM in zFile.namelist():
                    zFile.extract(fileM, 'data/')
                zFile.close()
                os.mkdir('data/precessed/')
                simple_preprocess_st1('data/', 'data/precessed/', 'WMH_challenge_dataset', 0, -1)
                preprocess_file_collation('data/' + path_in, 'data/', 'Singapore')
                preprocess_file_collation('data/' + path_in, 'data/', 'GE3T')
                preprocess_file_collation('data/' + path_in, 'data/', 'Utrecht')
                main('data/')
            else:
                st.write('error')
    else:
        st.write('please upload an original WMH challenge file')

    output()
# streamlit run welcome.py --server.maxUploadSize 10000
