# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:50:56 2023

@author: YOGB
"""
# IMPORT LIBRARIES ============================================================
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader
from streamlit_extras.app_logo import add_logo
from PIL import Image

# INITIAL AND DEFAULT SETTING =================================================
st.set_page_config(page_title="WIBOGINA", layout="wide")
# st.image("./logo1.png")
add_logo("./logo1_small.png", height=50)

# AUTHENTICATION ==============================================================
with open('./config.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.header(f'Welcome *{st.session_state["name"]}*')
# START OF CONTENT ==============================================================  
    st.title("Welcome!")
    st.markdown("""
                This is a Graphical User Interface for :blue[**WIBOGINA**], which you can use
                to create your own 3D stratigraphy model and generate cross sections.
                
                Within this web-app, there are two pages which serve two different workflows.
                With the :blue[**BUILD**] page, you can create your WIBOGINA model from scratch.
                It means that you need to have the following ingredients: individual borehole
                or CPT files (using Ginaloket XLSX template) and topography/bathymetry data
                in XLSX as well. You can refer to the following illustration. 
                
                Once the borehole or
                CPT files are uploaded (and topography data, if available), the lithology plot
                can be created by first clicking the line path from the generated map. Afterward,
                you can directly assign the stratigraphy layers by editing the stratigraphy table
                below the plot (don't forget to click the UPDATE button to compile the stratigraphy
                table into the main compiled dataframe).
                
                Finally, once you are happy with the model, you can download the compiled WIBOGINA 
                workbook (in XLSX). You can use this compiled XLSX again if you want to create more
                plots in the future, using the :blue[**DISPLAY**] page.
                
                """)
    st.write("#")
    st.write("#")
    image1 = Image.open("flow1.png")
    st.image(image1, caption="Workflow for model building using BUILD page", width=1100)
    st.write("#")
    st.write("#")
    image2 = Image.open("flow2.png")
    st.image(image2, caption="Workflow for model display using DISPLAY page", width=700)
    
    with st.sidebar.expander("Reset password"):
        if authentication_status:
            try:
                if authenticator.reset_password(username, 'Reset password', location="main"):
                    st.success('Password modified successfully')
                    with open('./config.yml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
            except Exception as e:
                st.error(e)
                
    with st.sidebar.expander("Update user details"):
        if authentication_status:
            try:
                if authenticator.update_user_details(username, 'Update user details', location="main"):
                    st.success('Entries updated successfully')
                    with open('./config.yml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
            except Exception as e:
                st.error(e)

# END OF CONTENT ==============================================================

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')

# if st.session_state["authentication_status"]:
#     pass
# else:
#     if st.checkbox("Sign up"):
#         try:
#             if authenticator.register_user('Register user', preauthorization=True):
#                 st.success('User registered successfully')
#                 with open('./config.yml', 'w') as file:
#                     yaml.dump(config, file, default_flow_style=False)
#         except Exception as e:
#             st.error(e)