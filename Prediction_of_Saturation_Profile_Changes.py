import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import math
from PIL import Image
from utils import *

st.set_page_config(
    page_title='Prediction of Saturation Changes during Inundation',
    page_icon=":arrow_right:"
)

image = Image.open('./Images/Figure 3.jpg')
st.image(image, caption = 'Schematics for the pavement modelling')

#Converter
inch2meter = 0.0254
meter2inch = 39.3701

# Page Title
# st.title('Prediction of Saturation Changes during Inundation')
st.set_option('deprecation.showPyplotGlobalUse', False)

## Loading ML Models
edge_sat_sur_model, edge_sat_base_model, edge_sat_sga_model, edge_pt_model, edge_rt_model, edge_vadose_model,\
           wp_sat_sur_model, wp_sat_base_model, wp_sat_sga_model, wp_pt_model, wp_rt_model, wp_vadose_model = load_models()

## Deal with input

st.sidebar.title('Input Parameters:')
st.sidebar.header('Thickness of the AC layer (inch)')
surT = st.sidebar.slider('', min_value = 3, max_value = 26, step = 1, value = 5)
st.sidebar.header('Thickness of the Base layer (inch)')
baseT = st.sidebar.slider('', min_value = 6, max_value = 28, step = 1, value = 12)
total = 120 + surT + baseT
st.sidebar.header('Groundwater table: Depth from top of pavement (inch)')
gwt = st.sidebar.slider("", min_value = surT+baseT+20, max_value = surT+baseT+60, step = 5, value=surT+baseT+40)
st.sidebar.markdown(f':red[**{gwt-surT-baseT} in**] from top of subgrade')

st.sidebar.header('Subgrade Type (AASHTO)')
sg_type = st.sidebar.selectbox("", ('A-1-b', 'A-2-4', 'A-4', 'A-6', 'A-7'))

st.sidebar.header('Flood Peak Time (h)')
pi_time = st.sidebar.selectbox("", (6, 12, 36, 72))
st.sidebar.image('./Images/Figure 2.png')

prediction_state = st.sidebar.radio('Predication', ('set parameters', 'show predictions'))

# Parameters
soil_params = {}

soil_params['A-1-b'] = {
    'theta_r': 0.045,
    'theta_s': 0.43,
    'a': 0.6665,
    'n': 2.68,
    'm': 0.6268
}

soil_params['A-2-4'] = {
    'theta_r': 0.025,
    'theta_s': 0.403,
    'a': 0.0383,
    'n': 1.3774,
    'm': 0.2740
}

soil_params['A-4'] = {
    'theta_r': 0.01,
    'theta_s': 0.439,
    'a': 0.0314,
    'n': 1.1804,
    'm': 0.1528
}

soil_params['A-5'] = {
    'theta_r': 0.01,
    'theta_s': 0.439,
    'a': 0.0314,
    'n': 1.1804,
    'm': 0.1528
}

soil_params['A-6'] = {
    'theta_r': 0.01,
    'theta_s': 0.614,
    'a': 0.0265,
    'n': 1.1033,
    'm': 0.0936
}

soil_params['A-7'] = {
    'theta_r': 0.01,
    'theta_s': 0.520,
    'a': 0.0367,
    'n': 1.1012,
    'm': 0.0919
}


sg_type_mapping = {'A-1-b': 0, 'A-2-4': 1, 'A-4': 2, 'A-6': 3, 'A-7': 4}

# Input for Prediction
input_features = pd.DataFrame(columns=['sur_thickness','base_thickness','subgrade_type', \
                                            'groundwater_table', 'peak_inundation_time',\
                                            'theta_r', 'theta_s', 'a', 'n', 'm'])

input_dict = {
    'sur_thickness':surT*inch2meter,
    'base_thickness': baseT*inch2meter,
    'subgrade_type': sg_type_mapping[sg_type],
    'groundwater_table': gwt-surT-baseT,
    'peak_inundation_time':pi_time,
    'theta_r': soil_params[sg_type]['theta_r'],
    'theta_s': soil_params[sg_type]['theta_s'],
    'a': soil_params[sg_type]['a'],
    'n': soil_params[sg_type]['n'],
    'm': soil_params[sg_type]['m']
    }

input_features = pd.DataFrame(input_dict, index=[0])

## Prediction and Visualization

# print(input_features)

st.subheader('Location: ')
show_option = st.selectbox("", ('Pavement Edge', 'Wheel Path'))

if prediction_state == 'show predictions' and show_option == 'Pavement Edge':
    edge_sat_sga_pred = edge_sat_sga_model.predict(input_features)
    edge_sat_base_pred = edge_sat_base_model.predict(input_features)
    edge_sat_sur_pred = edge_sat_sur_model.predict(input_features)
    edge_pt_pred = edge_pt_model.predict(input_features)[0]
    edge_rt_pred = edge_rt_model.predict(input_features)[0]
    edge_vadose_pred = edge_vadose_model.predict(input_features)
    if edge_vadose_pred[0][2] > 0:
        edge_vadose_pred[0][2] = - edge_vadose_pred[0][2]
    # print(edge_vadose_pred)
    st.markdown(f'The predicted peak saturation time for pavement edge is **:red[{math.ceil(edge_pt_pred)} h]**, and the restoration time for pavement edge is **:red[{math.ceil(edge_rt_pred)} h]**')
    X1 = np.arange(0, math.ceil(edge_pt_pred), 1, dtype=int)
    Y1_sgb = np.arange(0, (120-gwt)*inch2meter, 0.5*inch2meter)
    Y1_sga = np.arange((120-gwt)*inch2meter, 120*inch2meter, 0.5*inch2meter)
    Y1_base = np.arange(120*inch2meter, (120+baseT)*inch2meter, 0.5*inch2meter)
    Y1_sur = np.arange((120+baseT)*inch2meter, (120+baseT+surT)*inch2meter, 0.5*inch2meter)

    time1 = st.slider("Time past after inundation begins at pavement edge (h)", 0, 144)

    if time1 > math.ceil(edge_rt_pred):
        time1 = math.ceil(edge_rt_pred)
    divider_height = (120-gwt)*inch2meter
    Z1_sgb = np.ones(len(Y1_sgb))*soil_params[sg_type]['theta_s']
    Z1_sga = get_equation_value2(time1+1, Y1_sga, edge_sat_sga_pred, soil_params[sg_type]['theta_r'], soil_params[sg_type]['theta_s'], divider_height)
    Z1_sga = modify_points(Z1_sga, 100)
    Z1_base = get_equation_value1(time1+1, Y1_base, edge_sat_base_pred, 0.0065, 0.2)
    Z1_base = modify_points(Z1_base, Z1_sga[-1])
    Z1_sur = get_equation_value1(time1+1, Y1_sur, edge_sat_sur_pred, 0.001, 0.03)
    Z1_sur = modify_points(Z1_sur, Z1_base[-1])

    Y1 = np.concatenate([Y1_sgb, Y1_sga, Y1_base, Y1_sur])
    Z1 = np.concatenate([Z1_sgb, Z1_sga, Z1_base, Z1_sur])

    figure1 = get_figure(Y1, Z1, 'edge', total, surT, baseT, gwt)
    st.pyplot(figure1)

st.divider()

if prediction_state == 'show predictions' and show_option == 'Wheel Path':

    wp_sat_sur_pred = wp_sat_sur_model.predict(input_features)
    wp_sat_base_pred = wp_sat_base_model.predict(input_features)
    wp_sat_sga_pred = wp_sat_sga_model.predict(input_features)
    wp_pt_pred = wp_pt_model.predict(input_features)[0]
    wp_rt_pred = wp_rt_model.predict(input_features)[0]
    wp_vadose_pred = wp_vadose_model.predict(input_features)
    if wp_vadose_pred[0][2] > 0:
        wp_vadose_pred[0][2] = - wp_vadose_pred[0][2]
    print(wp_vadose_pred)
    st.markdown(f'The predicted peak saturation time for pavement wheel path is **:red[{math.ceil(edge_pt_pred)} h]**, and the restoration time for pavement wheel path is **:red[{math.ceil(edge_rt_pred)} h]**')
    X2 = np.arange(0, math.ceil(wp_pt_pred), 1, dtype=int)
    Y2_sgb = np.arange(0, (120-gwt)*inch2meter, 0.5*inch2meter)
    Y2_sga = np.arange((120-gwt)*inch2meter, 120*inch2meter, 0.5*inch2meter)
    Y2_base = np.arange(120*inch2meter, (120+baseT)*inch2meter, 0.5*inch2meter)
    Y2_sur = np.arange((120+baseT)*inch2meter, (120+baseT+surT)*inch2meter, 0.5*inch2meter)

    time2 = st.slider("Time past after inundation begins at wheel path (h)", 0, 144)

    if time2 > math.ceil(wp_rt_pred):
        time2 = math.ceil(wp_rt_pred)

    divider_height = (120-gwt)*inch2meter
    Z2_sgb = np.ones(len(Y2_sgb))*soil_params[sg_type]['theta_s']
    Z2_sga = get_equation_value2(time2+1, Y2_sga, wp_sat_sga_pred, soil_params[sg_type]['theta_r'], soil_params[sg_type]['theta_s'], divider_height)
    Z2_sga = modify_points(Z2_sga, 100)
    Z2_base = get_equation_value1(time2+1, Y2_base, wp_sat_base_pred, 0.0065, 0.2)
    Z2_base= modify_points(Z2_base, Z2_sga[-1])
    Z2_sur = get_equation_value1(time2+1, Y2_sur, wp_sat_sur_pred, 0.001, 0.03)
    Z2_sur= modify_points(Z2_sur, Z2_base[-1])

    Y2 = np.concatenate([Y2_sgb, Y2_sga, Y2_base, Y2_sur])
    Z2 = np.concatenate([Z2_sgb, Z2_sga, Z2_base, Z2_sur])
    # print(Y1)
    # print(Z1)
    figure2 = get_figure(Y2, Z2, 'wp', total, surT, baseT, gwt)
    st.pyplot(figure2)
    ## Visualization of Predictions

    # print(wp_sat_pred)