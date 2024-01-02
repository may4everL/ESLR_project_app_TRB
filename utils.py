import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pickle


#Converter
inch2meter = 0.0254
meter2inch = 39.3701

def get_equation_value1(x, Y, coef, min, max):
    Z = []
    for idx in range(len(Y)):
        term = np.array([1, x, Y[idx], x*Y[idx]])
        val = np.dot(coef, term)
        if isinstance(val, np.ndarray):
            val = val[0]
        # print(val)
        if val < min:
            val = min
        if val > max:
            val = max
        # print(np.dot(coef, term))
        Z.append(val)
    # print(Z)
    Z = np.array(Z)
    
    return Z

def get_equation_value2(x, Y, coef, min, max, divider_height):
    Z = []  
    for idx in range(len(Y)):
        if Y[idx] <= divider_height:
            Z.append(max)
        else:
            term = np.array([1, x, Y[idx], x*Y[idx], Y[idx]**2, x*Y[idx]**2])
            # print(coef.shape)
            # print(term.shape)
            val = np.dot(coef, term)
            if isinstance(val, np.ndarray):
                val = val[0]
            # print(val)
            if val < min:
                val = min
            if val > max:
                val = max
            # print(np.dot(coef, term))
            Z.append(val)
    # print(Z)
    Z = np.array(Z)
    return Z

def modify_points(Z, max):
    modified_Z = []
    for ele in Z:
        if ele > max:
            modified_Z.append(max)
        else:
            modified_Z.append(ele)
            max = ele
    return modified_Z

def get_figure(Y, Z, loc, total, surT, baseT, gwt):
    figure, ax = plt.subplots()
    Y = Y * meter2inch
    dep = [total - y for y in Y]
    ax.plot(Z, dep, marker='o', markerfacecolor='none', markeredgecolor='r', color='r')
    ax.axhline(y=surT, color='black', linestyle='--')
    ax.text(x=0.22, y=surT+7, s='Top of Base/Subbase Layer', color='black')
    ax.axhline(y=surT+baseT, color='green', linestyle='--')
    ax.text(x=0.22, y=surT+baseT+7, s='Top of Subgrade', color='green')
    ax.axhline(y=surT+baseT+gwt, color='blue', linestyle='--')
    ax.text(x=0.22, y=surT+baseT+gwt+7, s='Groundwater Table', color='blue')
    ax.set_ylim(0, total)
    ax.set_yticks(np.arange(0, total, 10))
    ax.invert_yaxis()
    ax.set_ylabel('Depth (in)')
    ax.set_xlabel('Volumetric Moisture Content')
    ax.xaxis.set_tick_params(top=True, labeltop=True)
    ax.xaxis.set_label_position('top')
    if loc == 'edge':
        ax.set_title("Volumetric Moisture Content Along Pavement Edge")
    else:
        ax.set_title("Volumetric Moisture Content Along Pavement Wheel Path")
    return figure

# Polynomial Features

def create_features(h, t, n = 2):
    features = {}
    for i in range(n+1):
        if i > 0:
            features[f"h^{i}*t^0"] = np.multiply(h**i, t**0).flatten()
        features[f"h^{i}*t^1"] = np.multiply(h**i, t**1).flatten()
    return features

def load_models():
    edge_sat_sur_model = pickle.load(open(f'./Models/edge_sat_sur_model.obj', 'rb'))
    edge_sat_base_model = pickle.load(open(f'./Models/edge_sat_base_model.obj', 'rb'))
    edge_sat_sga_model = pickle.load(open(f'./Models/edge_sat_sga_model.obj', 'rb'))
    edge_pt_model = pickle.load(open(f'./Models/edge_peak_time_model.obj', 'rb'))
    edge_rt_model = pickle.load(open(f'./Models/edge_rest_time_model.obj', 'rb'))
    edge_vadose_model = pickle.load(open(f'./Models/edge_sat_vadose_model.obj', 'rb'))
    wp_sat_sur_model = pickle.load(open(f'./Models/wp_sat_sur_model.obj', 'rb'))
    wp_sat_base_model = pickle.load(open(f'./Models/wp_sat_base_model.obj', 'rb'))
    wp_sat_sga_model = pickle.load(open(f'./Models/wp_sat_sga_model.obj', 'rb'))
    wp_pt_model = pickle.load(open(f'./Models/wp_peak_time_model.obj', 'rb'))
    wp_rt_model = pickle.load(open(f'./Models/wp_rest_time_model.obj', 'rb'))
    wp_vadose_model = pickle.load(open(f'./Models/wp_sat_vadose_model.obj', 'rb'))
    return edge_sat_sur_model, edge_sat_base_model, edge_sat_sga_model, edge_pt_model, edge_rt_model,edge_vadose_model,\
           wp_sat_sur_model, wp_sat_base_model, wp_sat_sga_model, wp_pt_model, wp_rt_model, wp_vadose_model