import pandas as pd
import numpy as np
data = pd.read_csv('/home/fahd/Downloads/data_set/advertising.csv')

def prediction(x,w,b,mu,std) :
    x_norm = (x - mu) / std
    prediction = x_norm @ w + b
    return prediction

def z_score_norm(x) :
    mu = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    x_norm = (x - mu)/std
    return x_norm,mu,std

def gradient_cost(x,y,w,b) :

    f_wb_i = (x @ w) + b - y
    f_wb_i = f_wb_i**2
    cost = np.sum(f_wb_i)
    m,_ = np.shape(x)
    total_cost = cost/(2*m)
    return total_cost

def deriv_gradient(x,y,w,b) :
    m,n = np.shape(x)
    dj_dw = np.zeros(n)
    dj_db = 0
    f_wb = (x @ w) + b 
    diff = f_wb - y
    dj_dw = (x.T @ diff) / m
    dj_db = np.sum(diff) / m
    return dj_dw,dj_db

def gradient_descent(x,y,w,b,iterations,alpha,deriv_gradient,gradient_cost,z_score_norm) :
    x_norm,_,_ = z_score_norm(x)
    J_hist = np.array([])
    for i in range(iterations+1) :
        dj_dw, dj_db = deriv_gradient(x_norm,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        cost = gradient_cost(x_norm,y,w,b)
        J_hist = np.append(J_hist,cost)
        if i % 1000 == 0 :
            print(f'Iteration {i:4d} : Cost : {J_hist[-1]}')
    return w,b


X = data[['TV','Radio','Newspaper']]
Y = data['Sales']
x = np.array(X)
y = np.array(Y)
m,n = np.shape(x)
w_init = np.zeros(n)
b_init = 0
alpha = 1
iterations = 5000
w_final , b_final = gradient_descent(x,y,w_init,b_init,iterations,alpha,deriv_gradient,gradient_cost,z_score_norm)
_, mu,std = z_score_norm(x)

new_campaign = np.array([110.0, 25.0, 0.0])
profit = prediction(new_campaign,w_final,b_final,mu,std)

print(f"Result predicted : {profit:.2f}K units sold")