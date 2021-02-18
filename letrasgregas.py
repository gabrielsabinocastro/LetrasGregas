#Bibliotecas

import numpy as np
import scipy.stats as si

##################################################################################

#DELTA

def delta(S, K, T, r, sigma, option = 'call'):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = si.norm.cdf(d1, 0.0, 1.0)
    if option == 'put':
        result = -si.norm.cdf(-d1, 0.0, 1.0)
        
    return result

delta(49, 50, 0.3846, 0.05, 0.2, option = 'put') #para retornar o valor da put

delta(49, 50, 0.3846, 0.05, 0.2, option = 'call') #para retornar o valor da call

##################################################################################

#THETA

def theta(S, K, T, r, sigma, option = 'call'):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    if option == 'call':
        theta = (-sigma * S * prob_density) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    if option == 'put':    
        theta = (-sigma * S * prob_density) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta

theta(49, 50, 0.3846, 0.05, 0.2, option = 'put') #para retornar o valor da put
theta(49, 50, 0.3846, 0.05, 0.2, option = 'call') #para retornar o valor da call

##################################################################################

#GAMA

def gamma(S, K, T, r, sigma):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    gamma = prob_density / (S * sigma * np.sqrt(T))
    
    return gamma

gamma(49, 50, 0.3846, 0.05, 0.2) #para retornar o valor de gama

##################################################################################

#VEGA

def vega(S, K, T, r, sigma):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    vega = S * prob_density * np.sqrt(T)
    
    return vega

vega(49, 50, 0.3846, 0.05, 0.2) #para retornar o valor de vega

##################################################################################

#RHO

def rho(S, K, T, r, sigma, option = 'call'):
    
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        rho = T * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    if option == 'put':
        rho = -T * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
        
    return rho

rho(49, 50, 0.3846, 0.05, 0.2, option = 'put') #para retornar o valor da put
rho(49, 50, 0.3846, 0.05, 0.2, option = 'call') #para retornar o valor da call

##################################################################################