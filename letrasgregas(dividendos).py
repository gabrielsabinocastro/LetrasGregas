#Bibliotecas

import numpy as np
import scipy.stats as si
#import sympy as sy
#import sympy.stats as systats

##################################################################################

#DELTA

def delta_call(S, K, T, r, q, sigma):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    delta_call = si.norm.cdf(d1, 0.0, 1.0) * np.exp(-q*T)
    
    return delta_call

def delta_put(S, K, T, r, q, sigma):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    delta_put = si.norm.cdf(-d1, 0.0, 1.0) * np.exp(-q*T)
    
    return -delta_put

def delta(S, K, T, r, q, sigma, option = 'call'):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = si.norm.cdf(d1, 0.0, 1.0) * np.exp(-q*T)
    if option == 'put':
        result = -si.norm.cdf(-d1, 0.0, 1.0) * np.exp(-q*T)
        
    return result

delta(100, 50, 1, 0.05, 0.1, 0.25, option = 'put') #para retornar o valor da put da ação que paga dividendos conhecidos de taxa q

delta(100, 50, 1, 0.05, 0.1, 0.25, option = 'call') #para retornar o valor da call da ação que paga dividendos conhecidos de taxa q

##################################################################################

#THETA

def theta_call(S, K, T, r, q, sigma):
    
    d1 = (np.log(S / K) + (r -q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r -q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    theta = (-sigma * S * prob_density * np.exp(-q*T)) / (2 * np.sqrt(T)) + q * S * si.norm.cdf(d1, 0.0, 1.0) * np.exp(-q*T) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    
    return theta

def theta_put(S, K, T, r, q, sigma):
    
    d1 = (np.log(S / K) + (r -q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r -q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    theta = (-sigma * S * prob_density * np.exp(-q*T)) / (2 * np.sqrt(T)) - q * S * si.norm.cdf(-d1, 0.0, 1.0) * np.exp(-q*T) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta

def theta(S, K, T, r, q, sigma, option = 'call'):
    
    d1 = (np.log(S / K) + (r -q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r -q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    if option == 'call':
        theta = (-sigma * S * prob_density * np.exp(-q*T)) / (2 * np.sqrt(T)) + q * S * si.norm.cdf(d1, 0.0, 1.0) * np.exp(-q*T) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    if option == 'put':    
        theta = (-sigma * S * prob_density * np.exp(-q*T)) / (2 * np.sqrt(T)) - q * S * si.norm.cdf(-d1, 0.0, 1.0) * np.exp(-q*T) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta

theta(110, 100, 2, 0.05, 0.1, 0.25, option = 'put') #para retornar o valor da put da ação que paga dividendos conhecidos de taxa q
theta(110, 100, 2, 0.05, 0.1, 0.25, option = 'call') #para retornar o valor da call da ação que paga dividendos conhecidos de taxa q

##################################################################################

#GAMA

def gamma(S, K, T, r, q, sigma):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    gamma = (prob_density * np.exp(-q*T)) / (S * sigma * np.sqrt(T))
    
    return gamma

gamma(110, 100, 1, 0.05, 0.1, 0.25) #para retornar o valor de gama da ação que paga dividendos conhecidos de taxa q

##################################################################################

#VEGA

def vega(S, K, T, r, q, sigma):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    vega = S * prob_density * np.sqrt(T) * np.exp(-q*T)
    
    return vega

vega(105, 100, 1, 0.05, 0.1, 0.25) #para retornar o valor de vega da ação que paga dividendos conhecidos de taxa q

##################################################################################

#RHO

def rho_call(S, K, T, r, q, sigma):
    
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    rho = T * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    
    return rho

def rho_put(S, K, T, r, q, sigma):
    
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    rho = -T * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return rho

def rho(S, K, T, r, q, sigma, option = 'call'):
    
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        rho = T * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    if option == 'put':
        rho = -T * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
        
    return rho

rho(110, 100, 2, 0.05, 0.1, 0.25, option = 'put') #para retornar o valor da put da ação que paga dividendos conhecidos de taxa q
rho(110, 100, 2, 0.05, 0.1, 0.25, option = 'call') #para retornar o valor da call da ação que paga dividendos conhecidos de taxa q

##################################################################################