#Bibliotecas

import numpy as np
import scipy.stats as si  #para calcular distribuição normal cumulativa

import yfinance as yf
import pandas_datareader.data as web
yf.pdr_override()

# 1 - AS VARIÁVEIS #


#valor do último preço de fechamento da ação

ticker = 'ABEV.3'  #defina o código da ação na variável ticker

year_data = web.get_data_yahoo(ticker, period="1y")["Adj Close"]
S = year_data.iloc[-1]  

S #para retornar o último preço de fechamento da ação

##################################################################################

#valor da taxa livre de risco SELIC

codigo_bcb = 1178  #1178 para consultar série histórica selic a.a.
url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
df = pd.read_json(url)
df['data'] = pd.to_datetime(df['data'], dayfirst=True)
df["valor"].iloc[-1]

r = df["valor"].iloc[-1]/100

r #para retornar a taxa livre de riscos SELIC ano



##################################################################################

#TODO: preparar códigos para imprimir volatilidade do ativo definido, além de consultar os preços de mercado para suas opções:  (sigma, call, put) 

##################################################################################

# 2 AS GREGAS #

#DELTA

def delta(S, K, T, r, q, sigma, option = 'call'):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = si.norm.cdf(d1, 0.0, 1.0) * np.exp(-q*T)
    if option == 'put':
        result = -si.norm.cdf(-d1, 0.0, 1.0) * np.exp(-q*T)
        
    return result

delta(49, 50, 0.3846, 0.05, 0.1, 0.2, option = 'put') #para retornar o valor da put da ação que paga dividendos conhecidos de taxa q

delta(49, 50, 0.3846, 0.05, 0.1, 0.2, option = 'call') #para retornar o valor da call da ação que paga dividendos conhecidos de taxa q

##################################################################################

#THETA

def theta(S, K, T, r, q, sigma, option = 'call'):
    
    d1 = (np.log(S / K) + (r -q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r -q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    if option == 'call':
        theta = (-sigma * S * prob_density * np.exp(-q*T)) / (2 * np.sqrt(T)) + q * S * si.norm.cdf(d1, 0.0, 1.0) * np.exp(-q*T) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    if option == 'put':    
        theta = (-sigma * S * prob_density * np.exp(-q*T)) / (2 * np.sqrt(T)) - q * S * si.norm.cdf(-d1, 0.0, 1.0) * np.exp(-q*T) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta

theta(49, 50, 0.3846, 0.05, 0.1, 0.2, option = 'put') #para retornar o valor da put da ação que paga dividendos conhecidos de taxa q
theta(49, 50, 0.3846, 0.05, 0.1, 0.2, option = 'call') #para retornar o valor da call da ação que paga dividendos conhecidos de taxa q

##################################################################################

#GAMA

def gamma(S, K, T, r, q, sigma):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    gamma = (prob_density * np.exp(-q*T)) / (S * sigma * np.sqrt(T))
    
    return gamma

gamma(49, 50, 0.3846, 0.05, 0.1, 0.2) #para retornar o valor de gama da ação que paga dividendos conhecidos de taxa q

##################################################################################

#VEGA

def vega(S, K, T, r, q, sigma):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
    
    vega = S * prob_density * np.sqrt(T) * np.exp(-q*T)
    
    return vega

vega(49, 50, 0.3846, 0.05, 0.1, 0.2) #para retornar o valor de vega da ação que paga dividendos conhecidos de taxa q

##################################################################################

#RHO

def rho(S, K, T, r, q, sigma, option = 'call'):
    
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        rho = T * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    if option == 'put':
        rho = -T * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
        
    return rho

rho(49, 50, 0.3846, 0.05, 0.1, 0.2, option = 'put') #para retornar o valor da put da ação que paga dividendos conhecidos de taxa q
rho(49, 50, 0.3846, 0.05, 0.1, 0.2, option = 'call') #para retornar o valor da call da ação que paga dividendos conhecidos de taxa q

##################################################################################