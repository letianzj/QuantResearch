import numpy as np
from scipy.stats import norm
import xlwings as xw

# https://en.wikipedia.org/wiki/Greeks_(finance)
@xw.func
def bsm(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16, CP='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = 0.0
    if CP.lower() == 'call':
        result = (S * np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    if CP.lower() == 'put':
        result = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0))

    return result

@xw.func
def bsm_delta(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16, CP='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = 0.0
    if CP.lower() == 'call':
        result = np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0)
    if CP.lower() == 'put':
        result = - np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0)

    return result

@xw.func
def bsm_vega(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = S * np.exp(-q * T) * norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    # result2 = K * np.exp(-r * T) *  norm.pdf(d2, 0.0, 1.0) * np.sqrt(T)
    # assert result == result2, 'vega failed'
    return result

@xw.func
def bsm_theta(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16, CP='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = 0.0
    if CP.lower() == 'call':
        result = - np.exp(-q * T) * S * norm.pdf(d1, 0.0, 1.0) * sigma / 2 / np.sqrt(T) \
                 - r * K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0) \
                 + q * S * np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0)
    if CP.lower() == 'put':
        result = - np.exp(-q * T) * S * norm.pdf(-d1, 0.0, 1.0) * sigma / 2 / np.sqrt(T) \
                 + r * K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) \
                 - q * S * np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0)

    return result

@xw.func
def bsm_rho(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16, CP='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = 0.0
    if CP.lower() == 'call':
        result = K * T * np.exp(-r*T) * norm.cdf(d2, 0.0, 1.0)
    if CP.lower() == 'put':
        result = -K * T * np.exp(-r*T) * norm.cdf(-d2, 0.0, 1.0)

    return result


@xw.func
def bsm_gamma(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = np.exp(-q*T) * norm.pdf(d1) / S / sigma / np.sqrt(T)
    # result = K * np.exp(-r*T) * np.pdf(d2) / S / S / sigma / np.sqrt(T)
    return result

@xw.func
def bsm_vanna(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16):
    """d^2V/dS/dsigma"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = -np.exp(-q*T) * norm.pdf(d1) * d2 / sigma
    return result

@xw.func
def bsm_volga(S, K, T = 1.0, r = 0.0, q = 0.0, sigma = 0.16):
    """d^2V/dsigma^2"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = S*np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) * d1 * d2 / sigma
    return result

@xw.func
def black76(F, K, T, r, sigma, CP='call'):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = 0.0
    if CP.lower() == 'call':
        result = np.exp(-r*T)*( F*norm.cdf(d1) - K*norm.cdf(d2) )
    else:
        result = np.exp(-r*T)*( K*norm.cdf(-d2) - F*norm.cdf(-d1) )

    return result

@xw.func
def black76_delta(F, K, T, r, sigma, CP='call'):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = 0.0
    if CP.lower() == 'call':
        result = np.exp(-r*T)*norm.cdf(d1)
    else:
        result = -np.exp(-r*T)*norm.cdf(-d1)

    return result

@xw.func
def black76_vega(F, K, T, r, sigma):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
    # result = K * np.exp(-r*T) * norm.pdf(d2)*np.sqrt(T)

    return result

@xw.func
def black76_theta(F, K, T, r, sigma, CP='call'):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = 0.0
    if CP.lower() == 'call':
        result = -F*np.exp(-r*T)*norm.pdf(d1)*sigma/2/np.sqrt(T) \
                 - r*K*np.exp(-r*T)*norm.cdf(d2) \
                 + r*F*np.exp(-r*T)*norm.cdf(d1)
    else:
        result = -F * np.exp(-r * T) * norm.pdf(-d1) * sigma / 2 / np.sqrt(T) \
                 + r * K * np.exp(-r * T) * norm.cdf(-d2) \
                 - r * F * np.exp(-r * T) * norm.cdf(-d1)

    return result


@xw.func
def black76_rho(F, K, T, r, sigma, CP='call'):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = 0.0
    if CP.lower() == 'call':
        result = K*T*np.exp(-r*T)*norm.cdf(d2)
    else:
        result = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    return result


@xw.func
def black76_gamma(F, K, T, r, sigma):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = np.exp(-r*T)*norm.pdf(d1)/F/sigma/np.sqrt(T)
    # result = k*np.exp(-r*T)*norm.pdf(d2)/F/F/sigma/np.sqrt(T)

    return result

@xw.func
def black76_vanna(F, K, T, r, sigma):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = -np.exp(-r*T)*norm.pdf(d1)*d2/sigma

    return result

@xw.func
def black76_volga(F, K, T, r, sigma):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    result = F*np.exp(-r*T)*norm.pdf(d1)*np.sqrt(T)*d1*d2/sigma

    return result