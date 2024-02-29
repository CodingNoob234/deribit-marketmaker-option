import numpy as np
from scipy import optimize
import time
import math
from pricing_engine import BSOption

from scipy.interpolate import splrep, BSpline

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# configure logging
# from configure_logger import configure_logger
# configure_logger()
import logging
LOGGER = logging.getLogger(__name__) 

class VolManager:
    def __init__(self, contracts_info: dict, contracts_now: dict = None, r: float = 0, q: float = 0):
        self.r = r # interest rate
        self.q = q # dividend

        # required for volatility modelling
        self.contracts_info = contracts_info # contains contract specifications
        self.contracts_now = contracts_now# contains orderbook and deribit greeks
        
        self.T_year_seconds = 365.25 * 24 * 3600
        
        self.vols = {}
        
    def estimate_vol(self):
        # get implied vols
        self.ivs = {k:v for k,v in zip(self.contracts_info.keys(), self.get_implied_vols())}
    
    def get_implied_vol(self, contract: dict, contract_now: dict, type = 'm'):
        # get option-specific specification
        instrument_name = contract['instrument_name']
        strike = contract['strike']
        underlying = contract_now['underlying_price']
        T = (contract['expiration_timestamp']/1000 - time.time()) / self.T_year_seconds
        
        # get market price to determine
        if type == 'm':
            # if no bids, mid market is difficult to determine
            if len(contract_now['bids']) == 0 or len(contract_now['asks']) == 0:
                return contract_now['mark_iv']/100
            else:
                best_bid = contract_now['bids'][0][0]
                best_ask = contract_now['asks'][0][0]
                market_price = .5 * (best_bid + best_ask)
            
        # if only one side
        elif type == 'b':
            if len(contract_now['bids']) == 0:
                return np.nan
            else:
                market_price = contract_now['bids'][0][0]          
        elif type == 'a':
            if len(contract_now['asks']) == 0:
                return np.nan
            else:
                market_price = contract_now['asks'][0][0]
        # if invalid
        else:
            raise ValueError(f'invalid \'type\' provided: {str(type)}')
    
        # find sigma which minimizes market pricing
        bs = BSOption(underlying, strike, T, self.r, None, self.q)
        def vol_loss(sigma):
            # set new vol estimate in pricing engine
            bs.sigma = sigma
            theo = bs.price(instrument_name[-1])/underlying
            return abs(theo - market_price)
        initial_guess = contract_now['mark_iv']/100
        result = optimize.minimize(vol_loss, initial_guess)
        return max(result.x[0],0)
    
    def get_implied_vols(self, type='m'):
        vols = [
            self.get_implied_vol(self.contracts_info[c], self.contracts_now[c], type) 
            for c in self.contracts_info
        ]
        vols = dict(zip(self.contracts_info.keys(), vols))
        if type == 'm':
            self.vols = vols
        return vols
    
    def fit(self, settings):
        self.model = {}
        
        for maturity in settings.TRADE_SETTINGS:
            
            # compute vol reference points for each sigma point
            maturity_settings = settings.TRADE_SETTINGS[maturity]['VOL_MODELLING']
            atm = maturity_settings['ATM']/100
            sps = []
            vols = []
            mults: dict = maturity_settings['MULT']
            for sp in mults.keys():
                
                vols.append(atm * mults[sp]/100)
                sps.append(float(sp))
            
            # fit spline model through reference points
            tck = splrep(sps, vols) # s=6
            
            # store model
            self.model[maturity] = {
                'spline': BSpline(*tck),
                'atm_vol': atm,
            }
        
    def predict(self, contract, contract_now):
        instrument_name = contract['instrument_name']
        strike = contract['strike']
        maturity = '-'.join(instrument_name.split('-')[:2])
        
        model_spec = self.model[maturity]
        atm_vol = model_spec['atm_vol']
        underlying_price = contract_now['underlying_price']
        seconds_to_expiration = contract['expiration_timestamp']/1000 - time.time()
        sp = self.price_to_sp(strike, underlying_price, atm_vol, seconds_to_expiration)
        
        return (
            sp, 
            self.model[maturity]['spline'](sp), 
            self.model[maturity]['spline'].derivative()(sp) / (underlying_price * atm_vol * math.sqrt(seconds_to_expiration / (365*24*3600)))
        )
    
    def price_to_sp(self, strike, underlying, vol, seconds_to_expiration):
        return (strike - underlying) / (underlying * vol * math.sqrt(seconds_to_expiration / (365*24*3600)))

    def model_vol(self, _, settings):
        
        # get implied vols
        self.get_implied_vols()
        
        # refit based on new underlying or coefficients
        self.fit(settings)
        
        # predict all contracts
        vols = {}
        vols_2nd = {}
        sps = {}
        
        for instrument_name in self.contracts_info.keys():
            
            contract_info = self.contracts_info[instrument_name]
            contract_now = self.contracts_now[instrument_name]
            
            sp, vol_pred, vol_pred_2nd = self.predict(contract_info, contract_now)
            vols[instrument_name] = float(vol_pred)
            vols_2nd[instrument_name] = float(vol_pred_2nd)
            sps[instrument_name] = float(sp)

        self.vols = vols
        self.vols_2nd = vols_2nd
        self.sps = sps
        return sps, vols