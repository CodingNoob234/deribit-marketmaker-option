# from configure_logger import configure_logger
# configure_logger()
import logging
LOGGER = logging.getLogger(__name__)

import numpy as np
from pricing_engine import BSOption

class Portfolio:
    def __init__(self, portfolio: dict):
        self.set_portfolio(portfolio)
        
    def set_portfolio(self, portfolio):
        self.portfolio = portfolio
        self.count_position_types()
        
    def count_position_types(self):        
        self.tot_size_of_options = sum([abs(p["size"]) for p in self.portfolio if abs(p["size"]) > 0])
        self.tot_size_of_options_call =  sum([p["size"] for p in self.portfolio if abs(p["size"]) > 0 and p["instrument_name"][-1] == "C"])
        self.tot_size_of_options_put =  sum([p["size"] for p in self.portfolio if abs(p["size"]) > 0 and p["instrument_name"][-1] == "P"])
        
        self.tot_number_of_options = len([p for p in self.portfolio if abs(p["size"]) > 0])
        self.tot_number_of_options_call = len([p for p in self.portfolio if abs(p["size"]) > 0 and p["instrument_name"][-1] == "C"])
        self.tot_number_of_options_put = len([p for p in self.portfolio if abs(p["size"]) > 0 and p["instrument_name"][-1] == "P"])
        
        self.tot_size_of_options_short = sum([abs(p["size"]) for p in self.portfolio if p["size"] < 0])
        self.tot_size_of_options_long =  sum([abs(p["size"]) for p in self.portfolio if p["size"] > 0])

        
    def compute_net_delta(self, options, vm):
        """
        arg:
            - options: contains all option information (including BS pricing instance)
        """
        net_delta = 0
        for position in self.portfolio:

            instrument_name = position['instrument_name']
            if instrument_name[-2] == "-":

                if instrument_name in options.keys():
                    bs_pricing: BSOption = options[instrument_name]['contract_pricing']
                    delta = bs_pricing.delta()
                    skew_delta = delta + (bs_pricing.vega()/bs_pricing.S) * (vm.vols_2nd[instrument_name] *100)
                    net_skew_delta = skew_delta - bs_pricing.price()/bs_pricing.S
                    net_skew_delta *= position['size']

                    LOGGER.debug(f'd size:{net_skew_delta}')
                    net_delta += net_skew_delta
                else:
                    LOGGER.warning(f'options {instrument_name} is not priced currently, using exchange delta: {position["delta"]}')
                    net_delta += position["delta"]

        self.net_delta = net_delta
        
def compute_delta(trade, options, vm):
    instrument_name = trade['instrument_name']
    if instrument_name[-2] == "-":

        if instrument_name in options.keys():
            bs_pricing: BSOption = options[instrument_name]['contract_pricing']
            delta = bs_pricing.delta()
            skew_delta = delta + (bs_pricing.vega()/bs_pricing.S) * (vm.vols_2nd[instrument_name] * 100)
            net_skew_delta = skew_delta - bs_pricing.price()/bs_pricing.S
            net_skew_delta *= trade['amount']
            LOGGER.debug(f'd size:{net_skew_delta}')
            return net_skew_delta
        else:
            LOGGER.info('option unavailable for pricing')
            raise ValueError(f'Option unavailable for pricing: {instrument_name}')
       
"""
Example of a portfolio

portfolio format =
[
    {'liquidity': 'T', 'self_trade': False, 'risk_reducing': False, 'order_type': 'limit', 'trade_id': '283752496', 
    'fee_currency': 'BTC', 'contracts': 0.1, 'underlying_price': 48724.33584422, 'reduce_only': False, 'post_only': False, 
    'api': False, 'mmp': False, 'instrument_name': 'BTC-14FEB24-51000-C', 'tick_direction': 0, 'fee': 8.75e-06, 'matching_id': None, 
    'order_id': '65516761994', 'trade_seq': 171, 'mark_price': 0.00046932, 'profit_loss': 0.0, 'amount': 0.1, 'index_price': 48712.24, 
    'direction': 'buy', 'price': 0.0007, 'iv': 65.91, 'state': 'filled', 'timestamp': 1707837703838}
]
"""