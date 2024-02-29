import logging
LOGGER = logging.getLogger(__name__)

CONFIG_PATH = 'config.json'

from utils import read_config
settings = read_config(CONFIG_PATH)
settings = settings.RISK_LIMITS

from portfolio import Portfolio

def risk_validator(portfolio: Portfolio, pricing: dict, new_order: dict, underlying_price: float, existing_orders: list = []):
    
    # theo in currency (btc/eth)
    theo = pricing['price'] / underlying_price
    
    # validate net delta
    delta = portfolio.net_delta
    def direction_to_mult(dir):
        return 1 if dir == 'buy' else -1
    new_delta = delta + pricing['delta'] * new_order['amount'] * direction_to_mult(new_order['direction'])

    if abs(new_delta) >= settings['MAX_NET_DELTA']:
        return False
    elif (new_order['amount'] == 1 and settings.ONLY_SELL == True) or (new_order['direction'] == -1 and settings.ONLY_BUY == True):
        return False    
    elif portfolio.tot_size_of_options >= settings['TOT_POSITION_LIMIT']:
        return False   
    elif portfolio.tot_size_of_options_short > settings['MAX_TOT_SIZE_OPTIONS_SHORT'] and new_order['direction'] == 'sell':
        return False 
    elif portfolio.tot_size_of_options_long > settings['MAX_TOT_SIZE_OPTIONS_LONG'] and new_order['direction'] == 'buy':
        return False
    
    return True