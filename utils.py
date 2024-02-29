# from configure_logger import configure_logger
# configure_logger()
import logging
LOGGER = logging.getLogger(__name__)

import json

class DictObject:
    def __init__(self, dict: dict = {}):
        self._dict = dict
        
    def __getattr__(self, __name: str):
        if __name in self._dict:
            return self._dict[__name]
        raise Exception(f'settings does not contain attribute "{__name}"')
  
# class DictObject:
#     def __init__(self, dict: dict = {}):
#         self._dict = dict
        
#     def __getattr__(self, __name: str):
#         if __name in self._dict:
#             val = self._dict[__name]
#             if type(val) == dict:
#                 return DictObject(val)
#             else:
#                 return val
#         raise Exception(f'settings does not contain attribute "{__name}"')

def read_config(config_path):
    """ Read config from specified 'config_path' """
    # open path
    with open(config_path, 'r') as config_file:   
        config = dict(json.load(config_file))

    LOGGER.info('config loaded')
    return DictObject(config)

def overwrite_settings(settings: DictObject, config_path: str):
    """ Overwrite settings dictionary at specified file path """
    with open(config_path, 'w') as config_file:
        json.dump(settings._dict, config_file)

def mid_market(b,a):
    """ compute mid market for bid and ask """
    return .5 * (b+a)

def round_price_to_instrument_tick(price, tick, tick_list = None):
    """
    Round a given price to a certain tick.
    An extra variable can be passed called 'tick_list',
    as for options the ticksize can vary depending on the option price, 
    i.e. above a certain price, tick size decreases
    """
    if tick_list:
        base_tick = tick
        n = len(tick_list)
        if n > 0:
            i = 0
            while i < n:
                if price > tick_list[i]["above_price"]:
                    base_tick = tick_list[i]["tick_size"]
                    i += 1
                else:
                    break
        return round(round(price / base_tick) * base_tick, 8)
    return round(round(price / tick) * tick, 8)

def round_size_to_instrument_tick(size, min_size):
    """ Round the quoted size to a minimum size """
    return round(round(size / min_size) * min_size, 10)

def find_equivalent(order, orders):
    for order2 in orders:
        if compare_orders(order, order2):
            return order2['order_id']
    return False
    
def compare_orders(order1, order2):
    compare_keys = ['instrument_name', 'direction', 'label']
    for key in compare_keys:
        if order1[key] != order2[key]:
            return False
    return order2['order_id']

import time

class Counter:
    """ A class that counts the number of API requests """
    def __init__(self):
        self.n = []
        
        self.max1s = 0
        self.max5s = 0
        
        self.log_threshold_1s = 50
        self.log_threshold_5s = 100
        
    def count(self, **kwargs):
        self.n.append(time.time())
        self.avg_1s()
        self.avg_5s()
        
    def avg_5s(self):
        s = time.time()
        sum5s = len([i for i in self.n if i + 5 > s])
        if sum5s > self.max5s:
            self.max5s = sum5s
            LOGGER.debug(f'new 5s max: {sum5s}')
        if sum5s > self.log_threshold_5s:
            LOGGER.info(f'5s log warning: {sum5s}')
    
    def avg_1s(self):
        s = time.time()
        sum1s = len([i for i in self.n if i + 1 > s])
        if sum1s > self.max1s:
            self.max1s = sum1s
            LOGGER.debug(f'new 1s max: {sum1s}')
        if sum1s > self.log_threshold_1s:
            LOGGER.info(f'1s log warning: {sum1s}')