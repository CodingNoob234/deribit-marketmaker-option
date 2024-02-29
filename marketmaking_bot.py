#!/usr/bin/env python

from configure_logger import configure_logger
configure_logger()
import logging
LOGGER = logging.getLogger(__name__)

from typing import Any
import numpy as np
import math
import time
import utils
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

from deribitv2 import DeriBit
from vol_manager import VolManager
from portfolio import Portfolio
from pricing_engine import BSOption
from risk_validator import risk_validator

CONFIG_PATH = 'config.json'
TRANSACTION_FEE = .0003
MAX_PCT_FEE = .125
PERPETUAL_FEE = 2 * .0005 # buying and selling of perpetual contract

settings = utils.read_config(CONFIG_PATH)

# paramters for refreshing orders and time series
MIN_LOOP_TIME = settings.MIN_LOOP_TIME
RESTART_INTERVAL = 5
PERPETUAL_CONTRACT = settings.CURRENCY + "-PERPETUAL"

DIVIDEND = 0
INTEREST = 0

VALID_ORDER_KEYS = [
    'instrument_name',
    'order_type',
    'amount',
    'price',
    'post_only',
    'reduce_only',
    'label'
]

class MarketMakingBot:
    def __init__(self):
        
        self.DERIBIT = DeriBit(settings.TEST_EXCHANGE)
        self.PORTFOLIO = Portfolio(self.DERIBIT.getpositions(settings.CURRENCY, kind = "option"))

        # request and store ticker information
        self.hedge_instrument = self.DERIBIT.getinstrument(PERPETUAL_CONTRACT)

        # cancel all orders before start market making
        self.DERIBIT.cancelbylabel(settings.ORDER_LABEL)
        self.account_sum = self.DERIBIT.account_sum(settings.CURRENCY)
        
        # get current positions
        self.portfolio = self.DERIBIT.getpositions(settings.CURRENCY, kind = "option")
        self.private_option_trades = self.DERIBIT.getusertradesbycurrency(settings.CURRENCY, 'option', start_timestamp = (time.time() - 500)*1000)['trades']
        
        # select the desired options to market make on
        self.select_desired_options() # also initializes self.options
        LOGGER.info("Succesfully initialised MarketMakingBot")
        
        self.vm = VolManager({key:item['contract_info'] for key, item in self.options.items()})
        
        self.orders = []
        
        self.do_plot_vols = True
        
    def reload_config(self):
        global settings
        settings = utils.read_config(CONFIG_PATH)
        
    def select_desired_options(self):
        """ 
        Here, we select only those options for the closest maturity that
        are within an absolute strike point of 3. These options will be 
        continiously monitored and quoted if priced inefficiently.
        """
        index_price = self.DERIBIT.get_index_price(settings.CURRENCY.lower() + "_usd")["index_price"]
        all_options = self.DERIBIT.getinstruments(settings.CURRENCY, "option")

        # only options for specific maturities
        maturities = settings.TRADE_SETTINGS
        maturities = [m for m in maturities.keys() if maturities[m]['ON'] == True]
        self.maturities = maturities

        all_options = [
            c for c in all_options if len([k for k in maturities if k in c['instrument_name']])>0
        ]

        # init vol manager
        options = {}
        for option in all_options:
            
            # get instrument name
            instrument_name = option['instrument_name']
            c_or_p = instrument_name[-1]
            
            # get ATM vol
            maturity = [maturity for maturity in maturities if maturity in instrument_name][0]
            vol = settings.TRADE_SETTINGS[maturity]['VOL_MODELLING']['ATM']/100
            
            # sigma point strike
            pct = np.log(option['strike']) - np.log(index_price)
            seconds_to_expiry = option['expiration_timestamp']/1000 - time.time()
            year_rel = seconds_to_expiry / (365 * 24 * 3600)
            sp = pct / (vol * math.sqrt(year_rel))

            # only within x sigma points
            if abs(sp) > settings.MAX_SP_PRICING:
                continue
            elif c_or_p == 'C' and sp < settings.MIN_SP_PRICING:
                continue
            elif c_or_p == 'P' and sp > -settings.MIN_SP_PRICING:
                continue    
            else:
                options[instrument_name] = {}
                options[instrument_name]['contract_info'] = option
                
        self.options = options
        n_instruments = len(options)
        if n_instruments > settings.MAX_PRODUCTS_PRICING:
            LOGGER.error(f'pricing too many instruments, currently {n_instruments}')
            raise Exception('Pricing too many instruments, problem for rate limits. Reduce maximum sp for pricing for example or less maturities')
        elif n_instruments == 0:
            raise Exception('Pricing zero instrument. You might want to check if the specified maturities in config.json are still actively traded')
        LOGGER.info('Trading in total: ' + str(len(self.options)) + ' instruments')
            
    def select_options_for_quoting(self):
        """
        Select options to quote on
        Compute THEO value
        """
        
        # MODEL VOL
        self.vm.contracts_now = {key:item['contract_now'] for key,item in self.options.items()}
        sps, vols = self.vm.model_vol({}, settings)
        impl_vols = self.vm.get_implied_vols('m')
        impl_vols_bid = self.vm.get_implied_vols('b')
        impl_vols_ask = self.vm.get_implied_vols('a')
        
        if self.do_plot_vols:
            self.plot_vols(sps, impl_vols, impl_vols_bid, impl_vols_ask)
        
        for instrument_name in self.options.keys():
            
            # get basic info
            contract_info = self.options[instrument_name]['contract_info']
            contract_now = self.options[instrument_name]['contract_now']
            T = (contract_info['expiration_timestamp']/1000 - time.time()) / (365 * 24 * 3600)
            type_op = instrument_name[-1]
            strike = contract_info['strike']
            vol = vols[instrument_name]
            underlying = contract_now['underlying_price']
            
            self.options[instrument_name]['contract_pricing'] = BSOption(underlying, strike, T, INTEREST, vol, DIVIDEND, type_op)
            
    def plot_vols(self, sps, impl_vols, impl_vols_bid, impl_vols_ask):
        # get keys for quoting
        pref = list(settings.TRADE_SETTINGS.keys())[0]
        keys = [id for id in impl_vols.keys() if pref in id]
        sps_pref = [sps[i] for i in keys]
        
        # drawing code
        plt.clf()
        
        # plot implied vols mid market
        impl_vols_pref = [impl_vols[i] for i in keys]
        plt.scatter(sps_pref, impl_vols_pref, label = 'mid market')
        
        # plot implied vols bid/ask
        impl_vols_pref_bid = [impl_vols_bid[i] for i in keys]
        impl_vols_pref_ask = [impl_vols_ask[i] for i in keys]
        plt.scatter(sps_pref, impl_vols_pref_bid, label = 'bid')
        plt.scatter(sps_pref, impl_vols_pref_ask, label = 'ask')
                
        # plot own smile
        n = np.arange(min(sps_pref)-.5, max(sps_pref)+.5, 0.05)
        p = [self.vm.model[pref]['spline'](i) for i in n]
        plt.plot(n, p, label = 'smile')
        plt.legend()
        plt.grid()
        plt.pause(.001)
        
    def compute_portfolio_delta(self):
        """ Compute the delta of the total portfolio and for solely the options portfolio """
        self.PORTFOLIO.set_portfolio(self.DERIBIT.getpositions(settings.CURRENCY, kind = 'option'))
        self.PORTFOLIO.compute_net_delta(self.options, self.vm)
    
    def compute_option_quotes(self):
        maturities = settings.TRADE_SETTINGS
        maturities = [m for m in maturities.keys() if maturities[m]['ON'] == True]
        
        # for each contract that we price on
        # compute the possible bid and ask quote
        orders = []
        for option_name in self.options.keys():
            
            option_contract_info = self.options[option_name]['contract_info']
            option_contract_current = self.options[option_name]['contract_now']
            option_contract_pricing = self.options[option_name]['contract_pricing'].compute_all()
            
            underlying_price = option_contract_current['underlying_price']
            bid_ask_orders = self.compute_option_quote(option_contract_info, option_contract_current, option_contract_pricing)
            
            best_bid = option_contract_current['bids'][0][0] if len(option_contract_current['bids']) > 0 else 10
            best_ask = option_contract_current['asks'][0][0] if len(option_contract_current['asks']) > 0 else 0
            
            # check if we want to quote based on     
            for order in bid_ask_orders:
                
                # max delta hedge limits
                instrument_name = order['instrument_name']
                contract_info = self.options[instrument_name]['contract_info']
                strike_price = contract_info['strike']
                pref = '-'.join(instrument_name.split('-')[:2]) # BTC-14OCT14
                skew = settings.TRADE_SETTINGS[pref]['SKEW'].get(str(int(strike_price)), 0)
                
                direction = 1 if order['direction'] == 'buy' else -1
                size = order['amount']
                delta = option_contract_pricing['delta']
                theo = option_contract_pricing['price'] / underlying_price
                price = order['price']
                
                # IF SETTINGS DESIRE ONLY BUYING OR SELLING
                if (direction == 1 and settings.ONLY_SELL == True) or (direction == -1 and settings.ONLY_BUY == True):
                    continue
                if not (order['direction'] == 'buy' and order['price'] >= best_bid) and not (order['direction'] == 'sell' and order['price'] <= best_ask):
                    continue
                if not risk_validator(self.PORTFOLIO, option_contract_pricing, order, underlying_price):
                    continue    
                
                # verify margin in trade
                absolute_margin = direction * (theo - price)
                order['margin_abs'] = absolute_margin
                order['vol_margin'] = absolute_margin / option_contract_pricing['vega'] * underlying_price
                if skew != 0: # skewed orders are 'forced' placed by setting extremely high margin
                    order['vol_margin'] = 1e3

                # log orders when quoting best            
                o = {k:(round(order[k], 5) if type(order[k]) == float else order[k]) for k in order}
                    
                orders.append(order)
                sign = 4
                log_string = f"{o['instrument_name']} - {o['direction']} - quote: {round(o['price'], sign)} - market: {best_bid}/{best_ask}"
                log_string += f" - vol_m: {round(o['vol_margin'],sign)} - theo: {round(theo,sign)} - sp: {round(self.vm.sps[o['instrument_name']],sign)}"
                LOGGER.debug(log_string)

        
        # sort orders based on potential margin
        from operator import itemgetter
        orders = sorted(orders, key = itemgetter('vol_margin'), reverse = True)

        # store previous round of orders
        self.orders = orders[:settings.MAX_QUOTES_OPEN]
        
        for o in self.orders:
            option_name = o['instrument_name']
            
            option_contract_info = self.options[option_name]['contract_info']
            option_contract_current = self.options[option_name]['contract_now']
            option_contract_pricing = self.options[option_name]['contract_pricing'].compute_all()
            
            underlying_price = option_contract_current['underlying_price']
            best_bid = option_contract_current['bids'][0][0] if len(option_contract_current['bids']) > 0 else 10
            best_ask = option_contract_current['asks'][0][0] if len(option_contract_current['asks']) > 0 else 0
            
            theo = option_contract_pricing['price'] / underlying_price
            
            sign = 4
            log_string = f"{o['instrument_name']} - {o['direction']} - quote: {round(o['price'], sign)} - market: {best_bid}/{best_ask}"
            log_string += f" - vol_m: {round(o['vol_margin'],sign)} - theo: {round(theo,sign)} - sp: {self.vm.sps[o['instrument_name']]}"
            LOGGER.debug(log_string)
            
        return self.orders
    
    def compute_option_quote(self, contract: dict, now: dict, pricing: dict):
        pref = '-'.join(contract['instrument_name'].split('-')[:2]) # BTC-14OCT14
        conf = settings.TRADE_SETTINGS[pref]['SPREAD']
        current_price = now["underlying_price"]
        strike_price = contract["strike"]
        
        # compute black scholes pricing
        theo = pricing['price']
        theo_in_currency = theo/current_price
        
        # get skew from settings
        skew = settings.TRADE_SETTINGS[pref]['SKEW'].get(str(int(strike_price)))
        if not skew:
            skew = 0
        skew /= 100
        skew = max(skew,-1)
        skew = min(skew,1)
        
        # only trade options that are 'open' to trade (of course)
        if now["state"] != "open":
            LOGGER.error(f"product state of '{contract['instrument_name']}': {now['state']}")
            return () # return empty list of orders
        
        tick_size = contract['tick_size']
        
        # determine spread
        def compute_spread(option):
            spread = \
                min(TRANSACTION_FEE, theo_in_currency * MAX_PCT_FEE) +\
                abs(option['delta']) * PERPETUAL_FEE
            return max(spread, tick_size)
        
        # compute spread and quotes
        spread = compute_spread(pricing)

        # COMPUTE BID ORDER
        my_bid_price = theo_in_currency - (1-skew) * spread
        if skew == 0:
            my_bid_price = min(my_bid_price, now["bids"][0][0] + contract["tick_size"] if len(now["bids"]) > 0 else my_bid_price) # look into this with tick size
        my_bid_price = utils.round_price_to_instrument_tick(my_bid_price, contract["tick_size"], contract["tick_size_steps"])
        my_bid_size = contract["min_trade_amount"] * max(settings.MIN_TRADE_MULT,1)
        my_bid_size = utils.round_size_to_instrument_tick(my_bid_size, contract["min_trade_amount"])
        
        # COMPUTE ASK ORDER
        my_ask_price = theo_in_currency + (1+skew) * spread
        if skew == 0:
            my_ask_price = max(my_ask_price, now["asks"][0][0] - contract["tick_size"] if len(now["asks"]) > 0 else my_ask_price) # look into this with tick size
        my_ask_price = utils.round_price_to_instrument_tick(my_ask_price, contract["tick_size"], contract["tick_size_steps"])
        my_ask_size = contract["min_trade_amount"] * max(settings.MIN_TRADE_MULT,1)
        my_ask_size = utils.round_size_to_instrument_tick(my_ask_size, contract["min_trade_amount"])
    
        base_order = {
            'instrument_name': contract['instrument_name'],
            'order_type': 'limit',
            'post_only': 'false',
            'label': settings.ORDER_LABEL,
            'theo': round(theo/current_price,4),
        }
        
        order_bid = base_order.copy()
        order_bid.update({
            "price": my_bid_price,
            "amount": my_bid_size,
            'direction': 'buy',
        })
        order_ask = base_order.copy()
        order_ask.update({
            "price": my_ask_price,
            "amount": my_ask_size,
            'direction': 'sell',
        })

        # we don't want to buy options with a theo below zero
        # however, we do like to still sell them
        if my_bid_price < tick_size:
            return (order_ask,)
        else:
            return (order_bid, order_ask)
    
    def lower_tick(self, price, tick: str, tick_sizes: list = None):
        if tick_sizes:
            base_tick = tick
            if len(tick_sizes) > 0:
                i = 0
                while price > tick_sizes[i]["above_price"] and i < len(tick_sizes):
                    base_tick = tick_sizes[i]["tick_size"]
                    i += 1
            tick = base_tick
        return price - tick
    
    def hedge_positions(self):
        """ Calculate total delta position and hedge to delta zero with perpetual contract """
        # portfolio_delta, portfolio_delta_options, portfolio_gamma_options = self.compute_portfolio_delta()

        # hedge if new orders are filled
        private_option_trades_new = self.DERIBIT.getusertradesbycurrency(settings.CURRENCY, 'option', start_timestamp = (time.time() - 100)*1000)['trades']
        private_option_trades_hedge = [p for p in private_option_trades_new if p not in self.private_option_trades]
        if len(private_option_trades_hedge) > 0:
            
            # compute delta of solely the new trades
            LOGGER.info('new trades to hedge: ' + str(private_option_trades_hedge))
            new_trades_delta = 0
            for trade in private_option_trades_hedge:
                """
                [
                    {'liquidity': 'T', 'self_trade': False, 'risk_reducing': False, 'order_type': 'limit', 'trade_id': '283752496', 
                    'fee_currency': 'BTC', 'contracts': 0.1, 'underlying_price': 48724.33584422, 'reduce_only': False, 'post_only': False, 
                    'api': False, 'mmp': False, 'instrument_name': 'BTC-14FEB24-51000-C', 'tick_direction': 0, 'fee': 8.75e-06, 'matching_id': None, 
                    'order_id': '65516761994', 'trade_seq': 171, 'mark_price': 0.00046932, 'profit_loss': 0.0, 'amount': 0.1, 'index_price': 48712.24, 
                    'direction': 'buy', 'price': 0.0007, 'iv': 65.91, 'state': 'filled', 'timestamp': 1707837703838}
                ]
                """
                import portfolio
                portfolio.compute_delta(trade, self.options, self.vm)
                
                instrument_name = trade['instrument_name']
                if instrument_name not in self.options.keys():
                    contract_info = self.DERIBIT.getinstrument(instrument_name)
                    contract_now = self.DERIBIT.getorderbook(instrument_name)
                    current_price = contract_now['underlying_price']
                    strike = contract_info['strike']
                    interest_rate = 0
                    sigma = self.vm.vols[instrument_name]
                    time_till_expiration_seconds = int(contract_info["expiration_timestamp"]/1000 - time.time())
                    option = BSOption(current_price, strike, time_till_expiration_seconds / (365 * 24 * 3600), r = interest_rate, sigma = sigma, DIVIDEND = 0)
                    direction = 1 if trade['direction'] == 'buy' else -1
                    new_trades_delta += option.delta(instrument_name[-1]) * trade['amount'] * direction
                else:
                    new_trades_delta += portfolio.compute_delta(trade, self.options, self.vm)
   
            # get data of perpetual contract
            b = self.DERIBIT.getorderbook(PERPETUAL_CONTRACT)
            mid = utils.mid_market(b["bids"][0][0], b["asks"][0][0])
            
            # compute amount to hedge
            amount_to_hedge = abs(mid * new_trades_delta)
            LOGGER.info("amount to hedge b/c of new trades: " + str(amount_to_hedge) + " | mid price: " + str(mid))
            
            # compute rounded hedge price
            rounded_amount_to_hedge = \
                utils.round_size_to_instrument_tick(amount_to_hedge, self.hedge_instrument["min_trade_amount"])
            
            # hedge the new trades by buying/selling the underlying
            if rounded_amount_to_hedge > 0:
                if new_trades_delta > 0:
                    self.DERIBIT.sell(PERPETUAL_CONTRACT, "market", amount = rounded_amount_to_hedge)
                elif new_trades_delta < 0:
                    self.DERIBIT.buy(PERPETUAL_CONTRACT, "market", amount = rounded_amount_to_hedge)

            # refresh private trades list
            self.private_option_trades = private_option_trades_new
        
    def request_data_all(self):
        """
        For all traded option contracts, request the orderbook
        """
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(processes=10)    
        r = pool.map(self.DERIBIT.getorderbook, iterable=[option_name for option_name in self.options.keys()])
        for i, option_name in enumerate(self.options.keys()):
            self.options[option_name]['contract_now'] = r[i]
            self.substract_own_orders_from_book(option_name)
            
    def place_option_quotes(self):
        """ 
        Cancel and replace all orders
        """
        # remove this, replace orders instead
        for i, order in enumerate(self.orders):
            
            # check if existing order for same instrument and direction and label
            existing_order_id = utils.find_equivalent(order, self.open_orders)
            if existing_order_id:
                # replace order
                result = self.DERIBIT.edit(existing_order_id, order.get('amount'), order.get('price'), order.get('post_only', None))
                self.orders[i] = result['order']
            elif order['direction'] == 'sell':
                temp_order = {k:v for k,v in order.items() if k in VALID_ORDER_KEYS}
                result = self.DERIBIT.sell(**temp_order)
                self.orders[i] = result['order']
            elif order['direction'] == 'buy':
                temp_order = {k:v for k,v in order.items() if k in VALID_ORDER_KEYS}
                result = self.DERIBIT.buy(**temp_order)
                self.orders[i] = result['order']
            else:
                raise ValueError(f'Unrecognized direction: {order["direction"]}')
            
        # cancel old orders that were not edited
        cancel_ids = []
        for order in self.open_orders:
            if order['order_id'] not in [order2['order_id'] for order2 in self.orders]:
                cancel_ids.append(order['order_id'])
        
        # cancel these orders
        for order_id in cancel_ids:
            try:
                result = self.DERIBIT.cancel(order_id)
            except Exception as e:
                LOGGER.error(e)


    def substract_own_orders_from_book(self, option_name: str):
        """ 
        The outstanding orders are cancelled just before the orders are replaced.
        Before that, the public data for each option is requested.
        The returned public orderbook will contain our outstanding orders as well.
        Beneath the oustanding orders are substracted from the public orderbook,
        preventing 'front running' ourselves.
        """
        option_contract = self.options[option_name]['contract_now']
        bids, asks = option_contract["bids"], option_contract["asks"]
        
        for open_order in self.open_orders:
            if open_order["instrument_name"] == option_contract["instrument_name"]:
                if open_order["direction"] == "buy":
                    i = 0
                    while i < len(bids):
                        if bids[i][0] == open_order["price"]:
                            bids[i][1] = max(0, bids[i][1] - open_order["amount"])
                            if bids[i][1] == 0: del bids[i]
                            else: i += 1
                        else: i += 1
                elif open_order["direction"] == "sell":
                    i = 0
                    while i < len(asks):
                        if asks[i][0] == open_order["price"]:
                            asks[i][1] = max(0, asks[i][1] - open_order["amount"])
                            if asks[i][1] == 0: del asks[i]
                            else: i += 1
                        else: i += 1
                else: LOGGER.warning(f"direction of order was neither buy nor sell:\n{open_order}")
    
        self.options[option_name]['contract_now']['bids'] = bids
        self.options[option_name]['contract_now']['asks'] = asks

    def repeat(self,):
        self.active = True
        while self.active:

            started_at = time.time()
            self.reload_config()
            
            self.open_orders = self.DERIBIT.getopenordersbycurrency(settings.CURRENCY)
            self.request_data_all()
            self.hedge_positions()
            self.select_options_for_quoting()
            
            self.compute_portfolio_delta()
            LOGGER.info(f'pf delta: {self.PORTFOLIO.net_delta}')
                        
            # price options
            self.compute_option_quotes()
            if settings.DRY_RUN:
                for i,order in enumerate(self.orders,1):
                    LOGGER.info(f'order {i}:\n{str(order)}')
            else:
                self.place_option_quotes()

            # refresh loop
            run_time = time.time() - started_at
            time.sleep(max(0, MIN_LOOP_TIME - run_time))
            
    def stop(self):
        self.active = False
        LOGGER.info('stop: cancelling all orders')
        self.DERIBIT.cancelbylabel(settings.ORDER_LABEL)
        
    def __del__(self):
        try:
            self.stop()
            LOGGER.info('class destroyed succesfully')
        except:
            LOGGER.error('class destroy failed')

if __name__ == "__main__":
    while True:
        try:
            bot = MarketMakingBot()
            bot.repeat()
        except (KeyboardInterrupt, SystemExit) as e:
            LOGGER.error("keyboard interrupt, cancelling outstanding orders ...")
            bot = MarketMakingBot()
            bot.DERIBIT.cancelbylabel(settings.ORDER_LABEL)
            break
        except Exception as e:
            LOGGER.exception(str(e))
            bot.__init__()
            bot.DERIBIT.cancelbylabel(settings.ORDER_LABEL)
        time.sleep(RESTART_INTERVAL)