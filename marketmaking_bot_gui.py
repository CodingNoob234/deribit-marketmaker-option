import logging
LOGGER = logging.getLogger(__name__)

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, pyqtSlot
import time

CONFIG_PATH = './config.json'
import utils
settings = utils.read_config(CONFIG_PATH)

from marketmaking_bot import MarketMakingBot
import numpy as np

MIN_LOOP_TIME = 10

class MarketMakingBotGUI(MarketMakingBot, QObject):
    
    variable = pyqtSignal(dict)
    
    recent_trades = pyqtSignal(list)
    
    def __init__(self):
        MarketMakingBot.__init__(self)
        QObject.__init__(self)
        self.do_plot_vols = False
        
    def get_gui_vol_data(self):
        sps, vols = self.vm.model_vol({}, settings)
        impl_vols = self.vm.get_implied_vols('m')
        impl_vols_bid = self.vm.get_implied_vols('b')
        impl_vols_ask = self.vm.get_implied_vols('a')
        
        # get keys for quoting
        pref = list(settings.TRADE_SETTINGS.keys())[0]
        keys = [id for id in impl_vols.keys() if pref in id]
        sps_pref = [sps[i] for i in keys]
        
        # plot implied vols mid market / bid / ask
        impl_vols_pref = [impl_vols[i] for i in keys]
        impl_vols_pref_bid = [impl_vols_bid[i] for i in keys]
        impl_vols_pref_ask = [impl_vols_ask[i] for i in keys]
        
        # plot own smile
        n = np.arange(min(sps_pref)-.5, max(sps_pref)+.5, 0.05)
        p = [self.vm.model[pref]['spline'](i) for i in n]
        
        data = {
            'sps_pref': sps_pref,
            'impl_vols_pref': impl_vols_pref,
            'impl_vols_pref_bid': impl_vols_pref_bid,
            'impl_vols_pref_ask': impl_vols_pref_ask,
            'n': n,
            'p': p
        }
        return data

    def repeat(self,):
        self.active = True
        while self.active:
            
            self.DERIBIT.get_historical_volatility('BTC')

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
                
            self.variable.emit(self.get_gui_vol_data())
            
            self.recent_trades.emit(self.private_option_trades)

            # refresh loop
            run_time = time.time() - started_at
            time.sleep(max(0, MIN_LOOP_TIME - run_time))
            
if __name__ == '__main__':
    m = MarketMakingBotGUI()
    m.repeat()