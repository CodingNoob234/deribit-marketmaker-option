from scipy.stats import norm
import numpy as np
import math
import time

class BSOption:
    N = int(1e4)
    def __init__(self, S, K, T, r, sigma, q=0, c_or_p=None):
        """
        S: current underlying price
        K: strike price
        T: time to expiration (years)
        r: annual interest rate
        q: dividend (including short stock fee yield)
        """
        # self.volmanager = volmanager

        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.type_ = c_or_p
    
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @staticmethod
    def N_pdf(x):
        return norm.pdf(x)
    
    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r':self.r,
                'q':self.q,
                'sigma':self.sigma}
    
    def d1(self):
        return (np.log(self.S/self.K) + (self.r -self.q + self.sigma**2/2)*self.T) \
                                / (self.sigma*np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)
    
    def _call_value(self):
        return self.S*np.exp(-self.q*self.T)*self.N(self.d1()) - \
                    self.K*np.exp(-self.r*self.T) * self.N(self.d2())
                    
    def _put_value(self):
        return self.K*np.exp(-self.r*self.T) * self.N(-self.d2()) -\
                self.S*np.exp(-self.q*self.T)*self.N(-self.d1())
    
    def price(self, type_ = None, S: float = None, K: float = None):
        if S != None:
            self.S = S
        if K != None:
            self.K = K
            
        if type_ == None:
            type_ = self.type_
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value() 
        if type_ == 'B':
            return  {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')
        
    def delta(self, type_: str = None, volmanager = None):
        if type_ == None:
            type_ = self.type_
        
        if type_ == 'C':
            delta = math.exp(-self.q*self.T) * self.N(self.d1())
        elif type_ == 'P':
            delta = math.exp(-self.q*self.T) * self.N(self.d1()) - 1
        else:
            raise ValueError('Unrecognized type')
        
        # adjust skew delta
        if volmanager:
            delta += self.vega() * self.volmanager.predict_2nd_order(self.K)
        return delta
    
    def gamma(self):
        return math.exp(-self.q*self.T) * self.N_pdf(self.d1()) / (self.sigma * self.S * math.sqrt(self.T))
        
    def vega(self):
        return math.exp(-self.q*self.T) * self.N_pdf(self.d1()) * (self.S * math.sqrt(self.T))/100
    
    def vedi(self, dtm: int):
        return self.vega() * math.sqrt(math.sqrt(720) / (self.T*365))
        
    def rho(self, type_:str = None):
        if type_ == None:
            type_ = self.type_
        if type_ == 'C':
            return self.K * self.T * math.exp(-self.r * self.T) * self.N_pdf(self.d2()) / 100
        elif type_ == 'P':
            return -self.K * self.T * math.exp(-self.r * self.T) * self.N_pdf(-self.d2()) / 100
        else:
            raise ValueError('Unrecognized type')
    
    def theta(self, type_:str = None):
        if type_ == None:
            type_ = self.type_
            
        if type_ == 'C':
            deri = math.exp(-.5 * (self.d1())**2) / math.sqrt(2 * math.pi)
            theta =\
                - (self.S * (deri) * self.sigma / (2*math.sqrt(self.T))) - self.r * self.K * math.exp(-self.r*self.T) * self.N_pdf(self.d2())
            return theta / 365
        elif type_ == 'P':
            deri = math.exp(-.5 * (self.d1())**2) / math.sqrt(2 * math.pi)
            theta =\
                - (self.S * (deri) * self.sigma / (2*math.sqrt(self.T))) + self.r * self.K * math.exp(-self.r*self.T) * self.N_pdf(self.d2())
            return theta / 365
        else:
            raise ValueError('Unrecognized type')
        
    def compute_all(self, type_: str = None, S: float = None, K: float = None):
        if type_ == None:
            type_ = self.type_
            
        if S != None:
            self.S = S
            
        if type_ == 'C' or type_ == 'P':
            return {
                'price': self.price(type_, self.S),
                'delta': self.delta(type_),
                'gamma': self.gamma(),
                'rho': self.rho(type_),
                'vega': self.vega(),
            }
        elif type_ == 'B':
            return (
                {
                    'price': self.price('C', self.S),
                    'delta': self.delta('C'),
                    'gamma': self.gamma(),
                    'rho': self.rho('C'),
                    'vega': self.vega(),
                },
                {
                    'price': self.price('P', self.S),
                    'delta': self.delta('P'),
                    'gamma': self.gamma(),
                    'rho': self.rho('P'),
                    'vega': self.vega(),
                },
            )
            
      
def simulate_price(dist, T, S):
    sample = np.random.choice(dist, math.floor(T), replace=True)
    if T % 1 > 0:
        added_sample = np.random.choice(dist, 1) * math.sqrt(T % 1)
        sample = np.append(sample, added_sample)
    return S * math.exp(sample.sum())

def evaluate_option(settlement_price, K, c_or_p = "C"):
    diff = settlement_price - K
    return max(diff,0) if c_or_p == 'C' else max(-diff,0)
    
class MonteCarloPricing:
    N = int(1e4)
    FEES = .0003
    FEES_HEDGE = .0005*2
    def __init__(self, S, K, T, r, settlement_prices, q=0, c_or_p=None, hedge=False):
        """
        S: current underlying price
        K: strike price
        T: time to expiration (years)
        r: annual interest rate
        q: dividend (including short stock fee yield)
        """
        # self.volmanager = volmanager

        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.type_ = c_or_p
        self.HEDGE = True
        self.settlement_prices = settlement_prices
    
    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r':self.r,
                'q':self.q,
                'sigma':self.sigma}
    
    def price(self, type_ = None, S: float = None, K: float = None):
        if S != None:
            self.S = S
        if K != None:
            self.K = K
            
        if type_ == None:
            type_ = self.type_
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value() 
        if type_ == 'B':
            return  {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')
        
    def mc_pricing(self, delta):
        n_hours = self.T*24*365
        settlement_prices = [simulate_price(self.settlement_prices, n_hours, self.S) for i in range(self.N)]            
        pnl_option = [evaluate_option(settlement_price, self.K, self.type_) for settlement_price in settlement_prices]
        pnl_option = np.array(pnl_option)
        if self.HEDGE:
            pnl_hedge = [-delta * (settlement_price - self.S) for settlement_price in settlement_prices]
            pnl_hedge = np.array(pnl_hedge)
            pnl_total = pnl_option + pnl_hedge
        else:
            pnl_total = pnl_option
        return np.average(pnl_total)