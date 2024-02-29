class Order(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_order_params(self):
        KEYS = ['k1', 'k2']
        
        res = {}
        for key in self.keys():
            if key in KEYS:
                res[key] = self.get(key)
        return res