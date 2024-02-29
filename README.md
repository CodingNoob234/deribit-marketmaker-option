# deribit-marketmaker-option

This market making bot places buy and sell orders for options on Deribit. If a buy or sell order is filled, the current delta will be hedge thourgh the underlying perpetual contract. Ultimately, market making in options consists if pricing volatility correctly, maintaining wide enough spreads to remain profitable and create a hedged portfolio.

# Running the Bot
To run the application, one needs to copy or rename the settings_template.py to settings.py and fill in the credentials.
Install all requirements by ```pip install -r requirements.txt``` and run the application through ```python marketmaking_bot.py```. The configurations for the bot are specified in config.json. Here, the volatility is modelled by defining the ATM volatility, with multipliers for options further in- or out-of-the-money. The config file also contains position limits and other parameters to manage risk. Please use at your own risk and try your strategy in the test environment. You can also perform a dry-run where no orders are actually placed, by setting dry-run: True in the config file.