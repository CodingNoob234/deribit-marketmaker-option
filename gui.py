from configure_logger import configure_logger
configure_logger()
import logging
LOGGER = logging.getLogger(__name__)

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem, QPushButton
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSlot
import pandas as pd
from marketmaking_bot_gui import MarketMakingBotGUI

from threading import Thread
import matplotlib.pyplot as plt
import utils

CONFIG_PATH = './config.json'

class OptionsMarketMakerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Options Market Maker')
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget to hold the different windows
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # START STOP MARKET MAKING BOT
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_thread)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_thread)
        layout.addWidget(self.stop_button)
        
        # PLOT MID/BID/ASK/FIT VOLATILITY
        self.figure, self.ax = plt.subplots()
        self.canvas = self.figure.canvas
        layout.addWidget(self.canvas)
        
        # TABLE WITH VOLATILITY COEFFICIENTS
        table = QTableWidget()
        layout.addWidget(table)
        settings = utils.read_config(CONFIG_PATH)
        self.settings = settings
        maturity_settings = list(settings.TRADE_SETTINGS.items())[0]
        maturity_name = maturity_settings[0]
        maturity_configuration = maturity_settings[1]['VOL_MODELLING']
        # settings config to pandas dataframe
        data = {
            'values': [],
            maturity_name: []
        }
        data['values'].append('ATM')
        data[maturity_name].append(maturity_configuration['ATM'])
        for sp, mult in maturity_configuration['MULT'].items():
            data['values'].append(sp)
            data[maturity_name].append(mult)
        data = pd.DataFrame(data)
        self.data = data
        # insert pandas dataframe into table
        table.setRowCount(len(data))
        table.setColumnCount(len(data.columns))
        table.setHorizontalHeaderLabels(data.columns)
        for row_index, row_data in data.iterrows():
            for col_index, cell_value in enumerate(row_data):
                item = QTableWidgetItem(str(cell_value))
                table.setItem(row_index, col_index, item)
        table.itemChanged.connect(self.log_change)
        
        # TABLE WITH RECENT PRIVATE TRADES
        # table = QTableWidget()
        # layout.addWidget(table)
        
        # TABLE WITH RECENT PUBLIC TRADES (ONE MATURITY)
        # table = QTableWidget()
        # layout.addWidget(table)
        
        # OPPERTUNITY VISUALIZER
        # table = QTableWidget()
        # layout.addWidget(table)
        
        
    def start_thread(self):
        self.worker = MarketMakingBotGUI()
        self.worker_thread = Thread(target=self.worker.repeat)
        self.worker.variable.connect(self.update_variable_display)
        self.worker_thread.start()
        
    def stop_thread(self):
        self.worker.stop()
        self.worker_thread.join()
        
    @pyqtSlot(dict)
    def update_variable_display(self, data):
        self.ax.clear()
        self.ax.scatter(data['sps_pref'], data['impl_vols_pref'], label = 'mid')
        self.ax.scatter(data['sps_pref'], data['impl_vols_pref_bid'], label = 'bid')
        self.ax.scatter(data['sps_pref'], data['impl_vols_pref_ask'], label = 'ask')
        self.ax.plot(data['n'], data['p'], label = 'fit')
        self.ax.legend()
        self.canvas.draw()
        
    def log_change(self, item):
        settings = utils.read_config(CONFIG_PATH)
        # print(item.text(), item.column(), item.row())
        
        self.data.loc[item.row()] = [self.data.loc[item.row()][0], float(item.text())]
        maturity_settings = list(settings.TRADE_SETTINGS.items())[0]
        # maturity_name = maturity_settings[0]
        maturity_configuration = maturity_settings[1]['VOL_MODELLING']
        maturity_configuration["ATM"] = float(self.data.loc[0][1])
        maturity_configuration["MULT"] = {}
        for i, row in self.data.iloc[1:].iterrows():
            maturity_configuration["MULT"][row[0]] = float(row[1])
        utils.overwrite_settings(settings, CONFIG_PATH)

def main():
    app = QApplication(sys.argv)
    main_window = OptionsMarketMakerApp()
    main_window.show()
    sys.exit(app.exec())
    

if __name__ == '__main__':
    main()
