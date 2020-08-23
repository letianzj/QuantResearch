#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import importlib
from datetime import datetime
import yaml
from PyQt5 import QtCore, QtWidgets, QtGui
from quanttrading2.gui.ui_main_window import MainWindow
import atexit
from signal import signal, SIGINT, SIG_DFL
from os import kill
from multiprocessing import Process
import logging

# https://stackoverflow.com/questions/4938723/what-is-the-correct-way-to-make-my-pyqt-application-quit-when-killed-from-the-co
signal(SIGINT, SIG_DFL)


def main(config_file):
    config = None
    today =  datetime.today().strftime('%Y%m%d')
    strategy_dict = {}
    try:
        # path = os.path.abspath(os.path.dirname(__file__))
        # config_file = os.path.join(path, 'config.yaml')
        with open(config_file, encoding='utf8') as fd:
            config = yaml.safe_load(fd)
    except IOError:
        print("config.yaml is missing")

    required_dirs = ['./log/', './tick/', './strategy/']
    for d in required_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    _logger = logging.getLogger('quanttrading2')
    _logger.setLevel(logging.DEBUG)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(f"./log/{today}.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    _logger.addHandler(handler1)
    _logger.addHandler(handler2)

    _logger2 = logging.getLogger('qtlive')
    _logger2.setLevel(logging.DEBUG)
    _logger2.addHandler(handler1)
    _logger2.addHandler(handler2)

    _logger3 = logging.getLogger('tick_recorder')
    _logger3.setLevel(logging.INFO)
    handler3 = logging.FileHandler(f"./tick/{today}.txt")
    formatter = logging.Formatter('')
    handler3.setFormatter(formatter)
    _logger3.addHandler(handler3)

    strategy_dict = {}
    for _, _, files in os.walk('./strategy'):
        for name in files:
            if 'strategy' in name and '.pyc' not in name:
                s = name.replace('.py', '')
                try:
                    moduleName = f'strategy.{s}'
                    # import module
                    module = importlib.import_module(moduleName)
                    for k in dir(module):
                        if ('Strategy' in k) and ('Abstract' not in k) and (k in config['strategy']):
                            v = module.__getattribute__(k)
                            _strategy = v()
                            strategy_dict[k] = _strategy
                except Exception as e:
                    _logger2.error(f'Unable to load strategy {s}: {str(e)}')

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon("gui/image/logo.ico"))
    mainWindow = MainWindow(config, strategy_dict)

    if config['theme'] == 'dark':
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    mainWindow.show()      # .showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live Engine')
    parser.add_argument('-f', '--config_file', dest = 'config_file', default='./config_live.yaml', help='config yaml file')
    args = parser.parse_args()

    main(args.config_file)