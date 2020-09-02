#!/usr/bin/env python
# -*- coding: utf-8 -*-
from quanttrading2.strategy.strategy_base import StrategyBase
from quanttrading2.order.order_event import OrderEvent
from quanttrading2.order.order_type import OrderType
from quanttrading2.data.tick_event import TickType
import logging

_logger = logging.getLogger('qtlive')


class OrderPerIntervalStrategy(StrategyBase):
    """
    buy on the first tick then hold to the end
    """
    def __init__(self):
        super(OrderPerIntervalStrategy, self).__init__()
        self.ticks = 0
        self.tick_trigger_threshold = 2000
        self.direction = 1
        _logger.info('OrderPerIntervalStrategy initiated')

    def on_tick(self, k):
        super().on_tick(k)     # extra mtm calc

        if k.tick_type != TickType.TRADE:
            print(k, f'{self.ticks}/{self.tick_trigger_threshold}')
        if (k.full_symbol == self.symbols[0]) & (self.ticks > self.tick_trigger_threshold):
            o = OrderEvent()
            o.full_symbol = k.full_symbol
            o.order_type = OrderType.MARKET
            o.order_size = self.direction
            self.direction = 1 if self.direction == -1 else -1
            _logger.info(f'OrderPerIntervalStrategy order placed on ticks {self.ticks}, {k.price}')
            self.place_order(o)
            self.ticks = 0
        else:
            self.ticks += 1