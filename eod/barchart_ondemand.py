#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests

class OnDemandError(RuntimeError):
    pass

class OnDemandClient(object):
    """
    from utils.barchart_ondemand import OnDemandClient
    od = OnDemandClient()
    resp = od.history('CLZ19', 'daily', maxRecords=500, startDate=20190101)
    df = pd.DataFrame(resp['results'])
    df.set_index('tradingDay', inplace=True)
    """
    def __init__(self, api_key=None, end_point='https://marketdata.websol.barchart.com/'):
        self.endpoint = end_point
        self.api_key = api_key
        self.debug = False

    def _do_call(self, url, params):
        if not isinstance(params, dict):
            params = dict()

        if self.api_key:
            params['apikey'] = self.api_key

        headers = dict()
        headers['X-OnDemand-Client'] = 'bc_python'

        if self.debug:
            print('do call with params: %s, url: %s' % (params, url))

        resp = requests.get(url, params=params, timeout=60, headers=headers)

        if self.debug:
            print('resp code: %s, resp text: %s' % (resp.status_code, resp.text))

        if resp.status_code != 200:
            raise OnDemandError('Request Failed: %s. Text: %s' % (resp.status_code, resp.text))

        try:
            result = resp.json()
        except Exception as e:
            raise OnDemandError(
                'Failed to parse JSON response %s. Resp Code: %s. Text: %s' % (e, resp.status_code, resp.text))
        finally:
            resp.connection.close()

        return result

    def quote(self, symbols, fields=''):
        params = dict(symbols=symbols, fields=fields)
        return self._do_call(self.endpoint + 'getQuote.json', params)

    def quote_eod(self, symbols, exchange):
        params = dict(symbols=symbols, exchange=exchange)
        return self._do_call(self.endpoint + 'getQuoteEod.json', params)

    def profile(self, symbols, fields=''):
        params = dict(symbols=symbols, fields=fields)
        return self._do_call(self.endpoint + 'getProfile.json', params)

    def equities_by_exchange(self, exchange, fields=''):
        params = dict(exchange=exchange, fields=fields)
        return self._do_call(self.endpoint + 'getEquitiesByExchange.json', params)

    def futures_by_exchange(self, exchange, **kwargs):
        params = dict(exchange=exchange)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getFuturesByExchange.json', params)

    def futures_options(self, root, **kwargs):
        params = dict(root=root)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getFuturesOptions.json', params)

    def special_options(self, root, **kwargs):
        params = dict(root=root)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getSpecialOptions.json', params)

    def equity_options(self, underlying_symbols, **kwargs):
        params = dict(underlying_symbols=underlying_symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getEquityOptions.json', params)

    def equity_options_intraday(self, underlying_symbols, **kwargs):
        params = dict(underlying_symbols=underlying_symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getEquityOptionsIntraday.json', params)

    def equity_options_history(self, symbol, **kwargs):
        params = dict(symbol=symbol)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getEquityOptionsHistory.json', kwargs)

    def forex_forward_curves(self, symbols, **kwargs):
        params = dict(symbols=symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getForexForwardCurves.json', kwargs)

    def history(self, symbol, historical_type, **kwargs):
        params = dict(symbol=symbol, type=historical_type)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getHistory.json', kwargs)

    def financial_highlights(self, symbols, fields=''):
        params = dict(fields=fields, symbols=symbols)
        return self._do_call(self.endpoint + 'getFinancialHighlights.json', params)

    def financial_ratios(self, symbols, fields=''):
        params = dict(fields=fields, symbols=symbols)
        return self._do_call(self.endpoint + 'getFinancialRatios.json', params)

    def cash_flow(self, symbols, **kwargs):
        params = dict(symbols=symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getCashFlow.json', kwargs)

    def ratings(self, symbols, fields=''):
        params = dict(fields=fields, symbols=symbols)
        return self._do_call(self.endpoint + 'getRatings.json', params)

    def index_members(self, symbol, fields=''):
        params = dict(fields=fields, symbol=symbol)
        return self._do_call(self.endpoint + 'getIndexMembers.json', params)

    def income_statements(self, symbols, frequency, **kwargs):
        params = dict(symbols=symbols, frequency=frequency)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getIncomeStatements.json', kwargs)

    def competitors(self, symbol, **kwargs):
        params = dict(symbol=symbol)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getCompetitors.json', kwargs)

    def insiders(self, symbol, insider_type, **kwargs):
        params = dict(symbol=symbol, type=insider_type)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getInsiders.json', kwargs)

    def balance_sheets(self, symbols, frequency, **kwargs):
        params = dict(symbols=symbols, frequency=frequency)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getBalanceSheets.json', kwargs)

    def corporate_actions(self, symbols, **kwargs):
        params = dict(symbols=symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getCorporateActions.json', params)

    def earnings_estimates(self, symbols, **kwargs):
        params = dict(symbols=symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getEarningsEstimates.json', params)

    def chart(self, symbols, **kwargs):
        params = dict(symbols=symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getChart.json', kwargs)

    def technicals(self, symbols, **kwargs):
        params = dict(symbols=symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getTechnicals.json', kwargs)

    def leaders(self, asset_type, **kwargs):
        params = dict(assetType=asset_type)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getLeaders.json', kwargs)

    def highs_lows(self, asset_type, **kwargs):
        params = dict(assetType=asset_type)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getHighsLows.json', kwargs)

    def sectors(self, sector_period, **kwargs):
        params = dict(sectorPeriod=sector_period)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getSectors.json', kwargs)

    def news(self, sources, **kwargs):
        params = dict(sources=sources)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getNews.json', kwargs)

    def news_sources(self, **kwargs):
        return self._do_call(self.endpoint + 'getNewsSources.json', kwargs)

    def news_categories(self, **kwargs):
        return self._do_call(self.endpoint + 'getNewsCategories.json', kwargs)

    def sec_filings(self, symbols, filing_type, **kwargs):
        params = dict(symbols=symbols, filingType=filing_type)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getSECFilings.json', kwargs)

    def weather(self, **kwargs):
        return self._do_call(self.endpoint + 'getWeather.json', kwargs)

    def usda_grain_prices(self, **kwargs):
        return self._do_call(self.endpoint + 'getUSDAGrainPrices.json', kwargs)

    def etf_details(self, symbols, **kwargs):
        kwargs.update(dict(symbols=symbols))
        return self._do_call(self.endpoint + 'getETFDetails.json', kwargs)

    def etf_constituents(self, symbol, **kwargs):
        kwargs.update(dict(symbol=symbol))
        return self._do_call(self.endpoint + 'getETFConstituents.json', kwargs)

    def crypto(self, symbols, **kwargs):
        params = dict(symbols=symbols)
        kwargs.update(params)
        return self._do_call(self.endpoint + 'getCrypto.json', kwargs)

    def get(self, api_name, **kwargs):
        return self._do_call(self.endpoint + api_name + '.json', kwargs)


