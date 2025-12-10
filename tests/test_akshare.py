# -*- coding: utf-8 -*-
import akshare as ak

stock_news_em_df = ak.stock_news_em(symbol="603777")
print(stock_news_em_df)
print("="*50)

stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
print(stock_zh_a_spot_em_df)

