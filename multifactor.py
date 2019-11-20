#!/usr/bin/python
#coding:utf-8

"""
@author: yirenliu
@contact: 12573439@qq.com
@time: 2019/08/05 14:13
"""

from sympy import *
import math
from atrader import *
import numpy as np
import pandas as pd
from scipy.stats import mstats
import datetime as dt
import statsmodels.api as sm

def init(context):
    set_backtest(initial_cash = initial_cash,stock_cost_fee=1)  # 设置回测初始信息
    reg_factor(factor_list)    # 注册因子数据
    context.len = len(context.target_list)  # 股票标的个数

    target_all = pd.Series(np.char.array(context.target_list).lower())
    context.code_list = pd.DataFrame(target_all, columns= ['code'])
    context.code_list.reset_index(inplace=True)
    context.code_list.rename(columns = {'index':'target_idx'},inplace = True)
    context.code_list.set_index('target_idx',inplace=True)
    # 存入所有可交易日期
    context.fre_date = get_trade_date_list(begin_date,end_date,_period=freq[0],begin_or_end= freq[1])
    context.cons = {}
    # 选取沪深300每个月初的成分股
    for date in context.fre_date:
        context.cons[date] = get_code_list(universe, date)['code'].tolist()
    context.exposure = rm.exposure(context.target_list, begin_date, end_date)
    a=1
    context.exposure['trade_date'] = context.exposure['trade_date'].apply(lambda x:str(x)[0:10])

def on_data(context):
    current_date = context.now.strftime('%Y-%m-%d')
    print(current_date)
    if current_date in context.fre_date:  # 调仓频率
        current_code = context.cons[current_date]
        factor = get_reg_factor(reg_idx=context.reg_factor[0],
                                 target_indices=[], length=1, df=True, sort_by='date')   # 获取因子值
        if len(factor) == 0:
            return
        factor_df = pd.DataFrame(index = current_code)
        for i in range(len(factor_list)):
            factor_data = factor.groupby('factor').get_group(factor_list[i])
            if direction[i] == -1:
                factor_data['value'] = 1/factor_data['value']
            factor_data.set_index('target_idx',inplace=True)
            factor_data = pd.merge(context.code_list,factor_data,left_index=True,right_index=True,how='inner')
            factor_data = factor_data.set_index('code').reindex(current_code)
            factor_data = factor_data['value']
            if i==0:
                factor_data = factor_data[factor_data > 0]
            if factor_processing[i]['winsorize'][0] == True:
                factor_data = mad_winsorize_series(factor_data,factor_processing[i]['winsorize'][1])   # 极值处理
            if factor_processing[i]['neutralize'][0] == True:
                factor_data = neutralize_series(factor_data,context.exposure, TradeDate= current_date,
                                                include_industry = factor_processing[i]['neutralize'][1],
                                                include_style_list = factor_processing[i]['neutralize'][2])   # 中性化处理
            if factor_processing[i]['standardize'] == True:
                factor_data = standardize_series(factor_data)     # 标准化处理
            factor_data.rename(columns={0:factor_list[i]},inplace=True)
            factor_df = factor_df.merge(pd.DataFrame(factor_data,columns=[factor_list[i]]),left_index=True,right_index=True,how='right')
            factor_df = factor_df.dropna(axis=0, how='any')

        # 因子对称正交化
        current_code = factor_df.index.tolist()
        N = len(factor_df)
        I = np.identity(len(factor_list))
        I = np.mat(I)
        factor_df = np.mat(factor_df.values)
        M = (N - 1) * np.cov(factor_df.T)
        M = np.mat(M)
        D, U = np.linalg.eig(M)
        D = np.diag(D)
        D = np.mat(D)

        v, Q = np.linalg.eig(D)
        V = np.diag(v ** (-0.5))
        D = Q * V * Q ** -1
        S = U * D * (U.T) * I
        factor_df = factor_df * S
        factor_df = pd.DataFrame(factor_df, index=current_code)
        w = np.array(Weight)
        # 计算加权后因子总分
        totalscore = pd.DataFrame(factor_df.dot(w))
        totalscore.reset_index(inplace=True)
        totalscore.rename(columns = {totalscore.columns[0]:'code',0:'Value'},inplace= True)
        totalscore = totalscore.sort_values('Value', ascending=False)
        totalscore.set_index('code',inplace=True)

        # 获取股票列表,并等权重分配资金
        stock_number = int(len(totalscore)/layer)
        targetlist = (totalscore.iloc[:stock_number])
        targetlist.dropna(inplace=True)
        targetlist['target_percent'] = 1/len(targetlist)

        mp = context.account().positions  # 获取当前仓位
        mp['code'] = mp['code'].apply(lambda x:str(x).lower())
        mp_stock = mp.iloc[:-1,:]     #  获取股票仓位
        if np.sum(mp_stock['volume_long']*mp_stock['price']) > 0:
            mp_stock['percent'] = mp_stock['volume_long']*mp_stock['price'] / np.sum(mp_stock['volume_long']*mp_stock['price'])
        elif np.sum(mp_stock['volume_long']*mp_stock['price']) == 0:
            mp_stock['percent'] = 0
        mp_stock = mp_stock.merge(targetlist,on='code',how='left')
        mp_stock['target_percent'].fillna(0,inplace=True)
        mp_stock[mp_stock['target_percent']<0]['target_percent'] == 0
        handle_stock_orders(context,mp_stock)      #股票委托操作
####################################################################################################################
# 订单委托
def handle_stock_orders(context,mp_stock):   #股票委托操作
    for i in range(len(mp_stock)):
        if mp_stock.loc[i, 'target_percent'] >= 0:
            order_target_percent(account_idx = 0, target_idx = i, target_percent = mp_stock.loc[i,'target_percent'],side = 1,order_type = 2, price = 0.0)
        print(mp_stock.loc[i,'code'], '以市价调整到', mp_stock.loc[i,'target_percent'],'仓位')

####################################################################################################################
# 其他函数
def winsorize_series(se,limits=0.025, inclusive= False):
    # 去极值
    limits = [limits,limits]
    inclusive = [inclusive,inclusive]
    data = mstats.winsorize(se, limits, inclusive)
    return pd.Series(index=se.index, data=data.ravel())

def mad_winsorize_series(se, sigma_n):
    # 去极值（绝对中位数差法）
    dm = se.median()
    dm1 = (se - dm).abs().median()
    upper = dm + sigma_n * dm1
    lower = dm - sigma_n * dm1

    se[se > upper] = upper
    se[se < lower] = lower
    return se

def standardize_series(se):
    # 标准化()
    se_std = se.std()
    se_mean = se.mean()
    return (se - se_mean) / se_std

def neutralize_series(se, total_Exposure,TradeDate, include_industry = True, include_style_list = []):
    # 中性化
    Exposure_group = total_Exposure.groupby('trade_date')
    Exposure = Exposure_group.get_group(TradeDate)
    Exposure.set_index('code',inplace=True)
    del Exposure['trade_date']
    if include_industry == True:
        Exposure_industry = Exposure[Exposure.columns[11:-1]]
    if include_style_list != []:
        Exposure_style = Exposure[include_style_list]
        New_Exposure = pd.merge(Exposure_industry,Exposure_style,on='code',how='inner').reset_index()
        New_Exposure.drop_duplicates('code',inplace=True)
    else:
        New_Exposure = Exposure_industry.reset_index()
    se = se.reset_index()
    se =  se.rename(columns= {se.columns[0]:'code'})
    neuthalizedata = pd.merge(se, New_Exposure, on='code', how='left')
    neuthalizedata.replace(np.nan,0,inplace =True)
    neuthalizedata.drop_duplicates('code',inplace=True)
    results = sm.OLS(neuthalizedata.iloc[:, 1], sm.add_constant(neuthalizedata.iloc[:, 2:])).fit()
    neuthalize_se = pd.DataFrame(np.array(results.resid),index = se['code'])
    return neuthalize_se

# 取交易日相关数据
def get_trade_date_list(begin_date,end_date,_period='monthly',begin_or_end= 'begin'):
    trade_date_list = get_trading_days('sse',begin_date,end_date)
    time_series = pd.Series(trade_date_list)
    week = time_series.apply(lambda x:x.week)
    month = time_series.apply(lambda x:x.month)
    quarter = time_series.apply(lambda x:x.quarter)
    year = time_series.apply(lambda x:x.year)
    if _period =='daily':
        trade_date_list = time_series.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    if _period == 'weekly' and begin_or_end == 'begin':
        trade_date_list = time_series[week!=week.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'weekly' and begin_or_end == 'end':
        trade_date_list = time_series[week!=week.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'monthly' and begin_or_end == 'begin':
        trade_date_list = time_series[month != month.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'monthly' and begin_or_end == 'end':
        trade_date_list = time_series[month != month.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'quarterly' and begin_or_end == 'begin':
        trade_date_list = time_series[quarter != quarter.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'quarterly' and begin_or_end == 'end':
        trade_date_list = time_series[quarter != quarter.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'yearly' and begin_or_end == 'begin':
        trade_date_list = time_series[year != year.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'yearly' and begin_or_end == 'end':
        trade_date_list = time_series[year != year.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    return trade_date_list

####################################################################################################################
# 整体回测参数
global universe,begin_date,end_date,initial_cash,benchmark
universe= 'HS300'    # 选择投资域
begin_date = '2018-04-16'    # 回测开始时间
end_date = '2019-07-30'      # 回测结束时间
initial_cash = 100000000      # 回测金额
benchmark = 'HS300'       # 选择基准

# 多因子参数设置
global freq,factor_list,factor_processing,Weight,direction,layer
freq = ['monthly','end']    # 股票刷新频率（daily为日，weekly为周，monthly为月，quarterly为季，yearly为年，begin为第一个交易日，end为最后一个交易日）
factor_list = ['PE','PS']    # 选择因子
factor_processing = {0:{'winsorize':[True,1], 'neutralize':[True,True,[]],'standardize':True},
                     1:{'winsorize':[True,1], 'neutralize':[True,True,[]],'standardize':True},}# 各个因子处理
Weight = [0.5,0.5]     # 因子权重
direction = [-1,-1]    # 因子方向，-1为负向，1为正向
layer = 10             # 分成几层

if __name__ == '__main__':
    target = get_code_list_set(universe,begin_date,end_date)['code'].tolist()
    strategy_name = 'PE+VOL20:'+str(factor_list) + '+' + universe
    run_backtest(strategy_name=strategy_name,file_path='.',target_list=target,frequency='day',fre_num=1,begin_date=begin_date,end_date=end_date,fq=1)

    universe_results['pe'] = neutralize_series(universe_results, bb, 'pe')
    universe_results['roe'] = neutralize_series(universe_results, bb, 'roe')
    universe_results['market'] = neutralize_series(universe_results, bb, 'market')
    universe_results['pe'] = standardize_series(universe_results['pe'])
    universe_results['roe'] = standardize_series(universe_results['roe'])
    universe_results['market'] = standardize_series(universe_results['market'])
    universe_results['socre'] = universe_results['pe'] + universe_results['roe'] + universe_results['market']
