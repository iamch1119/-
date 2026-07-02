import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 读取所有原始数据 ==========
user_balance = pd.read_csv('user_balance_table.csv')
share_interest = pd.read_csv('mfd_day_share_interest.csv')
bank_shibor = pd.read_csv('mfd_bank_shibor.csv')

# ========== 2. 聚合每日全平台交易总量 ==========
daily_trade = user_balance.groupby('report_date').agg(
    total_purchase=('total_purchase_amt', 'sum'),
    total_redeem=('total_redeem_amt', 'sum')
).reset_index()

# ========== 3. 统一日期格式，合并利率数据 ==========
daily_trade['report_date'] = pd.to_datetime(daily_trade['report_date'], format='%Y%m%d')
share_interest['mfd_date'] = pd.to_datetime(share_interest['mfd_date'], format='%Y%m%d')
bank_shibor['mfd_date'] = pd.to_datetime(bank_shibor['mfd_date'], format='%Y%m%d')

# 合并收益率 + Shibor利率，前向填充缺失值
daily_data = daily_trade.merge(
    share_interest, left_on='report_date', right_on='mfd_date', how='left'
).drop('mfd_date', axis=1)
daily_data = daily_data.merge(
    bank_shibor, left_on='report_date', right_on='mfd_date', how='left'
).drop('mfd_date', axis=1)

rate_cols = ['mfd_daily_yield', 'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_M']
for col in rate_cols:
    daily_data[col] = daily_data[col].ffill().bfill()

# ========== 4. 构造日历时间特征 ==========
def add_time_features(df):
    df = df.copy()
    df['weekday'] = df['report_date'].dt.weekday
    df['day_of_month'] = df['report_date'].dt.day
    df['month'] = df['report_date'].dt.month
    df['week_of_year'] = df['report_date'].dt.isocalendar().week.astype(int)
    
    # 月初月末精细化标记
    df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
    df['is_mid_month'] = ((df['day_of_month'] >= 14) & (df['day_of_month'] <= 16)).astype(int)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 2014年中秋节 9.8 标记节假日及前后效应
    df['holiday_type'] = 0
    mid_autumn = datetime(2014,9,8)
    df.loc[df['report_date'] == mid_autumn, 'holiday_type'] = 2
    df.loc[df['report_date'] == mid_autumn - timedelta(days=1), 'holiday_type'] = 1
    df.loc[df['report_date'] == mid_autumn + timedelta(days=1), 'holiday_type'] = 3
    return df

daily_data = add_time_features(daily_data)

# ========== 5. 特征工程函数 ==========
def add_history_features(df):
    df = df.copy().sort_values('report_date').reset_index(drop=True)
    target_cols = ['total_purchase', 'total_redeem']
    
    for col in target_cols:
        # 多阶滞后特征
        for lag in [1,2,3,7,14,21]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # 多窗口滑窗统计
        for window in [3,7,14,21]:
            df[f'{col}_mean{window}'] = df[col].shift(1).rolling(window).mean()
            df[f'{col}_std{window}'] = df[col].shift(1).rolling(window).std()
            df[f'{col}_max{window}'] = df[col].shift(1).rolling(window).max()
        
        # 同星期历史均值（核心周期特征）
        df[f'{col}_weekday_mean'] = df.groupby('weekday')[col].transform(lambda x: x.shift(1).expanding().mean())
        
        # 环比增长率
        df[f'{col}_rate_lag1'] = (df[col].shift(1) - df[col].shift(2)) / (df[col].shift(2) + 1e-6)
    
    # 利率滞后特征
    for col in rate_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag7'] = df[col].shift(7)
        df[f'{col}_mean7'] = df[col].shift(1).rolling(7).mean()
    
    return df

# 构造初始特征
daily_data = add_history_features(daily_data)

# ========== 6. 特征列定义 ==========
feature_cols = [
    'weekday', 'day_of_month', 'month', 'week_of_year',
    'is_month_start', 'is_month_end', 'is_mid_month', 'is_weekend', 'holiday_type',
    'total_purchase_lag1', 'total_purchase_lag2', 'total_purchase_lag7', 'total_purchase_lag14',
    'total_purchase_mean3', 'total_purchase_mean7', 'total_purchase_mean14',
    'total_purchase_std7', 'total_purchase_weekday_mean', 'total_purchase_rate_lag1',
    'total_redeem_lag1', 'total_redeem_lag2', 'total_redeem_lag7', 'total_redeem_lag14',
    'total_redeem_mean3', 'total_redeem_mean7', 'total_redeem_mean14',
    'total_redeem_std7', 'total_redeem_weekday_mean', 'total_redeem_rate_lag1',
    'mfd_daily_yield', 'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_M',
    'mfd_daily_yield_lag1', 'mfd_daily_yield_lag7', 'Interest_O_N_mean7'
]

# ========== 7. 模型参数 ==========
def get_models():
    lgb_params = dict(
        n_estimators=800, learning_rate=0.03, num_leaves=20,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    xgb_params = dict(
        n_estimators=800, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    return LGBMRegressor(**lgb_params), XGBRegressor(**xgb_params)

# ========== 8. 递归逐天预测（核心提升点） ==========
history = daily_data.copy()
predict_dates = pd.date_range('2014-09-01', '2014-09-30', freq='D')
result = []

for pred_date in predict_dates:
    # 构造当天样本
    day_row = pd.DataFrame({'report_date': [pred_date]})
    day_row = add_time_features(day_row)
    
    # 拼接历史计算特征
    temp = pd.concat([history, day_row], ignore_index=True)
    temp = add_history_features(temp)
    
    test_row = temp.iloc[-1:][feature_cols]
    train_data = temp.iloc[:-1].dropna(subset=feature_cols).reset_index(drop=True)
    
    # 对数变换目标值
    train_data['log_purchase'] = np.log1p(train_data['total_purchase'])
    train_data['log_redeem'] = np.log1p(train_data['total_redeem'])
    
    # 训练申购双模型
    lgb_pur, xgb_pur = get_models()
    lgb_pur.fit(train_data[feature_cols], train_data['log_purchase'])
    xgb_pur.fit(train_data[feature_cols], train_data['log_purchase'])
    
    # 训练赎回双模型
    lgb_red, xgb_red = get_models()
    lgb_red.fit(train_data[feature_cols], train_data['log_redeem'])
    xgb_red.fit(train_data[feature_cols], train_data['log_redeem'])
    
    # 预测 + 指数还原 + 加权融合
    pred_pur_lgb = np.expm1(lgb_pur.predict(test_row)[0])
    pred_pur_xgb = np.expm1(xgb_pur.predict(test_row)[0])
    purchase = int(pred_pur_lgb * 0.6 + pred_pur_xgb * 0.4)
    
    pred_red_lgb = np.expm1(lgb_red.predict(test_row)[0])
    pred_red_xgb = np.expm1(xgb_red.predict(test_row)[0])
    redeem = int(pred_red_lgb * 0.6 + pred_red_xgb * 0.4)
    
    result.append([pred_date.strftime('%Y%m%d'), purchase, redeem])
    
    # 预测值回填，用于下一天特征计算
    day_row['total_purchase'] = purchase
    day_row['total_redeem'] = redeem
    for col in rate_cols:
        day_row[col] = history[col].iloc[-1]
    history = pd.concat([history, day_row], ignore_index=True)
    
    print(f"预测完成：{pred_date.strftime('%Y-%m-%d')} | 申购：{purchase} | 赎回：{redeem}")

# ========== 9. 业务规则后处理 ==========
result_df = pd.DataFrame(result, columns=['report_date', 'purchase', 'redeem'])

# 周末修正
def weekend_correction(row):
    date = datetime.strptime(row['report_date'], '%Y%m%d')
    if date.weekday() >= 5:
        row['purchase'] = int(row['purchase'] * 0.62)
        row['redeem'] = int(row['redeem'] * 0.58)
    return row
result_df = result_df.apply(weekend_correction, axis=1)

# 中秋节修正
mask = result_df['report_date'] == '20140908'
result_df.loc[mask, 'purchase'] = (result_df.loc[mask, 'purchase'] * 0.5).astype(int)
result_df.loc[mask, 'redeem'] = (result_df.loc[mask, 'redeem'] * 0.7).astype(int)

# 月末赎回修正
end_mask = result_df['report_date'].isin(['20140928','20140929','20140930'])
result_df.loc[end_mask, 'redeem'] = (result_df.loc[end_mask, 'redeem'] * 1.08).astype(int)

result_df['purchase'] = result_df['purchase'].clip(lower=0)
result_df['redeem'] = result_df['redeem'].clip(lower=0)