import logging
from functools import reduce
from typing import Dict
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

class FreqaiBTC2(IStrategy):

    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"&-s_close": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.1
    use_exit_signal = True
    startup_candle_count: int = 40
    can_short = True

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        # Include only RSI, day, hour, and the Sharpe signal
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        url_1hr = 'https://raw.githubusercontent.com/spmcelrath/images/main/bitcoin-bitcoin-sharpe-signal-short-1hr.csv'

        try:
            # Load Sharpe signals from URLs
            sharpe_signals = pd.read_csv(url_1hr, index_col='timestamp', parse_dates=True)
            
            # Check if 'sharpe_signal' column exists
            if 'value' not in sharpe_signals.columns:
                logger.error("The 'sharpe_signal' column is missing in the CSV file.")
                raise ValueError("The 'sharpe_signal' column is missing in the CSV file.")
            
            # Rename 'value' column to 'sharpe_signal' for consistency
            sharpe_signals.rename(columns={'value': 'sharpe_signal'}, inplace=True)

            # Resample to the same frequency as the OHLCV data
            sharpe_signals = sharpe_signals.resample(self.timeframe).ffill()

            # Debugging: Print the first few rows of the sharpe_signals dataframe
            logger.debug(f"Sharpe signals head:\n{sharpe_signals.head()}")

            # Join the signals with the dataframe and forward fill to match OHLCV data
            dataframe.set_index('date', inplace=True)
            dataframe = dataframe.join(sharpe_signals, how='left').ffill()
            dataframe.reset_index(inplace=True)

            # Debugging: Print the first few rows of the dataframe after joining
            logger.debug(f"Dataframe after join:\n{dataframe.head()}")

            # Save the desired columns to a CSV file
            save_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'sharpe_signal']
            save_dataframe = dataframe[save_columns]

            # Create a unique file name with a timestamp
            save_path = "user_data/backtest_results/results.csv"

            # Attempt to save the DataFrame to a CSV file
            save_dataframe.to_csv(save_path, index=False)

            # Incorporate the Sharpe signal as a feature
            dataframe["%-sharpe_signal"] = dataframe["sharpe_signal"]

        except Exception as e:
            logger.error(f"Error loading or processing Sharpe signals: {e}")
            raise

        # the model will return all labels created by user in `set_freqai_targets()`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `set_freqai_targets()` for each training period.

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > 0.01,
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < -0.01,
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["&-s_close"] < 0]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df["do_predict"] == 1, df["&-s_close"] > 0]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
