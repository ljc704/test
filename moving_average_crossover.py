"""
均線交叉策略（Moving Average Crossover）完整範例

功能包含：
1) 計算短期/長期均線
2) 產生買賣訊號、持有部位
3) 計算策略與買進持有報酬
4) 繪製價格/訊號與累積報酬圖
5) 計算績效指標（總報酬、年化報酬、最大回撤、勝率、交易次數）

可直接執行：
- 無提供 CSV 時，會自動使用範例假資料
- 有提供 CSV 時，需至少包含 Date 與 Close 欄位（可用參數對應中文欄位）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """策略參數設定。"""

    short_window: int = 5
    long_window: int = 20
    annual_trading_days: int = 252


def load_price_data(
    csv_path: str | None = None,
    date_col: str = "Date",
    close_col: str = "Close",
) -> pd.DataFrame:
    """
    載入價格資料並完成前處理。

    需求：
    - 至少要有日期與收盤價欄位
    - 日期轉 datetime 並排序
    - 回傳僅含 Date, Close 的 DataFrame
    """
    if csv_path is None:
        df = generate_sample_data()
    else:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"找不到檔案：{csv_path}") from exc
        except Exception as exc:
            raise ValueError(f"讀取 CSV 失敗：{exc}") from exc

    # 欄位名稱標準化（支援常見中文）
    rename_map = {
        "日期": "Date",
        "交易日期": "Date",
        "收盤": "Close",
        "收盤價": "Close",
        "收盘价": "Close",
    }
    df = df.rename(columns=rename_map)

    if date_col != "Date" and date_col in df.columns:
        df = df.rename(columns={date_col: "Date"})
    if close_col != "Close" and close_col in df.columns:
        df = df.rename(columns={close_col: "Close"})

    required_cols = {"Date", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}，請確認 CSV 至少包含 Date 與 Close")

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("資料筆數不足，至少需要 2 筆以上資料。")

    return df


def generate_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """計算均線、買賣訊號與持有部位。"""
    if config.short_window <= 0 or config.long_window <= 0:
        raise ValueError("均線週期必須為正整數。")
    if config.short_window >= config.long_window:
        raise ValueError("短期均線週期建議小於長期均線週期。")

    data = df.copy()
    data["SMA_Short"] = data["Close"].rolling(window=config.short_window, min_periods=1).mean()
    data["SMA_Long"] = data["Close"].rolling(window=config.long_window, min_periods=1).mean()

    # 黃金交叉 / 死亡交叉判斷
    cross_up = (data["SMA_Short"] > data["SMA_Long"]) & (
        data["SMA_Short"].shift(1) <= data["SMA_Long"].shift(1)
    )
    cross_down = (data["SMA_Short"] < data["SMA_Long"]) & (
        data["SMA_Short"].shift(1) >= data["SMA_Long"].shift(1)
    )

    data["Buy_Signal"] = cross_up.astype(int)
    data["Sell_Signal"] = cross_down.astype(int)

    # Position: 持有=1, 空手=0（訊號當日收盤後建立，隔日套用到報酬）
    signal = np.where(data["Buy_Signal"] == 1, 1, np.where(data["Sell_Signal"] == 1, 0, np.nan))
    data["Position"] = pd.Series(signal, index=data.index).ffill().fillna(0).astype(int)

    return data


def compute_returns(data: pd.DataFrame) -> pd.DataFrame:
    """計算策略日報酬、累積報酬，以及買進持有累積報酬。"""
    result = data.copy()
    result["Market_Return"] = result["Close"].pct_change().fillna(0)
    result["Strategy_Return"] = result["Position"].shift(1).fillna(0) * result["Market_Return"]

    result["Strategy_Equity"] = (1 + result["Strategy_Return"]).cumprod()
    result["BuyHold_Equity"] = (1 + result["Market_Return"]).cumprod()

    result["Strategy_CumReturn"] = result["Strategy_Equity"] - 1
    result["BuyHold_CumReturn"] = result["BuyHold_Equity"] - 1
    return result


def extract_trade_returns(data: pd.DataFrame) -> list[float]:
    """擷取每筆完整交易（買到賣）的報酬，用於勝率與交易次數。"""
    trades = []
    in_position = False
    entry_price = None

    for _, row in data.iterrows():
        if row["Buy_Signal"] == 1 and not in_position:
            in_position = True
            entry_price = row["Close"]
        elif row["Sell_Signal"] == 1 and in_position and entry_price is not None:
            trades.append(row["Close"] / entry_price - 1)
            in_position = False
            entry_price = None

    return trades


def calculate_performance_metrics(data: pd.DataFrame, config: StrategyConfig) -> Dict[str, float]:
    """計算績效指標：總報酬、年化報酬、最大回撤、勝率、交易次數。"""
    equity = data["Strategy_Equity"]

    total_return = equity.iloc[-1] - 1

    n_days = len(data)
    if n_days > 1:
        annual_return = equity.iloc[-1] ** (config.annual_trading_days / n_days) - 1
    else:
        annual_return = np.nan

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()

    trades = extract_trade_returns(data)
    trade_count = len(trades)
    win_rate = (sum(1 for r in trades if r > 0) / trade_count) if trade_count > 0 else np.nan

    return {
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Number of Trades": float(trade_count),
    }


def plot_results(data: pd.DataFrame, config: StrategyConfig) -> None:
    """繪製價格/均線/買賣點，以及策略 vs 買進持有累積報酬。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 圖1：價格、均線、買賣點
    axes[0].plot(data["Date"], data["Close"], label="Close", color="black", linewidth=1.2)
    axes[0].plot(
        data["Date"],
        data["SMA_Short"],
        label=f"SMA {config.short_window}",
        color="royalblue",
        linewidth=1.2,
    )
    axes[0].plot(
        data["Date"],
        data["SMA_Long"],
        label=f"SMA {config.long_window}",
        color="darkorange",
        linewidth=1.2,
    )

    buy_points = data[data["Buy_Signal"] == 1]
    sell_points = data[data["Sell_Signal"] == 1]

    axes[0].scatter(
        buy_points["Date"],
        buy_points["Close"],
        marker="^",
        color="green",
        s=80,
        label="Buy",
        zorder=3,
    )
    axes[0].scatter(
        sell_points["Date"],
        sell_points["Close"],
        marker="v",
        color="red",
        s=80,
        label="Sell",
        zorder=3,
    )

    axes[0].set_title("Moving Average Crossover: Price & Signals")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    # 圖2：累積報酬比較
    axes[1].plot(
        data["Date"],
        data["Strategy_CumReturn"],
        label="Strategy Cumulative Return",
        color="seagreen",
        linewidth=1.5,
    )
    axes[1].plot(
        data["Date"],
        data["BuyHold_CumReturn"],
        label="Buy & Hold Cumulative Return",
        color="gray",
        linewidth=1.5,
        linestyle="--",
    )
    axes[1].set_title("Cumulative Return Comparison")
    axes[1].set_ylabel("Cumulative Return")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_sample_data(rows: int = 260) -> pd.DataFrame:
    """
    建立可直接測試的台股風格範例資料：Date, Close。
    （營業日頻率 + 隨機漫步）
    """
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-01-02", periods=rows)

    daily_returns = np.random.normal(loc=0.0005, scale=0.015, size=rows)
    prices = 100 * np.cumprod(1 + daily_returns)

    sample = pd.DataFrame(
        {
            "Date": dates,
            "Close": prices.round(2),
        }
    )
    return sample


def run_strategy(
    csv_path: str | None,
    short_window: int,
    long_window: int,
    date_col: str,
    close_col: str,
    show_plot: bool,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """策略主流程：載入資料 -> 訊號 -> 報酬 -> 績效。"""
    config = StrategyConfig(short_window=short_window, long_window=long_window)

    data = load_price_data(csv_path=csv_path, date_col=date_col, close_col=close_col)
    data = generate_signals(data, config)
    data = compute_returns(data)
    metrics = calculate_performance_metrics(data, config)

    if show_plot:
        plot_results(data, config)

    return data, metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """格式化輸出績效指標。"""
    print("\n=== 策略績效指標 ===")
    print(f"總報酬率 (Total Return): {metrics['Total Return']:.2%}")
    print(f"年化報酬率 (Annualized Return): {metrics['Annualized Return']:.2%}")
    print(f"最大回撤 (Max Drawdown): {metrics['Max Drawdown']:.2%}")

    win_rate = metrics["Win Rate"]
    if np.isnan(win_rate):
        print("勝率 (Win Rate): N/A（尚無完整交易）")
    else:
        print(f"勝率 (Win Rate): {win_rate:.2%}")

    print(f"交易次數 (Number of Trades): {int(metrics['Number of Trades'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="均線交叉策略回測程式")
    parser.add_argument("--csv", type=str, default=None, help="價格資料 CSV 路徑")
    parser.add_argument("--short-window", type=int, default=5, help="短期均線天數（預設 5）")
    parser.add_argument("--long-window", type=int, default=20, help="長期均線天數（預設 20）")
    parser.add_argument("--date-col", type=str, default="Date", help="日期欄位名稱")
    parser.add_argument("--close-col", type=str, default="Close", help="收盤價欄位名稱")
    parser.add_argument("--no-plot", action="store_true", help="不顯示圖表")
    parser.add_argument(
        "--save-sample-csv",
        type=str,
        default=None,
        help="將範例假資料輸出成 CSV（例如 sample_tw_stock.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.save_sample_csv:
        sample = generate_sample_data()
        sample.to_csv(args.save_sample_csv, index=False, encoding="utf-8-sig")
        print(f"已輸出範例資料：{args.save_sample_csv}")

    try:
        data, metrics = run_strategy(
            csv_path=args.csv,
            short_window=args.short_window,
            long_window=args.long_window,
            date_col=args.date_col,
            close_col=args.close_col,
            show_plot=not args.no_plot,
        )
        print_metrics(metrics)

        print("\n=== 最近 5 筆結果 ===")
        cols = [
            "Date",
            "Close",
            "SMA_Short",
            "SMA_Long",
            "Buy_Signal",
            "Sell_Signal",
            "Position",
            "Strategy_CumReturn",
            "BuyHold_CumReturn",
        ]
        print(data[cols].tail(5).to_string(index=False))

    except Exception as exc:
        print(f"執行失敗：{exc}")


if __name__ == "__main__":
    main()
