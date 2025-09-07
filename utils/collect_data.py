import asyncio
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
from binance import AsyncClient
from datetime import datetime
from dateutil import parser

async def fetch_klines(
    symbol='BTCUSDT',
    interval='1m',
    start_date='2025-08-08',
    end_date='2025-09-07',
    output_path='data/klines/klines_BTCUSDT_default.parquet'
):
    """
    Fetches historical 1-minute kline data for BTCUSDT from Binance Spot API and saves it as a Parquet file.

    Args:
        symbol (str): Trading pair symbol, e.g., 'BTCUSDT'.
        interval (str): Timeframe for klines, e.g., '1m' for 1-minute.
        start_date (str): Start date for data, e.g., '2025-08-08'.
        end_date (str): End date for data, e.g., '2025-09-07'.
        output_path (str): Path to save the Parquet file.

    Returns:
        bool: True if data is successfully fetched and saved, False otherwise.
    """
    try:
        # Initialize Binance client
        client = await AsyncClient.create()
        print(f"✅ Initialized Binance Spot client for {symbol} {interval}")

        # Convert dates to milliseconds
        start_time = int(parser.parse(start_date).timestamp() * 1000)
        end_time = int(parser.parse(end_date).timestamp() * 1000)
        max_batch = 1000  # Maximum records per request

        klines = []
        current_time = start_time

        print("Fetching klines...")
        while current_time < end_time:
            batch_klines = await client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=max_batch,
                startTime=current_time,
                endTime=end_time
            )
            if not batch_klines:
                print("✅ No more klines available")
                break

            klines.extend(batch_klines)
            print(f"✅ Fetched {len(batch_klines)} klines, total: {len(klines)}")
            current_time = int(batch_klines[-1][0]) + 1  # Move to next millisecond
            if len(batch_klines) < max_batch:
                break
            await asyncio.sleep(0.2)  # Respect rate limits

        if not klines:
            print("❌ No kline data retrieved")
            return False

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
            'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
        })
        print(f"✅ Processed {len(df)} kline records from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')
        print(f"✅ Saved kline data to: {output_path}")

        # Verify saved file
        if os.path.exists(output_path):
            saved_df = pd.read_parquet(output_path)
            print(f"✅ Verified saved data: {len(saved_df)} rows with columns {list(saved_df.columns)}")
            return True
        else:
            print(f"❌ Failed to verify saved file: {output_path}")
            return False

    except Exception as e:
        print(f"❌ Error fetching klines: {e}")
        return False
    finally:
        await client.close_connection()
        print("✅ Binance client closed")

if __name__ == "__main__":
    asyncio.run(fetch_klines())
