"""
Script to fetch stock_news table from Supabase and save locally.
"""
import os
import json
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

def fetch_stock_news():
    """Fetch stock_news table from Supabase and save to local data directory."""

    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Connecting to Supabase...")
    print(f"URL: {SUPABASE_URL}")

    # First, get the total count
    print("Getting total count...")
    count_response = supabase.table("stock_news").select("*", count="exact").limit(1).execute()
    total_count = count_response.count
    print(f"Total records in table: {total_count}")

    # Fetch all records using pagination
    print("Fetching stock_news table...")
    BATCH_SIZE = 1000
    all_data = []

    for offset in range(0, total_count, BATCH_SIZE):
        print(f"Fetching records {offset} to {min(offset + BATCH_SIZE, total_count)}...")
        response = supabase.table("stock_news").select("*").range(offset, offset + BATCH_SIZE - 1).execute()
        all_data.extend(response.data)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    print(f"\nFetched {len(df)} records from stock_news table")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Save to multiple formats
    csv_path = data_dir / "stock_news.csv"
    json_path = data_dir / "stock_news.json"

    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved to {csv_path}")

    df.to_json(json_path, orient="records", indent=2)
    print(f"✓ Saved to {json_path}")

    # Print basic statistics
    print(f"\nDataset Info:")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\nColumn types:")
    print(df.dtypes)

    return df

if __name__ == "__main__":
    df = fetch_stock_news()
