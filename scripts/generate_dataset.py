#!/usr/bin/env python
"""
Synthetic E-Commerce Demand Data Generator

Generates realistic e-commerce sales data with multiple products and warehouses,
including patterns such as seasonality, promotions, and demand spikes.

Usage:
    python generate_dataset.py --products 5 --warehouses 3 --days 730 --output-full ../data/raw/sales_data.csv --output-sample ../data/sample/sample_sales_data.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_data(
    num_products=5,
    num_warehouses=3,
    num_days=730,
    random_seed=42,
    start_date='2022-01-01'
):
    """
    Generate synthetic e-commerce demand data.
    
    Args:
        num_products: Number of products to generate data for
        num_warehouses: Number of warehouses
        num_days: Number of days of data to generate
        random_seed: Random seed for reproducibility
        start_date: Start date for the dataset (YYYY-MM-DD format)
    
    Returns:
        DataFrame with columns: date, product_id, warehouse_id, units_sold, 
                               stock_level, price, promotion, temperature, event
    """
    
    np.random.seed(random_seed)
    
    # Parse start date
    start = pd.to_datetime(start_date)
    
    # Generate date range
    dates = pd.date_range(start=start, periods=num_days, freq='D')
    
    # Product and warehouse IDs
    products = [f'SKU-{i:04d}' for i in range(1, num_products + 1)]
    warehouses = [f'WH-{i:02d}' for i in range(1, num_warehouses + 1)]
    
    # Create base data
    data = []
    
    # Product characteristics (base demand, seasonality pattern)
    product_chars = {
        products[i]: {
            'base_demand': np.random.uniform(20, 100),
            'seasonality_strength': np.random.uniform(0.3, 0.8),
            'price_sensitivity': np.random.uniform(0.5, 1.5),
        }
        for i in range(num_products)
    }
    
    # Warehouse characteristics (multiplier for demand)
    warehouse_chars = {
        warehouses[j]: np.random.uniform(0.7, 1.3)
        for j in range(num_warehouses)
    }
    
    for date in dates:
        day_of_year = date.dayofyear
        day_of_week = date.dayofweek  # 0=Monday, 6=Sunday
        month = date.month
        
        # Events: holiday spikes and special events
        is_holiday = month in [11, 12]  # November, December holidays
        is_black_friday = (month == 11 and day_of_week == 4 and 20 <= date.day <= 30)
        is_event = np.random.random() < 0.02  # Random events 2% of days
        
        # Temperature (varies seasonally, affects demand for certain products)
        temp = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2)
        
        for product in products:
            for warehouse in warehouses:
                # Base demand pattern
                base = product_chars[product]['base_demand']
                warehouse_mult = warehouse_chars[warehouse]
                
                # Weekly seasonality (higher on weekends)
                if day_of_week >= 5:  # Saturday, Sunday
                    weekly_mult = 1.3
                else:
                    weekly_mult = 1.0
                
                # Monthly/seasonal pattern
                seasonal_mult = 1 + product_chars[product]['seasonality_strength'] * np.sin(
                    2 * np.pi * day_of_year / 365
                )
                
                # Promotion effect (random ~5% of days have promotions)
                promotion = np.random.random() < 0.05
                promo_mult = 1.5 if promotion else 1.0
                
                # Holiday effect
                holiday_mult = 2.5 if is_black_friday else (1.8 if is_holiday else 1.0)
                
                # General event spike
                event_mult = 3.0 if is_event else 1.0
                
                # Price (inverse relationship with demand)
                base_price = np.random.uniform(10, 100)
                price_variance = np.random.normal(0, 0.1)
                price = base_price * (1 + price_variance)
                price_mult = 1 - product_chars[product]['price_sensitivity'] * price_variance
                
                # Combined demand
                demand = (
                    base * 
                    warehouse_mult * 
                    weekly_mult * 
                    seasonal_mult * 
                    promo_mult * 
                    holiday_mult * 
                    event_mult * 
                    price_mult
                )
                
                # Add random noise
                demand += np.random.normal(0, demand * 0.15)  # 15% noise
                demand = max(0, int(np.round(demand)))
                
                # Stock level (simplified: assume replenishment every 10 days)
                # Add some randomness and account for sales
                stock_base = demand * np.random.uniform(10, 30)  # Stock covers 10-30 days
                stock_level = max(0, int(np.round(stock_base)))
                
                # Event label
                event_label = None
                if is_black_friday:
                    event_label = 'BlackFriday'
                elif is_holiday:
                    event_label = 'Holiday'
                elif is_event:
                    event_label = 'SpecialEvent'
                
                data.append({
                    'date': date,
                    'product_id': product,
                    'warehouse_id': warehouse,
                    'units_sold': demand,
                    'stock_level': stock_level,
                    'price': round(price, 2),
                    'promotion': promotion,
                    'temperature': round(temp, 1),
                    'event': event_label,
                })
    
    df = pd.DataFrame(data)
    
    # Ensure data types
    df['date'] = pd.to_datetime(df['date'])
    df['promotion'] = df['promotion'].astype(bool)
    
    return df.sort_values(['date', 'product_id', 'warehouse_id']).reset_index(drop=True)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic e-commerce demand data for SmartStock AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default dataset (5 products, 3 warehouses, 730 days)
  python generate_dataset.py
  
  # Generate larger dataset with custom parameters
  python generate_dataset.py --products 10 --warehouses 5 --days 1095
  
  # Generate and save to custom locations
  python generate_dataset.py --output-full custom_full.csv --output-sample custom_sample.csv
        """
    )
    
    parser.add_argument(
        '--products',
        type=int,
        default=5,
        help='Number of products to generate (default: 5)'
    )
    parser.add_argument(
        '--warehouses',
        type=int,
        default=3,
        help='Number of warehouses (default: 3)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='Number of days of data to generate (default: 730, ~2 years)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2022-01-01',
        help='Start date for data generation (YYYY-MM-DD format, default: 2022-01-01)'
    )
    parser.add_argument(
        '--output-full',
        type=str,
        default='../data/raw/sales_data.csv',
        help='Output path for full dataset (default: ../data/raw/sales_data.csv)'
    )
    parser.add_argument(
        '--output-sample',
        type=str,
        default='../data/sample/sample_sales_data.csv',
        help='Output path for sample dataset (default: ../data/sample/sample_sales_data.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SmartStock AI - Synthetic Data Generator")
    print("=" * 70)
    print()
    print(f"Parameters:")
    print(f"  Products:     {args.products}")
    print(f"  Warehouses:   {args.warehouses}")
    print(f"  Days:         {args.days} (~{args.days/365:.1f} years)")
    print(f"  Start Date:   {args.start_date}")
    print(f"  Random Seed:  {args.seed}")
    print()
    
    # Generate full dataset
    print("Generating synthetic data...")
    df = generate_synthetic_data(
        num_products=args.products,
        num_warehouses=args.warehouses,
        num_days=args.days,
        random_seed=args.seed,
        start_date=args.start_date,
    )
    
    print(f"  Generated {len(df):,} records")
    print()
    
    # Create output directories if they don't exist
    full_path = Path(args.output_full)
    sample_path = Path(args.output_sample)
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    print(f"Saving full dataset to: {full_path}")
    df.to_csv(full_path, index=False)
    print(f"  ✓ Saved {len(df):,} records")
    
    # Save sample dataset (first 3 months per product/warehouse combination)
    print()
    print("Generating sample dataset (first 90 days)...")
    sample_df = df[df['date'] <= df['date'].min() + timedelta(days=90)]
    print(f"  Generated {len(sample_df):,} records")
    
    print(f"Saving sample dataset to: {sample_path}")
    sample_df.to_csv(sample_path, index=False)
    print(f"  ✓ Saved {len(sample_df):,} records")
    
    # Print summary statistics
    print()
    print("Dataset Summary Statistics:")
    print("-" * 70)
    print(f"Date Range:        {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Products:          {df['product_id'].nunique()}")
    print(f"Warehouses:        {df['warehouse_id'].nunique()}")
    print(f"Total Records:     {len(df):,}")
    print()
    print(f"Units Sold:")
    print(f"  Mean:            {df['units_sold'].mean():.1f}")
    print(f"  Std Dev:         {df['units_sold'].std():.1f}")
    print(f"  Min:             {df['units_sold'].min():.0f}")
    print(f"  Max:             {df['units_sold'].max():.0f}")
    print()
    print(f"Price:")
    print(f"  Mean:            ${df['price'].mean():.2f}")
    print(f"  Min:             ${df['price'].min():.2f}")
    print(f"  Max:             ${df['price'].max():.2f}")
    print()
    print(f"Promotions:        {df['promotion'].sum():,} days ({df['promotion'].mean()*100:.1f}%)")
    print(f"Events:            {df['event'].notna().sum():,} records ({df['event'].notna().mean()*100:.1f}%)")
    print()
    print("=" * 70)
    print("✓ Dataset generation complete!")
    print()


if __name__ == '__main__':
    main()
