import pandas as pd

def clean_aqi_data(file_path):
    print(f"Loading {file_path}...")
    
    # Open-Meteo CSVs have 3 rows of metadata at the top
    # The actual data starts at row 4 (index 3)
    df = pd.read_csv(file_path, skiprows=3)
    
    print(f"Original shape: {df.shape}")
    
    # Rename columns to simpler names if needed, but let's keep them as is for now
    # We just need to make sure the time column is datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Drop rows where ALL pollutant values are NaN (usually the first few months of 2022)
    pollutant_cols = [col for col in df.columns if col != 'time']
    df.dropna(subset=pollutant_cols, how='all', inplace=True)
    
    print(f"Shape after dropping completely empty rows: {df.shape}")
    
    # Forward fill any intermittent missing values
    df.ffill(inplace=True)
    
    # Save the cleaned dataset
    output_file = '../../data/cleaned/aqi_cleaned.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully cleaned and saved to {output_file}!")
    print("\nFirst 5 rows of cleaned data:")
    print(df.head())

if __name__ == "__main__":
    input_file = '../../data/raw/aqi-hist.csv'
    clean_aqi_data(input_file)
