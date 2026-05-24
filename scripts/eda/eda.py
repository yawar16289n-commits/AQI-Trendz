import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    print("Loading final historical data for EDA...")
    df = pd.read_csv('../../data/cleaned/final_historical_data.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    # Create directory for saving plots
    os.makedirs('../../eda_plots', exist_ok=True)
    
    print("\n--- Basic Statistics ---")
    print(df.describe().to_string())
    
    # 1. Correlation Matrix Heatmap
    print("\nGenerating Correlation Matrix...")
    plt.figure(figsize=(12, 10))
    numeric_df = df.drop(columns=['time'])
    corr_matrix = numeric_df.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of AQI and Weather Features')
    plt.tight_layout()
    plt.savefig('../../eda_plots/correlation_matrix.png', dpi=300)
    plt.close()
    
    # 2. Time Series Plot of PM2.5 and PM10
    print("Generating Time Series Plot for Pollutants...")
    plt.figure(figsize=(15, 6))
    plt.plot(df['time'], df['pm2_5'], label='PM 2.5', alpha=0.7)
    plt.plot(df['time'], df['pm10'], label='PM 10', alpha=0.7)
    plt.title('PM2.5 and PM10 Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Concentration (μg/m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../eda_plots/pollutant_trends.png', dpi=300)
    plt.close()
    
    # 3. Distribution of PM2.5
    print("Generating PM2.5 Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pm2_5'], bins=50, kde=True, color='maroon')
    plt.title('Distribution of PM2.5 Values in Karachi')
    plt.xlabel('PM2.5 Concentration')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../eda_plots/pm25_distribution.png', dpi=300)
    plt.close()
    
    print("\n✅ EDA completed! Plots saved in the 'eda_plots' directory.")

if __name__ == "__main__":
    perform_eda()
