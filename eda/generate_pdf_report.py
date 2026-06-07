import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def create_pdf():
    pdf_path = "EDA_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    title = Paragraph("AQI-Trendz: Exploratory Data Analysis Report", styles['Title'])
    Story.append(title)
    Story.append(Spacer(1, 12))
    
    intro = Paragraph("This report contains key visualizations from the Exploratory Data Analysis (EDA) of the air quality dataset, along with summaries of the findings.", styles['Normal'])
    Story.append(intro)
    Story.append(Spacer(1, 24))

    # Define images and their summaries
    plots = [
        {
            "img": "eda_plots/01_leading_pollutant.png",
            "title": "01. The 'Leading Pollutant'",
            "summary": "This chart shows which pollutant most frequently exceeds its WHO limit. In Karachi, PM2.5 and PM10 typically dominate as the primary threats to air quality, highlighting the severe particulate matter pollution."
        },
        {
            "img": "eda_plots/02_co_occurrence.png",
            "title": "02. Pollutant Co-occurrence Signatures",
            "summary": "The scatter plots illustrate relationships between pollutants. The NO₂ vs CO plot shows a 'traffic signature' (both from vehicle exhaust), while the PM2.5 vs PM10 plot shows a 'dust/smog signature'. Strong positive correlations indicate shared emission sources."
        },
        {
            "img": "eda_plots/03_wind_profiles.png",
            "title": "03. Wind Source Profiles",
            "summary": "These normalized wind rose plots show the directional origin of each pollutant. They help identify whether pollution is blowing in from industrial zones, traffic hubs, or being dispersed by the sea breeze from the Arabian Sea."
        },
        {
            "img": "eda_plots/04_weekend_shift.png",
            "title": "04. Weekend vs Weekday Shift",
            "summary": "This bar chart analyzes the percentage change in pollutants on weekends compared to weekdays. A drop in levels (negative shift) typically reflects reduced commuter traffic and industrial activity on weekends."
        },
        {
            "img": "eda_plots/05_weather_impacts.png",
            "title": "05. Weather Interaction Profiles",
            "summary": "The correlation heatmap reveals how weather affects pollution. For example, wind speed usually has a negative correlation (dispersion), while humidity might be positively correlated with certain particulates."
        },
        {
            "img": "eda_plots/06_pm_ratio_seasonality.png",
            "title": "06. Seasonal Ratio Shifts (PM2.5 / PM10)",
            "summary": "This line chart tracks the PM2.5 to PM10 ratio over the year. A higher ratio indicates more toxic fine smog (often in winter due to inversions), while a lower ratio points to coarse dust (often during summer or dry periods)."
        },
        {
            "img": "eda_plots_extra/01_aqi_category_pie.png",
            "title": "07. AQI Category Distribution",
            "summary": "The pie chart breaks down the historical data by US EPA AQI categories. It provides a clear view of how often the air quality is 'Good', 'Moderate', or 'Unhealthy', underscoring the overall health risk profile."
        },
        {
            "img": "eda_plots_extra/02_heatmap_hour_vs_dow_aqi.png",
            "title": "08. AQI Heatmap: Hour vs Day of Week",
            "summary": "This heatmap highlights temporal pollution patterns. The highest AQI values typically cluster around morning and evening rush hours on weekdays, confirming the massive impact of urban traffic on air quality."
        }
    ]

    for plot in plots:
        if os.path.exists(plot["img"]):
            # Add title
            Story.append(Paragraph(plot["title"], styles['Heading2']))
            Story.append(Spacer(1, 12))
            
            # Add image
            try:
                img = Image(plot["img"])
                # Resize image to fit page width while maintaining aspect ratio
                max_width = 6.5 * inch
                max_height = 4 * inch
                ratio = min(max_width / img.drawWidth, max_height / img.drawHeight)
                img.drawWidth = img.drawWidth * ratio
                img.drawHeight = img.drawHeight * ratio
                Story.append(img)
            except Exception as e:
                Story.append(Paragraph(f"[Error loading image: {e}]", styles['Normal']))
            
            Story.append(Spacer(1, 12))
            
            # Add summary
            Story.append(Paragraph(plot["summary"], styles['Normal']))
            Story.append(Spacer(1, 24))

    doc.build(Story)
    print(f"Successfully generated {pdf_path}")

if __name__ == '__main__':
    create_pdf()
