"""
Utility functions for calculating the official US EPA Air Quality Index (AQI)
based on raw pollutant concentrations.
"""

def calc_aqi_subindex(c, breakpoints):
    """Calculate US EPA AQI subindex given concentration c and list of tuples (C_low, C_high, I_low, I_high)"""
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= c <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
            return round(aqi)
    
    last = breakpoints[-1]
    aqi = ((last[3] - last[2]) / (last[1] - last[0])) * (c - last[0]) + last[2]
    return round(aqi)

def get_overall_aqi(pm25, pm10, no2, so2, co):
    """Convert raw hourly concentrations to US EPA AQI."""
    
    pm25_bp = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    pm10_bp = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)]
    
    no2_ppb = no2 * 0.53
    no2_bp = [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)]
    
    so2_ppb = so2 * 0.38
    so2_bp = [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400), (805, 1004, 401, 500)]
    
    co_ppm = co * 0.000873
    co_bp = [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)]
    
    aqi_pm25 = calc_aqi_subindex(pm25, pm25_bp)
    aqi_pm10 = calc_aqi_subindex(pm10, pm10_bp)
    aqi_no2 = calc_aqi_subindex(no2_ppb, no2_bp)
    aqi_so2 = calc_aqi_subindex(so2_ppb, so2_bp)
    aqi_co = calc_aqi_subindex(co_ppm, co_bp)
    
    return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_so2, aqi_co)
