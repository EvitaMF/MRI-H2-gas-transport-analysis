import pandas as pd

# Constants
R = 8.314462618  # J/(mol K)
T = 298.15       # K

# Molar masses [kg/mol]
M_CH4 = 16.043e-3
M_H2 = 2.016e-3

# density values from NIST [kg/m3]
data = {
    "Pressure_bar": [9.9, 20.2, 30.5, 40.3, 50.7, 60.9, 70.7, 80.8, 89.5, 100.9, 109.4],
    "Density_kg_m3": [6.52, 13.5, 20.7, 28.1, 35.8, 43.7, 51.8, 60.1, 68.3, 76.8, 85.3]
}

df = pd.DataFrame(data)

# Convert pressure from bar to Pa
df["Pressure_Pa"] = df["Pressure_bar"] * 1e5

# Compressibility factor:
# Z = pM / (rho R T)
df["Z_CH4"] = df["Pressure_Pa"] * M_CH4 / (df["Density_kg_m3"] * R * T)

print(df[["Pressure_bar", "Density_kg_m3", "Z_CH4"]])
df.to_excel("CH4_Z_factor_from_NIST.xlsx", index=False)