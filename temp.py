import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
path = r"C:\Users\gudap\Desktop\T1.csv"
df = pd.read_csv(path)
df.rename(columns={
    'Date/Time': 'Time',
    'LV ActivePower(KW)': 'ActivePower(KW)',
    'Wins Speed (m/s)': 'WindSpeed(m/s)',
    'Wind Direction': 'Wind_Direction'
}, inplace=True)
sns.pairplot(df)
plt.show()
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

ax = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Correlation Heatmap')
plt.show()

# Print the numerical correlations
print("Correlation Matrix:")
print(corr)
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from CSV file
path = r"C:\Users\gudap\Desktop\T1.csv"
df = pd.read_csv(path)

# Rename columns for easier access (if needed)
df.rename(columns={
    'Date/Time': 'Time',
    'LV ActivePower (kW)': 'ActivePower(KW)',
    'Theoretical_Power_Curve (KWh)': 'Theoretical_Power_Curve(KWh)',
    'Wind Speed (m/s)': 'WindSpeed(m/s)',
    'Wind Direction (°)': 'Wind_Direction'  
}, inplace=True)
df.columns = df.columns.str.strip()
if 'ActivePower(KW)' in df.columns:
    y = df['ActivePower(KW)']
else:
    print("Column 'ActivePower(KW)' does not exist in the DataFrame")
y = df['ActivePower(KW)']  
X = df[['Theoretical_Power_Curve(KWh)', 'WindSpeed(m/s)', 'Wind_Direction']]  # Features, adjusted to match actual column names

# Splitting the data into train and test sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Example to print shapes to verify split
print("Training Data Shapes:")
print("Features (train_X):", train_X.shape)
print("Target (train_y):", train_y.shape)
print("\nValidation Data Shapes:")
print("Features (val_X):", val_X.shape)
print("Target (val_y):", val_y.shape)

forest_model = RandomForestRegressor(n_estimators = 750, max_depth = 4, max_leaf_nodes =500, random_state=1)
forest_model.fit(train_X, train_y)
#Predicting for Test Data
power_preds = forest_model.predict(val_X)
#Evaluating the score of our model
print("Mean Absolute Error:",mean_absolute_error(val_y, power_preds))
print("R^2 Score:",r2_score(val_y,power_preds))
#Saving the model for future reference
joblib.dump(forest_model, "power_prediction.sav")



