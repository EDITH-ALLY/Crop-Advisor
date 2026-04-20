import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

data = {
    'N': [90, 85, 60, 70],
    'P': [40, 50, 30, 35],
    'K': [40, 45, 30, 35],
    'temperature': [20, 25, 30, 28],
    'humidity': [80, 70, 60, 65],
    'ph': [6.5, 7, 6, 6.8],
    'rainfall': [200, 150, 100, 120],
    'label': ['Rice', 'Wheat', 'Maize', 'Cotton']
}

df = pd.DataFrame(data)

X = df.drop('label', axis=1)
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open('model/model.pkl', 'wb'))

print("Model trained and saved!")
