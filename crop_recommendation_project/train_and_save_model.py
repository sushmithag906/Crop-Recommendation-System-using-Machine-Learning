import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/Crop_recommendation.csv")

X = data[['N','P','K','temperature','humidity','ph','rainfall']]
y = data['label']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model correctly (BINARY)
with open("crop_recommendation_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved correctly")
