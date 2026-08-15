import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Preprocessing steps
df.drop(columns=["CustomerID","Unnamed: 0"], inplace=True)

df['Gender'].replace(to_replace="Fe Male", value="Female", inplace=True)
df['MaritalStatus'].replace(to_replace="Unmarried", value="Single", inplace=True)
# drop null values if exist
df.dropna(axis = 0,inplace = True)

# Train and Test dataset preparation
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
