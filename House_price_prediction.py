import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "Area":[1000,1200,1500,1800,2000],
    "Bedrooms":[2,2,3,3,4],
    "Price":[3000000,3500000,4500000,5500000,6500000]
}

df = pd.DataFrame(data)

X = df[["Area","Bedrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

area = float(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))

new_house = pd.DataFrame({
    "Area":[area],
    "Bedrooms":[bedrooms]
})

prediction = model.predict(new_house)

print(f"Predicted House Price = ₹{prediction[0]:,.2f}")