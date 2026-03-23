import streamlit as st
import pandas as pd
import qrcode
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Page setup
st.set_page_config(page_title="SmartPark AI", layout="wide")

# Dark theme
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.stApp {background-color: #0e1117;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 SmartPark AI – Intelligent Parking System")

# Load dataset
df = pd.read_csv("parking_data.csv")

# Dynamic pricing (₹10–₹30)
df["Price"] = df["Price"].apply(lambda x: random.randint(10,30))

# Auto slot update
df["Available"] = df["Available"].apply(lambda x: max(0, x + random.randint(-2,2)))

# Select location
location = st.selectbox("📍 Select Area", df["Location"])
selected = df[df["Location"] == location].iloc[0]

# Distance function
def distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)

# Calculate distance
df["Distance"] = df.apply(lambda row: distance(
    selected["lat"], selected["lon"],
    row["lat"], row["lon"]), axis=1)

# Best parking
available_df = df[df["Available"] > 0]
best = available_df.sort_values(by=["Distance","Price"]).iloc[0]

# UI cards
st.subheader("📍 Best Parking Spot")

col1, col2, col3 = st.columns(3)
col1.metric("Location", best["Location"])
col2.metric("Available Slots", int(best["Available"]))
col3.metric("Price/hr", f"₹{best['Price']}")

# 🏷️ Smart Insights (Fixed)
st.subheader("🏷️ Smart Insights")

nearby_area = df[df["Distance"] < 0.05]

if not nearby_area.empty:
    cheapest = nearby_area.loc[nearby_area["Price"].idxmin()]
else:
    cheapest = df.loc[df["Price"].idxmin()]

closest = df.loc[df["Distance"].idxmin()]

st.info(f"⭐ Recommended: {best['Location']}")
st.success(f"💰 Cheapest Nearby: {cheapest['Location']} (₹{cheapest['Price']})")
st.warning(f"⚡ Closest: {closest['Location']}")

# Recommendation logic
st.write("💡 System Recommendation:")
if best["Available"] < 3:
    st.warning("Better to choose another nearby option due to low availability")
else:
    st.success("This is the best option based on distance and price")

# Top 3 ranking
st.subheader("🏆 Top 3 Parking Options")

top3 = df.sort_values(by=["Distance","Price"]).head(3)

for i, row in top3.iterrows():
    st.write(f"{row['Location']} → {row['Available']} slots | ₹{row['Price']}")

# Map
st.subheader("🗺️ Parking Locations")
st.map(df[["lat", "lon"]])

# Table
st.subheader("💰 Price Comparison")
st.dataframe(df[["Location","Price","Available"]].sort_values(by="Price"))

# Graphs
st.subheader("📊 Availability Graph")
fig, ax = plt.subplots()
ax.plot(df["Location"], df["Available"])
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("💰 Price Chart")
fig2, ax2 = plt.subplots()
ax2.bar(df["Location"], df["Price"])
plt.xticks(rotation=45)
st.pyplot(fig2)

# Booking system
if st.button("🎟️ Book Slot"):
    booking_id = str(uuid.uuid4())[:8]
    time = datetime.now().strftime("%H:%M:%S")

    info = f"ID: {booking_id}\nLocation: {best['Location']}\nTime: {time}"

    qr = qrcode.make(info)
    qr.save("qr.png")

    st.success("✅ Booking Confirmed!")
    st.text(info)
    st.image("qr.png")

# ML prediction
st.subheader("🤖 Prediction")

time_data = np.array([[1],[2],[3],[4],[5],[6]])
slot_data = np.array([10,8,6,5,3,2])

model = LinearRegression()
model.fit(time_data, slot_data)

prediction = model.predict([[7]])

st.write(f"Next hour slots: {int(prediction[0])}")

if prediction[0] <= 3:
    st.error("⚠️ High demand expected!")
else:
    st.success("✅ Slots available")
