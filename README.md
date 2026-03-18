# 🛒 Market Basket Analysis — Grocery Dataset

## 📌 Project Overview

This project performs **Market Basket Analysis (MBA)** on a grocery dataset to uncover customer purchasing patterns. Using association rule mining techniques, we identify which items are frequently bought together and visualize insights through interactive and aesthetic plots.


## 🎯 Objectives

* Analyze customer transaction behavior
* Identify frequently purchased items
* Discover strong association rules between products
* Visualize trends, patterns, and relationships in data


## 🧰 Tech Stack

* **Python**
* **Pandas, NumPy** — Data manipulation
* **Matplotlib, Seaborn** — Data visualization
* **mlxtend** — Apriori & Association Rules
* **NetworkX** — Network graph visualization
* **WordCloud** — Text visualization


## 📂 Dataset Details

* **File:** `Groceries_dataset.csv`
* Contains:

  * `Member_number` → Customer ID
  * `Date` → Transaction date
  * `itemDescription` → Purchased item



## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

* Removed extra spaces & standardized text (lowercase)
* Converted date column to datetime format
* Grouped transactions by customer

### 2️⃣ Exploratory Data Analysis (EDA)

* Top purchased items
* Monthly transaction trends
* Basket size distribution

### 3️⃣ Market Basket Analysis

* Applied **Apriori Algorithm**
* Generated **Association Rules**
* Evaluated using:

  * **Support**
  * **Confidence**
  * **Lift**



## 📊 Visualizations

### 🔹 Top 20 Purchased Items

* Shows most frequently bought products

### 🔹 Monthly Trend Analysis

* Identifies peak and low transaction periods

### 🔹 Basket Size Analysis

* Histogram + Boxplot showing number of items per purchase

### 🔹 Association Rules (Top 20)

* Ranked using **Lift value**

### 🔹 Co-occurrence Heatmap

* Displays how often top items appear together

### 🔹 Network Graph

* Visual representation of item relationships
* Node size → number of connections
* Edge color → strength (lift)

### 🔹 Support vs Lift Scatter Plot

* Color-coded by confidence
* Helps identify strong and reliable rules


## 📈 Key Insights

* Certain items are frequently purchased together → useful for **cross-selling**
* Peak shopping months identified → helpful for **inventory planning**
* Most customers purchase a moderate number of items per visit
* High lift rules indicate strong product associations


## 💡 Applications

* Product recommendation systems
* Store layout optimization
* Targeted marketing & promotions
* Inventory management


## 📌 Future Improvements

* Deploy as **Streamlit Web App**
* Add real-time recommendations
* Integrate user segmentation (clustering)


## 🙌 Conclusion

This project demonstrates how data-driven insights can help businesses understand customer behavior and improve decision-making using **Market Basket Analysis**.



✨ *If you like this project, give it a ⭐ on GitHub!*

