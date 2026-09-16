# 🍳 Predicting High Traffic Recipes: Machine Learning Project

Predicting the probability that a recipe will drive high traffic if featured on a recipe website's homepage, using nutrition and category data.

---

## 📌 Problem Statement

A recipe site's homepage can only feature a handful of recipes at a time, and homepage traffic is what drives subscriptions. When the wrong recipe gets featured, it costs the business:

- A wasted homepage slot that could have gone to a recipe that actually performs
- Lower overall homepage traffic than the site could otherwise be getting
- No reliable way to plan which recipes to push each week
- Guesswork replacing a process that should be testable

The product manager set one clear bar for a model to be worth using: at least 80% precision.

## 📊 The Data

947 recipes across 8 fields: an id, four nutrition columns (calories, carbohydrate, sugar, protein), category, servings, and a high_traffic label.

## 🧹 Data Cleaning

Cleaning it turned up more than expected:

- 52 recipes were missing all four nutrition values at once, which ruled out random data entry gaps and pointed to something upstream that never loaded that block of values. Each was filled with the median for its own category rather than a dataset wide median, since nutrition varies a lot by recipe type.
- Category had 11 unique values despite documentation listing 10, because "Chicken Breast" had been entered as its own category instead of folding into "Chicken."
- The high_traffic column only ever recorded "High" or left the cell blank. Blank meant not high traffic rather than missing data, an easy trap if you don't catch it, so it was filled as "Low" and converted into a binary target.
- After cleaning: 947 rows, 7 columns, zero missing values, and a target split of about 61% high traffic to 39% not.

## 🔍 Exploratory Data Analysis

Category carried most of the signal well before any model got trained:

- Vegetable recipes went high traffic 98.8% of the time
- Potato hit 94.3%, Pork hit 91.7%
- Beverages barely cracked 5.4%, the lowest of any category

The nutrition numbers told a quieter story. High traffic recipes ran slightly higher on average in calories, carbs, and protein, but the gap was modest next to the category split above.

## ⚙️ Approach

Two models on the same pipeline: numeric features standardized, category one hot encoded, 80/20 stratified train test split.

- Logistic Regression as the baseline
- Random Forest (300 trees, max depth 6) as the comparison model

## 📈 Results

On the holdout test set:

- Logistic Regression: 0.779 accuracy, 0.848 precision, 0.774 recall, 0.809 F1
- Random Forest: 0.758 accuracy, 0.800 precision, 0.800 recall, 0.800 F1

5 fold cross validation, a more reliable estimate than a single split:

- Logistic Regression: 0.800 precision on average (± 0.043)
- Random Forest: 0.740 precision on average (± 0.030)
- ROC AUC: 0.87 for Logistic Regression, 0.85 for Random Forest

Logistic Regression is the recommendation. It's the only one that clears the 0.80 precision bar on a reliable multi-fold estimate, it's far simpler to explain to a non technical product team, and it wins on the metric the business actually asked for. Random Forest catches more of the true high traffic recipes, higher recall, but that comes from making more false positive calls, the expensive mistake given the original ask.

## 🔑 What Drives the Prediction

[Feature importance]

Category is doing most of the work, more than the nutrition numbers. Beverages is the single strongest feature in the entire model, ahead of every nutrition column, and it points toward low traffic. Protein content is the strongest numeric signal on its own.

[Traffic share by category]

## 💡 Recommendation

- Feature Vegetable, Potato, Pork, and Meat recipes when the goal is maximizing homepage traffic
- Deprioritize Beverages recipes for high visibility placement unless there's a specific reason to test that category anyway
- Use the model's predicted probability instead of category alone, since nutrition profile still shifts the odds within a category

## 🗂️ Repo Structure

├── notebook.ipynb    full analysis: cleaning, EDA, modeling
├── images/            exported charts used in this README
└── README.md

## 🛠️ Tools

Python, pandas, scikit-learn, seaborn, matplotlib.
