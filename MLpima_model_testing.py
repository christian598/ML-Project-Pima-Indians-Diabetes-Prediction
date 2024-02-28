

# Regression Analysis

# Predict Blood Glucose level based on the remaining 7 features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar


data = pd.read_csv("Updated.csv")
data.dtypes
if data.isnull().values.any():
    data.fillna(data.mean(), inplace=True)

data.describe()
X = data.drop(columns=["Glucose", "Outcome"])
y = data["Glucose"]

#Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R-squared: ", r2)

# Create a line plot  and Scatter plot of the actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual", linestyle='-', marker='o', alpha=0.8)
plt.plot(y_pred, label="Predicted", linestyle='-', marker='o', alpha=0.8)
plt.ylabel('Blood Glucose')
plt.xlabel('Sample Index')
plt.legend()
plt.title("Actual vs. Predicted Blood Glucose Levels")
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Blood Glucose")
plt.ylabel("Predicted Blood Glucose")
plt.title("Actual vs Predicted Blood Glucose")
min_value = min(np.min(y_test), np.min(y_pred))
max_value = max(np.max(y_test), np.max(y_pred))
plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', lw=2)



 # Generalization
 
degree = 2  # You can choose any degree depending on the complexity of the relationship
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X_scaled)
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)


mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("Polynomial Regression")
print("Mean Squared Error: ", mse_poly)
print("Mean Absolute Error: ", mae_poly)
print("R-squared: ", r2_poly)



# Plotting
plt.scatter(range(len(y_test)), y_test, color='blue', label="Actual")
plt.scatter(range(len(y_pred_poly)), y_pred_poly, color='red', label="Predicted")

# Set plot labels
plt.xlabel("Data Points")
plt.ylabel("Glucose Level")
plt.title("Actual vs Predicted Glucose Levels")
plt.legend()
plt.show()

lambdas = np.logspace(-4, 2, 200)
#lambdas = [0.001, 0.01, 0.1, 1, 10,17,20,40,60, 100]
mse_scores = []
kf = KFold(n_splits=10, random_state=42, shuffle=True)

for lamb in lambdas:
    ridge = Ridge(alpha=lamb)
    neg_mse = cross_val_score(ridge, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    mse_scores.append(-np.mean(neg_mse))
optimal_lambda = lambdas[np.argmin(mse_scores)]
print("Optimal λ value: ", optimal_lambda)

plt.figure(figsize=(10, 6))
plt.plot(lambdas, mse_scores, marker='o', linestyle='-', label="Mean Squared Error")
plt.xlabel("λ (Regularization Parameter)")
plt.ylabel("Mean Squared Error")
plt.xscale("log")
plt.legend()
plt.title("Generalization Error vs λ")
plt.show()


h_values = [1, 5, 10, 15, 20]
lambda_values = [0.01, 0.1, 1, 10, 100]
k1 = 10
k2 = 10
kf1 = KFold(n_splits=k1, shuffle=True, random_state=42)
kf2 = KFold(n_splits=k2, shuffle=True, random_state=42)

baseline_model = LinearRegression()
reg_model = Ridge()
ann_model = MLPRegressor()

table_data = []

for i, (train_index1, test_index1) in enumerate(kf1.split(X, y)):
    X_train1, X_test1 = X.iloc[train_index1], X.iloc[test_index1]
    y_train1, y_test1 = y.iloc[train_index1], y.iloc[test_index1]
    
    # Baseline model
    baseline_model.fit(X_train1, y_train1)
    y_pred_baseline = baseline_model.predict(X_test1)
    baseline_mse = mean_squared_error(y_test1, y_pred_baseline)
    baseline_row = ["Baseline", "-", "-", baseline_mse]
    table_data.append([i+1] + baseline_row)
    
    # Regularized linear regression model
    reg_best_params = None
    reg_best_score = np.inf
    for alpha in lambda_values:
        reg_model.set_params(alpha=alpha)
        reg_scores = []
        for train_index2, test_index2 in kf2.split(X_train1, y_train1):
            X_train2, X_test2 = X_train1.iloc[train_index2], X_train1.iloc[test_index2]
            y_train2, y_test2 = y_train1.iloc[train_index2], y_train1.iloc[test_index2]
            reg_model.fit(X_train2, y_train2)
            y_pred_reg = reg_model.predict(X_test2)
            reg_scores.append(mean_squared_error(y_test2, y_pred_reg))
        reg_mean_score = np.mean(reg_scores)
        if reg_mean_score < reg_best_score:
            reg_best_score = reg_mean_score
            reg_best_params = alpha
    reg_model.set_params(alpha=reg_best_params)
    reg_model.fit(X_train1, y_train1)
    y_pred_reg = reg_model.predict(X_test1)
    reg_mse = mean_squared_error(y_test1, y_pred_reg)
    reg_row = ["Regularized Linear Regression", "-", reg_best_params, reg_mse]
    table_data.append([i+1] + reg_row)
    
    # ANN model
    ann_best_params = None
    ann_best_score = np.inf
    for h in h_values:
        ann_model.set_params(hidden_layer_sizes=(h,))
        ann_scores = []
        for train_index2, test_index2 in kf2.split(X_train1, y_train1):
            X_train2, X_test2 = X_train1.iloc[train_index2], X_train1.iloc[test_index2]
            y_train2, y_test2 = y_train1.iloc[train_index2], y_train1.iloc[test_index2]
            ann_model.fit(X_train2, y_train2)
            y_pred_ann = ann_model.predict(X_test2)
            ann_scores.append(mean_squared_error(y_test2, y_pred_ann))
        ann_mean_score = np.mean(ann_scores)
        if ann_mean_score < ann_best_score:
            ann_best_score = ann_mean_score
            ann_best_params = h
    ann_model.set_params(hidden_layer_sizes=(ann_best_params,))
    ann_model.fit(X_train1, y_train1)
    y_pred_ann = ann_model.predict(X_test1)
    ann_mse = mean_squared_error(y_test1, y_pred_ann)
    ann_row = ["ANN", ann_best_params, "-", ann_mse]
    table_data.append([i+1] + ann_row)

table_df = pd.DataFrame(table_data, columns=["Fold", "Model", "h", "lambda", "Test MSE"])
print(table_df)



#  Classification

X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = data["Outcome"]
lambda_values = [0.01, 0.1, 1, 10, 100]
h_values = [1, 5, 10, 15, 20]

k1 = 10
k2 = 10
kf1 = KFold(n_splits=k1, shuffle=True, random_state=42)
kf2 = KFold(n_splits=k2, shuffle=True, random_state=42)

class BaselineModel:
    def fit(self, X, y):
        self.mode = y.mode().values[0]
    
    def predict(self, X):
        return np.full(X.shape[0], self.mode)
    
baseline_model = BaselineModel()
log_reg_model = LogisticRegression(penalty="l2")
ann_model = MLPClassifier()
baseline_scores = []
log_reg_scores = []
ann_scores = []


for i, (train_index1, test_index1) in enumerate(kf1.split(X, y)):
    X_train1, X_test1 = X.iloc[train_index1], X.iloc[test_index1]
    y_train1, y_test1 = y.iloc[train_index1], y.iloc[test_index1]
    
    # Baseline model
    baseline_model.fit(X_train1, y_train1)
    y_pred_baseline = baseline_model.predict(X_test1)
    baseline_acc = accuracy_score(y_test1, y_pred_baseline)
    baseline_scores.append(baseline_acc)
    
    # Logistic regression model
    log_reg_best_params = None
    log_reg_best_score = -np.inf
    for C in lambda_values:
        log_reg_model.set_params(C=C)
        log_reg_scores = []
        for train_index2, test_index2 in kf2.split(X_train1, y_train1):
            X_train2, X_test2 = X_train1.iloc[train_index2], X_train1.iloc[test_index2]
            y_train2, y_test2 = y_train1.iloc[train_index2], y_train1.iloc[test_index2]
            log_reg_model.fit(X_train2, y_train2)
            y_pred_log_reg = log_reg_model.predict(X_test2)
            log_reg_scores.append(accuracy_score(y_test2, y_pred_log_reg))
        log_reg_mean_score = np.mean(log_reg_scores)
        if log_reg_mean_score > log_reg_best_score:
            log_reg_best_score = log_reg_mean_score
            log_reg_best_params = C
    log_reg_model.set_params(C=log_reg_best_params)
    log_reg_model.fit(X_train1, y_train1)
    y_pred_log_reg = log_reg_model.predict(X_test1)
    log_reg_acc = accuracy_score(y_test1, y_pred_log_reg)
    log_reg_scores.append(log_reg_acc)
    
    # ANN model
    ann_best_params = None
    ann_best_score = -np.inf
    scaler = StandardScaler()
    X_train1_scaled = scaler.fit_transform(X_train1)
    X_test1_scaled = scaler.transform(X_test1)
    for h in h_values:
        ann_model.set_params(hidden_layer_sizes=(h,))
        ann_scores = []
        for train_index2, test_index2 in kf2.split(X_train1_scaled, y_train1):
            X_train2_scaled, X_test2_scaled = X_train1_scaled[train_index2], X_train1_scaled[test_index2]
            y_train2, y_test2 = y_train1.iloc[train_index2], y_train1.iloc[test_index2]
            ann_model.fit(X_train2_scaled, y_train2)
            y_pred_ann = ann_model.predict(X_test2_scaled)
            ann_scores.append(accuracy_score(y_test2, y_pred_ann))
        ann_mean_score = np.mean(ann_scores)
        if ann_mean_score > ann_best_score:
            ann_best_score = ann_mean_score
            ann_best_params = h
    ann_model.set_params(hidden_layer_sizes=(ann_best_params,))
    ann_model.fit(X_train1_scaled, y_train1)
    y_pred_ann = ann_model.predict(X_test1_scaled)
    ann_acc = accuracy_score(y_test1, y_pred_ann)
    ann_scores.append(ann_acc)
    
    print(f"Fold {i+1}: Baseline = {baseline_acc:.3f}, Log Reg = {log_reg_acc:.3f}, ANN = {ann_acc:.3f}")

ac = accuracy_score(y_test1, y_pred_ann)
ac1 = accuracy_score(y_test1, y_pred_log_reg)

cm = confusion_matrix(y_test1, y_pred_log_reg)
cm1 = confusion_matrix(y_test1, y_pred_ann)

plt.boxplot([baseline_scores, log_reg_scores, ann_scores])
plt.xticks([1, 2, 3], ["Baseline", "Logistic Regression", "ANN"])
plt.ylabel("Mean Accuracy")
plt.show()


table_data = []
for i, (train_index1, test_index1) in enumerate(kf1.split(X, y)):
    X_train1, X_test1 = X.iloc[train_index1], X.iloc[test_index1]
    y_train1, y_test1 = y.iloc[train_index1], y.iloc[test_index1]
    
    # Baseline model
    baseline_model.fit(X_train1, y_train1)
    y_pred_baseline = baseline_model.predict(X_test1)
    baseline_error = np.mean(y_pred_baseline != y_test1)
    baseline_row = ["Baseline", "-", "-", baseline_error]
    table_data.append([i+1] + baseline_row)
    
    # Logistic regression model
    log_reg_best_params = None
    log_reg_best_score = -np.inf
    for C in lambda_values:
        log_reg_model.set_params(C=C)
        log_reg_scores = []
        for train_index2, test_index2 in kf2.split(X_train1, y_train1):
            X_train2, X_test2 = X_train1.iloc[train_index2], X_train1.iloc[test_index2]
            y_train2, y_test2 = y_train1.iloc[train_index2], y_train1.iloc[test_index2]
            log_reg_model.fit(X_train2, y_train2)
            y_pred_log_reg = log_reg_model.predict(X_test2)
            log_reg_scores.append(np.mean(y_pred_log_reg != y_test2))
        log_reg_mean_score = np.mean(log_reg_scores)
        if log_reg_mean_score > log_reg_best_score:
            log_reg_best_score = log_reg_mean_score
            log_reg_best_params = C
    log_reg_model.set_params(C=log_reg_best_params)
    log_reg_model.fit(X_train1, y_train1)
    y_pred_log_reg = log_reg_model.predict(X_test1)
    log_reg_error = np.mean(y_pred_log_reg != y_test1)
    log_reg_row = ["Logistic Regression", "-", log_reg_best_params, log_reg_error]
    table_data.append([i+1] + log_reg_row)
    
    # ANN model
    # ANN model
ann_best_params = None
ann_best_score = -np.inf
scaler = StandardScaler()
X_train1_scaled = scaler.fit_transform(X_train1)
X_test1_scaled = scaler.transform(X_test1)
for h in h_values:
    ann_model.set_params(hidden_layer_sizes=(h,))
    ann_scores = []
    for train_index2, test_index2 in kf2.split(X_train1_scaled, y_train1):
        X_train2_scaled, X_test2_scaled = X_train1_scaled[train_index2], X_train1_scaled[test_index2]
        y_train2, y_test2 = y_train1.iloc[train_index2], y_train1.iloc[test_index2]
        ann_model.fit(X_train2_scaled, y_train2)
        y_pred_ann = ann_model.predict(X_test2_scaled)
        ann_scores.append(np.mean(y_pred_ann != y_test2))
    ann_mean_score = np.mean(ann_scores)
    if ann_mean_score > ann_best_score:
        ann_best_score = ann_mean_score
        ann_best_params = h
ann_model.set_params(hidden_layer_sizes=(ann_best_params,))
ann_model.fit(X_train1_scaled, y_train1)
y_pred_ann = ann_model.predict(X_test1_scaled)
ann_error = np.mean(y_pred_ann != y_test1)
ann_row = ["ANN", ann_best_params, "-", ann_error]
table_data.append([i+1] + ann_row)

    
print(f"Fold {i+1}: {baseline_row}, {log_reg_row}, {ann_row}")


# Baseline vs. Logistic Regression
baseline_log_reg_table = np.array([[np.sum(y_pred_baseline == y_pred_log_reg), np.sum((y_pred_baseline == y_test1) & (y_pred_log_reg != y_test1))],
                                   [np.sum((y_pred_baseline != y_test1) & (y_pred_log_reg == y_test1)), np.sum((y_pred_baseline != y_test1) & (y_pred_log_reg != y_test1))]])
result_bl_lr = mcnemar(baseline_log_reg_table, exact=True)

# Baseline vs. ANN
baseline_ann_table = np.array([[np.sum(y_pred_baseline == y_pred_ann), np.sum((y_pred_baseline == y_test1) & (y_pred_ann != y_test1))],
                               [np.sum((y_pred_baseline != y_test1) & (y_pred_ann == y_test1)), np.sum((y_pred_baseline != y_test1) & (y_pred_ann != y_test1))]])
result_bl_ann = mcnemar(baseline_ann_table, exact=True)

# Logistic Regression vs. ANN
log_reg_ann_table = np.array([[np.sum(y_pred_log_reg == y_pred_ann), np.sum((y_pred_log_reg == y_test1) & (y_pred_ann != y_test1))],
                              [np.sum((y_pred_log_reg != y_test1) & (y_pred_ann == y_test1)), np.sum((y_pred_log_reg != y_test1) & (y_pred_ann != y_test1))]])
result_lr_ann = mcnemar(log_reg_ann_table, exact=True)

print("Baseline vs. Logistic Regression:")
print(f"Statistical significance: p={result_bl_lr.pvalue:.4f}")
ci_low, ci_high = proportion_confint(np.sum(baseline_log_reg_table[0, 1] < baseline_log_reg_table[1, 0]), np.sum(baseline_log_reg_table))
print(f"Confidence interval: ({ci_low:.4f}, {ci_high:.4f})")
print()

print("Baseline vs. ANN:")
print(f"Statistical significance: p={result_bl_ann.pvalue:.4f}")
ci_low, ci_high = proportion_confint(np.sum(baseline_ann_table[0, 1] < baseline_ann_table[1, 0]), np.sum(baseline_ann_table))
print(f"Confidence interval: ({ci_low:.4f}, {ci_high:.4f})")
print()

print("Logistic Regression vs. ANN:")
print(f"Statistical significance: p={result_lr_ann.pvalue:.4f}")
ci_low, ci_high = proportion_confint(np.sum(log_reg_ann_table[0, 1] < log_reg_ann_table[1, 0]), np.sum(log_reg_ann_table))
print(f"Confidence interval: ({ci_low:.4f}, {ci_high:.4f})")
print()



## _______________________ ## ______________ ##

















