import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("diabetes.csv")
print(df.describe())
df1=df.copy()

# Check for number of " 0 "in dataset except 1st Attribute
print((df1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]== 0).sum())

# Replace " 0 " with NAN for data handling
df1.iloc[:, 1 : -1] = df1.iloc[:, 1 : -1].replace(0, np.nan)
print(df1.isna().sum())
print(df1.shape)

#Remove Rows With Missing Values
df1.dropna(inplace = True)
print(df1.shape)

# Number of Rows reduced from 768 to 392. Not a got technique
# Imputation of Missing Values
df2=df.copy()
df2.iloc[:, 1 : -1] = df2.iloc[:, 1 : -1].replace(0, np.nan)
df2.fillna(df2.mean(), inplace = True)
print(df2.isnull().sum())

# Determine Distriution
df2=df2.drop('Outcome',axis = 1)
df2.hist(figsize=(10,10))
plt.show()

# Standardization
y=df["Outcome"]
mean = df2.mean(axis=0)
std = df2.std(axis=0)
df2_s=pd.concat([mean,std],axis=1)
print(df2_s)

norm=(df2-mean)/std
df_norm = pd.concat([y,norm],axis=1)
print(df_norm)

# plot KDE plots for each attribute
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
axs = axs.flatten()
sns.set_style('whitegrid')
for i, col in enumerate(df_norm.columns[1:]):
    sns.kdeplot(df_norm[col], ax=axs[i], shade=True)
    axs[i].set_title(col)
fig.tight_layout()
plt.suptitle('Normalization of PIMA Indian Dataset', fontsize=16, y=1.05)
plt.show()

# Correlation - Higher Positive value indicates correlation
corr = df2.corr()
sns.heatmap(corr, annot=True)
plt.show()

# Outliers

sns.set(style="whitegrid")
fig, axs = plt.subplots(ncols=8, figsize=(16, 5))
for i, col in enumerate(df2.columns):
    sns.boxplot(x=df["Outcome"], y=col, data=df2, ax=axs[i])
plt.tight_layout()
plt.show()
median =df2.median(axis=0)
print(median)


q1_p= df['Pregnancies'].quantile(0.25)
q3_p = df['Pregnancies'].quantile(0.75)
iqr_p = q3_p - q1_p

# define the upper and lower bounds for outliers
lower_bound = q1_p - 1.5 * iqr_p
upper_bound = q3_p + 1.5 * iqr_p

# filter the pregnancies column to remove outliers
pregnancies_filtered = df2[(df2['Pregnancies'] >= lower_bound) & (df2['Pregnancies'] <= upper_bound)]['Pregnancies']

# replace the original 'pregnancies' column with the filtered values
df2.loc[(df2['Pregnancies'] < lower_bound) | (df2['Pregnancies'] > upper_bound), 'Pregnancies'] = pregnancies_filtered

q1_bmi = df2['BMI'].quantile(0.25)
q3_bmi = df2['BMI'].quantile(0.75)
iqr_bmi = q3_bmi - q1_bmi

# define the upper and lower bounds for BMI column outliers
lower_bound_bmi = q1_bmi - 1.5 * iqr_bmi
upper_bound_bmi = q3_bmi + 1.5 * iqr_bmi

# filter the BMI column to remove outliers
bmi_filtered = df2[(df2['BMI'] >= lower_bound_bmi) & (df2['BMI'] <= upper_bound_bmi)]['BMI']

# replace the original 'BMI' column with the filtered values
df2.loc[(df2['BMI'] < lower_bound_bmi) | (df2['BMI'] > upper_bound_bmi), 'BMI'] = bmi_filtered

q1_B = df['BloodPressure'].quantile(0.25)
q3_B = df['BloodPressure'].quantile(0.75)
iqr_B = q3_B- q1_B

# define the upper and lower bounds for outliers
lower_bound = q1_B - 1.8 * iqr_B
upper_bound = q3_B + 2.5 * iqr_B

# filter the BloodPressure column to remove outliers
BloodPressure_filtered = df2[(df2['BloodPressure'] >= lower_bound) & (df2['BloodPressure'] <= upper_bound)]['BloodPressure']

# replace the original 'BloodPressure' column with the filtered values
df2.loc[(df2['BloodPressure'] < lower_bound) | (df2['BloodPressure'] > upper_bound), 'BloodPressure'] = BloodPressure_filtered


print(df2['Pregnancies'].describe()) # Almost impossible higher value
print(df2['BMI'].describe()) # Almost impossible higher value
print(df2['BloodPressure'].describe()) #BP can't be zero
print(df2.describe())

##########################################
##########################################

#   PCA

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply PCA with all components
pca = PCA()
pca.fit(X)

# Plot scree plot
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
plt.title("Scree Plot")
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained Ratio")
plt.show()

# Plot cumulative variance plot
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Variance Plot")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained Ratio")
plt.show()

# Choose the number of components to include based on the scree plot and cumulative variance plot
n_components = 2

# Apply PCA with chosen number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Visualize the PCA components in 2D plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("PCA Plot ({} components)".format(n_components))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# Print the loadings of each component
print("Loadings:\n", pca.components_)

# Plot the principal directions
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_aspect('equal', 'box')
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for i, feature in enumerate(df.columns[:-1]):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', width=0.05, head_width=0.2)
    plt.text(pca.components_[0, i], pca.components_[1, i], feature, color='r', ha='center', va='center', fontsize=12)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Principal Directions")
plt.show()

print("Transformed dataset shape:", X_pca.shape)

# Plot the data projected onto the principal components
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Data Projected onto Principal Components")
plt.show()