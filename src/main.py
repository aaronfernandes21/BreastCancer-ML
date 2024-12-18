import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\user\OneDrive\Desktop\BreastCancerMLProject\data\data.csv')

df = df.drop(columns=['Unnamed: 32', 'id'])


df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


print("Missing values check:")
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

X = df.drop('diagnosis', axis=1) 
y = df['diagnosis']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)


rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)


knn_model = KNeighborsClassifier(n_neighbors=7)  
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"SVM Accuracy: {accuracy_svm}")
print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"KNN Accuracy: {accuracy_knn}")


print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_knn = confusion_matrix(y_test, y_pred_knn)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('SVM Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_title('KNN Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.show()


models = ['SVM', 'Random Forest', 'KNN']
accuracies = [accuracy_svm, accuracy_rf, accuracy_knn]

plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Models')
plt.ylim(0.9, 1) 
plt.show()

feature_importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance from Random Forest')
plt.show()

cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"\nRandom Forest Cross-Validation Accuracy: {cv_scores_rf.mean()} ± {cv_scores_rf.std()}")

cv_scores_svm = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
print(f"SVM Cross-Validation Accuracy: {cv_scores_svm.mean()} ± {cv_scores_svm.std()}")

cv_scores_knn = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)
print(f"KNN Cross-Validation Accuracy: {cv_scores_knn.mean()} ± {cv_scores_knn.std()}")
