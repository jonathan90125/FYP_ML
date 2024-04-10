from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 定义XGBoost分类器
model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)

# 定义随机森林分类器
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 定义投票集成模型
voting_model = VotingClassifier(estimators=[('xgb',  model), ('rf', rf_model)], voting='soft')

# 在训练数据上训练投票集成模型
voting_model.fit(X_train, y_train)

# 在测试数据上进行预测
y_pred = voting_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
