import joblib
import pandas as pd

# 載入模型和預處理器
model = joblib.load('models/logistic_regression_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

print("--- 檢查模型內部 ---")

# 從預處理器中獲取特徵名稱
# 數值特徵名稱
num_features = preprocessor.transformers_[0][2]

# 分類特徵經過 OneHotEncoder 後的名稱
cat_features_original = preprocessor.transformers_[1][2]
cat_features_encoded = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features_original)

# 合併所有特徵名稱
all_feature_names = list(num_features) + list(cat_features_encoded)

# 獲取模型的係數
# model.coef_ 是一個二維陣列 [[...]]，我們只需要第一個元素
coefficients = model.coef_[0]

# 建立一個 DataFrame 來清晰地展示特徵和其對應的係數
feature_importance = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': coefficients
})

# 根據係數的絕對值進行排序，來看看哪些特徵影響最大
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

# 移除輔助排序的欄位
del feature_importance['Abs_Coefficient']

print("\n邏輯斯迴歸模型的係數 (按影響力排序):")
print("(正係數表示增加壞帳風險，負係數表示降低壞帳風險)")
print(feature_importance.to_string())

print("\n\n--- 模型截距 (Intercept) ---")
print(model.intercept_[0])
