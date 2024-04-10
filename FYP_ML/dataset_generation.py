import numpy as np
from sklearn.datasets import make_regression

#design several effective parameters first(linear and onehot)

import numpy as np
array_length=1000
#有效特征
#生产量
array1 = np.random.rand(array_length)*8000+2000
#发货量
array2 = array1*(np.random.rand(array_length) * 0.5 + 0.5)
#总销量
array3 = array2*(np.random.rand(array_length) * 0.8 + 0.2)
#是否处于电商节日
array4 = np.random.randint(2, size=array_length)
#季节
array5 = np.random.randint(4, size=array_length)
#产品价格
array6 = np.random.rand(array_length)*500+500
#促销力度（5个级别）
array7 = np.random.randint(5, size=array_length)
#好评率
array8 = np.random.rand(array_length) * 0.5 + 0.5
#能效等级（5个级别）
array9 = np.random.randint(5, size=array_length)
#是否家电下乡
array10 = np.random.randint(2, size=array_length)

#随机特征
#暂未生成

# 合并所有feature
X = np.column_stack((array1, array2, array3, array4, array5, array6, array7, array8, array9, array10))

# 生成y值
y=np.sum([array1 / 100,
   array2 / 100,
   array3 / 100,
   array4 * 200,
   array5 * 500,
   -array6 / 5,
   array7 * 50,
   array8 * 200,
   array9 * 50,
   array10 * 100
   ],axis=0)
y=y*(np.random.rand(array_length) * 0.1 + 0.9)

# 计算每个特征的皮尔逊相关系数
correlation_coefficients = np.corrcoef(np.concatenate((X, y.reshape(-1, 1)), axis=1), rowvar=False)

# 打印每个特征与目标变量的皮尔逊相关系数
for i, col in enumerate(correlation_coefficients[:-1]):
    print(f"Feature {i+1} and Target Pearson correlation coefficient: {col[-1]}")



np.savetxt("X.csv", X, delimiter=',')
np.savetxt("y.csv", y, delimiter=',')
