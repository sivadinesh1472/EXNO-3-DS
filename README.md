## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
 ```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
 ```
<img width="483" height="388" alt="Screenshot 2026-03-14 100747" src="https://github.com/user-attachments/assets/8bccf01a-d7d6-4f4b-8881-73d87cd0211e" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="215" height="190" alt="image" src="https://github.com/user-attachments/assets/03fe6fb2-f352-4982-bde1-c68e446de9e9" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="497" height="397" alt="image" src="https://github.com/user-attachments/assets/aa28d166-760c-456d-b232-b06b9c3e84a3" />


```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="407" height="380" alt="image" src="https://github.com/user-attachments/assets/4af2fbdc-714b-4615-985c-0deca37e7293" />


```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

one = OneHotEncoder(sparse_output=False)   # <-- use sparse_output
df2 = df.copy()
enc = pd.DataFrame(one.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
```
<img width="532" height="391" alt="image" src="https://github.com/user-attachments/assets/207dc071-c447-4e4f-8c16-a2963927219f" />


```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="720" height="388" alt="image" src="https://github.com/user-attachments/assets/59e85ffd-2c31-4d12-bd9f-8eb255556a89" />


```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="542" height="390" alt="image" src="https://github.com/user-attachments/assets/210230dc-1e0d-47c8-a5f4-c76a582e5753" />


```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="609" height="376" alt="image" src="https://github.com/user-attachments/assets/e656b017-bcd0-422d-91c3-9e36e7f2ad86" />


```
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="830" height="404" alt="image" src="https://github.com/user-attachments/assets/bf119995-7db1-4b1d-aaea-7e4022bb3563" />



```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="703" height="397" alt="image" src="https://github.com/user-attachments/assets/0dfb0f2e-3830-4dae-882e-15551003f6f4" />


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="856" height="457" alt="image" src="https://github.com/user-attachments/assets/6585058c-a96a-49ab-a1d5-c232ddeffee1" />


```
df.skew()
```
<img width="438" height="104" alt="image" src="https://github.com/user-attachments/assets/67836f46-d08d-4584-a076-ac0b21d96217" />


```
np.log(df["Highly Positive Skew"])
```
<img width="561" height="236" alt="image" src="https://github.com/user-attachments/assets/3c3b1a63-b8bb-45a9-b8ff-7860b1d7415b" />


```
np.sqrt(df["Highly Positive Skew"])

```
<img width="596" height="244" alt="image" src="https://github.com/user-attachments/assets/f08daa53-a366-4345-b4ba-e874807600b6" />


```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="600" height="220" alt="image" src="https://github.com/user-attachments/assets/0f6b2dc4-4341-443d-8bd6-80e0bd032d88" />


```
np.square(df["Highly Positive Skew"])
```
<img width="577" height="230" alt="image" src="https://github.com/user-attachments/assets/fe654ab3-f120-4547-a61f-e5555c766c3c" />


```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="844" height="438" alt="image" src="https://github.com/user-attachments/assets/7cb8ae9a-c265-4727-9146-9397059b9c06" />


```
df.skew()
```
<img width="454" height="126" alt="image" src="https://github.com/user-attachments/assets/b3a38ee2-5593-4d8e-952e-06ab0a3d0203" />


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="476" height="121" alt="image" src="https://github.com/user-attachments/assets/b7c7da37-c0b3-4af9-a2c4-42bef323d171" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

```
<img width="838" height="514" alt="image" src="https://github.com/user-attachments/assets/55defbdc-0678-4c63-b51e-afb5ba01836c" />


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="757" height="489" alt="image" src="https://github.com/user-attachments/assets/9cc42e32-13ff-4206-a141-5cb0482effee" />


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="688" height="469" alt="image" src="https://github.com/user-attachments/assets/d5be3233-0535-467f-95a7-fc0c73b55b4f" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="774" height="507" alt="image" src="https://github.com/user-attachments/assets/50541977-35c3-4977-b601-ddea3e4685ed" />



```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="700" height="492" alt="image" src="https://github.com/user-attachments/assets/1e1ab163-0a10-4f24-8a27-9d6a84b2b267" />


```
dt=pd.read_csv("data.csv")
dt
```
<img width="613" height="425" alt="image" src="https://github.com/user-attachments/assets/221a268b-4136-4c32-a664-04e195b06497" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Ord_1"]=qt.fit_transform(dt[["Target"]])
sm.qqplot(dt['Target'],line='45')
plt.show()


```
<img width="764" height="489" alt="image" src="https://github.com/user-attachments/assets/0b532937-52bb-40a1-a134-01fce0bb6104" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```


<img width="756" height="500" alt="image" src="https://github.com/user-attachments/assets/94330df4-2ca5-4f4e-a72e-66c667e07547" />


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.



       
