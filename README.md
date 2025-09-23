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
import matplotlib.pyplot as plt
df=pd.read_csv("C:\\Users\\admin\\Downloads\\data.csv")
df
 ```
<img width="686" height="342" alt="image" src="https://github.com/user-attachments/assets/c29e504b-592f-4a7d-a11b-47208afa0159" />

```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder
df1=df.copy()
education=["High School","Diploma","Bachelors","Masters","PhD"]
enc=OrdinalEncoder(categories=[education])
enc.fit_transform(df1[['Ord_2']])
```
<img width="470" height="191" alt="image" src="https://github.com/user-attachments/assets/5e6a1f68-67ec-4a31-b85f-4c69e4331e1d" />
```
df1['ordinal_encoder']=enc.fit_transform(df1[['Ord_2']])
df1
```
<img width="636" height="314" alt="image" src="https://github.com/user-attachments/assets/af82683a-8b68-4bdf-87e8-2c56f62d2cfa" />

```
df2=df.copy()
enc=LabelEncoder()
enc.fit_transform(df2[['Ord_2']])
df2['Labelencoder']=enc.fit_transform(df1[['Ord_2']])
df2
```
<img width="633" height="313" alt="image" src="https://github.com/user-attachments/assets/0d5c8d41-9b9b-4c20-9d3e-804d616c951e" />

```
df3=df.copy()
enc=OneHotEncoder()
newdata=pd.DataFrame(enc.fit_transform(df3[['City']]))
df4=pd.concat([df3,newdata],axis=1)

df4
```
<img width="790" height="312" alt="image" src="https://github.com/user-attachments/assets/dfd03718-271a-43eb-b919-c6dda5509712" />

```
pd.get_dummies(df4,columns=['City'])
```
<img width="851" height="327" alt="image" src="https://github.com/user-attachments/assets/5e2c343f-e469-4b14-a026-601e16c97a0a" />

```
from category_encoders import TargetEncoder
df5=df.copy()
enc=BinaryEncoder()
newdata=pd.DataFrame(enc.fit_transform(df5[['Ord_1']]))
df6=pd.concat([df5,newdata],axis=1)
df6
```
<img width="746" height="321" alt="image" src="https://github.com/user-attachments/assets/969e521d-a9e1-4016-ac66-c75cc49e0405" />

```
df7=df.copy()
enc=TargetEncoder()
newdata=pd.DataFrame(enc.fit_transform(df7[['Ord_1']],df7['Target']))
df6=pd.concat([df5,newdata],axis=1)
df6
```

<img width="834" height="321" alt="image" src="https://github.com/user-attachments/assets/7136654b-d54e-4e79-b1f6-60da84a65ab5" />

```
import pandas as pd
df=pd.read_csv("C:\\Users\\admin\\Downloads\\Data_to_Transform(1).csv")
df
```
<img width="855" height="365" alt="image" src="https://github.com/user-attachments/assets/400facb0-1310-48b8-b128-b5310b209f1e" />

```
df.skew()
```
<img width="537" height="122" alt="image" src="https://github.com/user-attachments/assets/8d061cf9-602e-4209-ae50-020df897d88d" />

```
sm.qqplot(df["Moderate Positive Skew"], line="45")
plt.show()
```

<img width="716" height="460" alt="image" src="https://github.com/user-attachments/assets/682ba88e-aac7-4858-9d61-e808813404ae" />

```
sm.qqplot(df["Moderate Positive Skew"], line="45")
plt.show()
```
<img width="800" height="440" alt="image" src="https://github.com/user-attachments/assets/5eea8438-4eea-4281-965b-e51b960cd284" />

```
sm.qqplot(df["Moderate Negative Skew"], line="45")
plt.show()
```
<img width="715" height="447" alt="image" src="https://github.com/user-attachments/assets/0bb681dd-f425-4351-b03f-ed9bf3d8681d" />

```
import statsmodels.api as sm
import matplotlib.pyplot as plt  

sm.qqplot(df["Moderate Negative Skew"], line="45")

plt.show()

```
<img width="873" height="437" alt="image" src="https://github.com/user-attachments/assets/8b743a14-46de-4bcd-8dcb-a00f78032f58" />

```
sm.qqplot(df["Moderate Positive Skew"], line="45")
plt.show()
```
<img width="889" height="357" alt="image" src="https://github.com/user-attachments/assets/2dd086f5-0463-4426-9da3-a237f883f28a" />

```
sm.qqplot(df["Highly Positive Skew"], line="45")
plt.show()
```
<img width="888" height="464" alt="image" src="https://github.com/user-attachments/assets/6c0272da-6c2d-4df8-ac0c-de00a1ccdda0" />

```
sm.qqplot(df["Highly Negative Skew"], line="45")
plt.show()
```
<img width="790" height="438" alt="image" src="https://github.com/user-attachments/assets/574d5393-5c95-4d76-99eb-8838b3863663" />

```
df1=df.copy()
df1['log transformation']=np.log(df["Moderate Positive Skew"])
df1
```
<img width="877" height="415" alt="image" src="https://github.com/user-attachments/assets/8676cbb2-f50e-4c87-9bfe-8ecf37e3be75" />

```
import numpy as np
sm.qqplot(df1['log transformation'],line="45")
plt.show()
```
<img width="847" height="436" alt="image" src="https://github.com/user-attachments/assets/99b8c151-6a0b-486b-98da-3ba530908f4b" />

```
df2=df.copy()
df2['log transformation']=np.sqrt(df2["Moderate Positive Skew"])
df2

```
<img width="895" height="495" alt="image" src="https://github.com/user-attachments/assets/f7f261a5-dc03-4faa-a1bb-f310de626a8f" />

```
df2=df.copy()
df2['sqrt transformation']=np.sqrt(df2["Highly Positive Skew"])
df2
sm.qqplot(df2['sqrt transformation'],line="45")
plt.show()
```
<img width="930" height="446" alt="image" src="https://github.com/user-attachments/assets/ef5f7d9a-57ea-43c3-a41d-087e37fb5422" />

```
df3=df.copy()
df3['square transformation']=np.square(df3["Moderate Negative Skew"])
df3
sm.qqplot(df3['square transformation'],line="45")
plt.show()
```
<img width="871" height="471" alt="image" src="https://github.com/user-attachments/assets/5df7fa72-d465-4ce4-b458-1141d7756560" />

```

df5 = df.copy()
df5['reciprocal transformation'] = 1 / (df5["Moderate Positive Skew"])
df5


sm.qqplot(df5['reciprocal transformation'], line="45")
plt.show()
```
<img width="888" height="469" alt="image" src="https://github.com/user-attachments/assets/e0964835-2072-471b-a498-b24e649487ca" />


```
df5=df.copy()
df5['boxcox transformation'],p=stats.boxcox(df5["Moderate Positive Skew"])
df5
sm.qqplot(df5['boxcox transformation'],line="45")
plt.show()
```
<img width="847" height="448" alt="image" src="https://github.com/user-attachments/assets/eeca9313-5ed5-4547-8700-84aca1d45f07" />

```
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(0) 
data = np.random.lognormal(mean=0, sigma=1, size=100)

df = pd.DataFrame({'Moderate Skew': data - 2}) 

df6 = df.copy()
df6['yeojohnson transformation'], p = stats.yeojohnson(df6["Moderate Skew"])

sm.qqplot(df6['yeojohnson transformation'], line="45")
plt.show()
```
<img width="834" height="480" alt="image" src="https://github.com/user-attachments/assets/4fdc3d52-c011-480a-a869-b40d2db886ca" />

```
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

df = pd.DataFrame({
    'Skewed Data': np.random.lognormal(mean=1, sigma=0.5, size=100)
})


df7 = df.copy()

qt = QuantileTransformer(output_distribution='normal', random_state=0)

df7['quantile transformation'] = qt.fit_transform(df7[['Skewed Data']])

sm.qqplot(df7['quantile transformation'], line="45")

plt.show()


```

<img width="825" height="465" alt="image" src="https://github.com/user-attachments/assets/250a9f27-ff68-46ef-b166-03a9e0763a6d" />


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.



       
