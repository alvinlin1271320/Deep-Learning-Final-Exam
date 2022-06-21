# **Deep Learing Final Exam**

**一、請利用flower_photos 這個資料集(Dataset)來訓練一個 Model。**
* 將Dataset 分 90% 是 Train 及 10% 是validation。
  
* 使用 ImageDataGenerator 來擴增圖片樣式並使用flow_from_directory來讀取 Dataset。

* 自行建立模型內容並訓練模型。

* 請使用 sunflowers_1.jpg 來預測此張圖片是屬於哪一種花類。

**二、請利用flower_photos 這個資料集(Dataset)來訓練一個 Model (使用VGG16)。**
* 將Dataset 分 90% 是 Train 及 10% 是validation。
  
* 使用 ImageDataGenerator 來擴增圖片樣式並使用flow_from_directory來讀取 Dataset。

* 請使用 VGG16來做 Transfer learning。第0 ~ 10 層的 weight 不用重新訓練，第 11 層以後的 weight 要重新訓練。

* 請使用 sunflowers_1.jpg 來預測此張圖片是屬於哪一種花類。

**三、簡單比較一下第一題及第二題的準確度及說明理由。**
