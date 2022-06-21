# **Deep Learing Final Exam**  

## **一、請利用flower_photos 這個資料集(Dataset)來訓練一個 Model。**  
* 將Dataset 分 90% 是 Train 及 10% 是validation。  
  ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic1-1.png)  
  ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic1-2.png)
* 使用 ImageDataGenerator 來擴增圖片樣式並使用flow_from_directory來讀取 Dataset。  
  ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic1-3.png)
* 自行建立模型內容並訓練模型。  
  ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic1-4.png)
* 請使用 sunflowers_1.jpg 來預測此張圖片是屬於哪一種花類。  
  ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic1-5.png)  
## **二、請利用flower_photos 這個資料集(Dataset)來訓練一個 Model (使用VGG16)。**  
* 將Dataset 分 90% 是 Train 及 10% 是validation。  
   ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic2-1.png)  
   ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic2-2.png)
* 使用 ImageDataGenerator 來擴增圖片樣式並使用flow_from_directory來讀取 Dataset。  
   ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic2-3.png)  
* 請使用 VGG16來做 Transfer learning。第0 ~ 10 層的 weight 不用重新訓練，第 11 層以後的 weight 要重新訓練。  
   ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic2-4.png)  
* 請使用 sunflowers_1.jpg 來預測此張圖片是屬於哪一種花類。  
   ![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic2-6.png)  
## **三、簡單比較一下第一題及第二題的準確度及說明理由。**  
**自己建立model的準確度->83%**  
![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic1-6.png)  
**使用VGG16的準確度->30%**  
![image](https://github.com/ALVIN-SMITH/Deep-Learning-Final-Exam/blob/main/md_pic/pic2-5.png)  

    我發現自己建立的模型準確率比vgg16效果還要好，可能是因為訓練時vgg16參數沒有調整好，所以訓練效果不好，可能可以增加batchsize或是更換loss function來增加準確率。
