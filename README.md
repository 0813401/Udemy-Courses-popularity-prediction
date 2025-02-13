# Udemy-Courses-popularity-prediction

## 一、 Problem Statement
* Online courses seems to be more popular in recent years, specifically due to the Covid-19.
* We study the data on Udemy courses dataset with data mining techniques, expect to **mining some rules** to helps people prepares their lecture in more efficient way.
* Moreover, **predict the rating level of a course** by feature, so people can understand how to create popular courses.


## 二、 Datasets description
🔗 [資料來源](https://www.kaggle.com/datasets/thedevastator/udemy-courses-revenue-generation-and-course-anal>)

| column name | type | example |
|-------|:-----:|------:|
| course_id   |  int  |   49798 |
| course_title   |  str  |   Bitcoin or How I Learned to Stop Worrying and Love Crypto |
| url   |  str  |   https://www.udemy.com/bitcoin-or-how-i-learned-to-stop-worrying-and-love-crypto/ |
| price   |  int  |   0 |
| num_subscribers   |  int  |   65576 |
| num_reviews   |  int  |   936 |  
| Num_lectures   |  int  |   24 |
| level   |  str  |   All levels |
| rating   |  float  |   0.56 |
| content_duration   |  float  |   8 |
| published_timestamp  |  str  |   2013-04-20T02:25:22Z |
| subject   |  str  |   subject |  
| Free/Paid   |  str  |   Free |  

## 三、Analysis workflow
### **Data cleaning**  
* Data may contain encoding error
![image](https://github.com/user-attachments/assets/8e5d7d11-2393-4f47-8d3c-f8144f96b4a4)
* Data may contain misalignment
![image](https://github.com/user-attachments/assets/5a21ade6-4b0a-4d4f-a079-1cd87f8383c2)
* Data may contain missing value    
So, we decide to remove encoding error data, and align data, and the missing data will be dropped out.

### **Data analysis**
First, we conducted an initial analysis based on the dataset, aiming to visualize certain data distributions to better understand its characteristics. Notably, we observed that the distribution of **new_reviews** and **num_subscribers** was overly **concentrated**. To address this, we applied **logarithmic transformation** and **quantile-based extraction** to derive new features. As shown in the visualization, the data distribution has become more balanced.
* Feature condense issue  
![image](https://github.com/user-attachments/assets/f69a055f-219f-4da6-a72d-7b3fe960a857)
* Log level extraction and quantile extraction
![image](https://github.com/user-attachments/assets/ff7d6390-e614-44dd-984f-6409b76cac7d)  
Another notable finding is that **subject has a significant influence on rating**, as demonstrated in the visualization. This aspect will also be discussed in the association rule mining analysis.
![image](https://github.com/user-attachments/assets/bcf304e9-7930-4b7b-becf-be8613c233f9)

### **Association Rule Mining**
* P(price = 0 | Paid, Business-Finance, rating = 3) = 10.9 / 18.5 = 58.9% (< 62.9%)
* The courses of Business-Finance with rating = 3 tends to be more expensive
* P(rating = 3 | Paid, Business-Finance, price = 0) = 10.9 / 17.6 = 61.9% (> 50.7%)
* The courses of Business-Finance with price = 0 tends to get higher rating
![image](https://github.com/user-attachments/assets/47ec4225-ee49-45cd-8e97-7389dac6707e)

* P(Business-Finance| All-levels, Paid, Rating = 3) = 10.2 / 25.2 = 40.4% (> 32.1%)
* Compare with the other subjects, the courses of Business-Finance get higher rating
![image](https://github.com/user-attachments/assets/bec08d83-4842-449c-891f-32cc6f2def0f)

### **Regression**
Next, we aim to predict the rating score using a regression approach. To achieve this, we designed an experiment based on our previous findings, splitting the data into a 70:30 train/test ratio. Initially, we employed **XGBoost** as the predictive model, anticipating that it would effectively forecast the  **rating score**.  
* Baselines - origin numeric raw feature
![image](https://github.com/user-attachments/assets/38ce18b7-384e-49c7-8ce5-d5ede0c06c39)
* Disc_one_hot_encoded model - using loglevel and discretecize with one hot encoding
![image](https://github.com/user-attachments/assets/aae1a4ee-58b7-4d68-912e-64f9428074eb)
* Taking best performance model prediction result into discretize categorical as a classification, and calculates the result by f1-score
![image](https://github.com/user-attachments/assets/99f8cf7f-0ac9-4a77-872d-6ac1f0ad337b)

### **Classification**
Building upon the conclusions drawn from our regression model, we developed three distinct models: **SVM**, **Random Forest**, and **XGBoost**. To enhance their performance, we refined each model by applying **RandomizedSearchCV** to identify the optimal parameters, which were then incorporated into the models. Finally, we computed the **rating score** to evaluate their predictive accuracy.
![image](https://github.com/user-attachments/assets/8bd3fc9d-dfb2-471d-8a28-da495617bd22)
Among the three models, the F1-score performance was suboptimal, likely due to the nature of the dataset. In terms of accuracy, SVM performed the worst, while Random Forest and XGBoost showed significant improvement. After comparing AUROC and accuracy, we concluded that the tuned Random Forest model demonstrated the best classification performance.
![image](https://github.com/user-attachments/assets/e7865391-a05e-4a11-b74c-e707bea8970f)
Next, we compared the confusion matrices of the three models. Since their F1-scores were relatively similar, no significant differences were observed in this aspect.

### **Conclusion**
* Some feature factors does effect the rating
* Discretize feature has better performance than originally value
* Classification method having better performance than regression
* Although we cannot precisely predict the rating score, we can still find out the trends
* Expect to help people to prepares the courses to having better result







