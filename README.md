这是一个电商广告推荐系统中的商品分类的机器学习项目

### Install depencies
```
pip3 install -r requirements.txt
```


### Model Training Process
- Create training set with advertiser_id, name, description, product_type, category_id column
  
- Sample from the training set to balance the category_id distribution

- Traing the model with name, description, product_type as feature and category_id as target with the 
   sklearn grid search to tuing the hyper-parameter

- Training the model with the selected hyper-parameter using PMML pipeline to create PMML
  



### Model Accuracy Score

For 200 samples.

| category     | accuracy score |
| ------------ | -------------- |
| Major        | 98.0           |
| Shoes        | 99.5           |
| Clothing     | 97.0           |
| Handbags     | 98.0           |
| Jewellery    | 99.5           |
| Accessories  | 99.0           |
| Begin_beauty | 98.5           |

