import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 导入csv数据
Corpus = pd.read_csv(r"C:\Users\钟顺民\Desktop\macys\test02.csv")

# 分配数据
Train_X, Test_X, Train_Y, Test_Y = model_selection \
    .train_test_split(Corpus['text_final'].values, Corpus['category_id'].values, test_size=0.2)

# print(len(Train_X))
# 标签对目标变量进行编码-这样做是为了将数据集中的字符串类型的分类数据转换为模型可以理解的数值。
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(Train_X)
X_test_counts = count_vect.fit_transform(Test_X)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)


def get_matrix(f):
    dt = pd.DataFrame(f)
    # 将数据放入矩阵中
    train = np.zeros(f.shape)
    for i in range(0, len(dt) - 1):
        for x in dt[0]:
            a = str(x).split("\n")
            for b in a:
                c = b.split("\t")
                num = c[1]
                try:
                    local = c[0].replace("(", "").replace(")", "").replace(" ", "").split(",")[1]
                    train[i][local] = num
                except:
                    continue
    return train


X_train = get_matrix(X_train_tfidf)
X_test = get_matrix(X_test_tfidf)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Train_Y, epochs=10)

model.evaluate(X_test, Test_Y, verbose=2)

"""
神经网络进行商品分类：
放入神经网络训练的数据必须是矩阵或者是向量，不能是带有不同秩的向量组成的list，而根据name和description字段通过TF-IDF操作后再清洗的数据刚好是上述类型的list，不能用作训练数据。
可行的是，将每个分类类型的特征放入一个向量中，而描述中出现了这个词汇（词频），则记为1（1次），没出现记为0，然后组成一个0、1、n向量，把向量组成矩阵放入网络中训练。
因此，现在需要拿到对应分类的词汇组成

Neural network for goods Classification

The data put into the neural network training must be a matrix or a vector, and cannot be a list of vectors with 
different ranks. According to the name and description fields, the data cleaned after TF-IDF operation is just the 
above type of list, and cannot be used as training data.

It is feasible to put the features of each classification type into a vector, and if the word (word frequency) 
appears in the description, it will be recorded as 1 (once), and if it does not appear, it will be recorded as 0, 
and then form a 0, 1 or n vector, and put the vector composition matrix into the network for training.

Therefore, we need to get the word composition of the corresponding classification


"""
