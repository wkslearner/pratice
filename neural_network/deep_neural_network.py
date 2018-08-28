
import tensorflow as tf
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np

iris=load_iris()
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df['species']=iris.target

def data_generator():
    dataset = np.array(range(9))
    for i in dataset:
        yield i


#dataset = tf.data.Dataset.from_generator(data_generator, (tf.int32))

# dataset = tf.data.Dataset.from_tensor_slices([1,2,2,3,4,5,6,7,8,9])
# dataset=dataset.take(5)


#dataset=dataset.map(lambda x:x+1)

# dataset=dataset.batch(3)

dataset_a=tf.data.Dataset.from_tensor_slices([1,2,3])
dataset_b=tf.data.Dataset.from_tensor_slices([2,6,8])
zip_dataset=tf.data.Dataset.zip((dataset_a,dataset_b))

iterator = zip_dataset.make_initializable_iterator()

element = iterator.get_next()

with tf.Session() as sess:
   sess.run(iterator.initializer)
   for i in range(3):
       print(sess.run(element))



'''
x_train,x_test,y_train,y_test=train_test_split(iris_df.ix[:,0:4],iris_df['species'],
                                               test_size=0.2,random_state=1)



my_feature_columns = []
for key in x_train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

print(my_feature_columns)

clf=tf.estimator.DNNClassifier(hidden_units=[10,20,10],
                               feature_columns=my_feature_columns,
                               n_classes=3,
                               activation_fn=tf.nn.relu)


def train_input_fn(features, labels, batch_size):
    """为了训练写的输入函数"""
    # 转变输入的数据为Dataset的对象
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    print(dataset)
    # 随机拖拽，重复和批生成数据。（其实就是搅拌样本使随机化，并随机组合生成一批样本）
    #dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


input_fn=train_input_fn(x_train,y_train,100)
print(input_fn)




clf.train(input_fn,steps=1000)
acc=clf.evaluate(x_test,y_test,100)['accuracy']
print(acc)
'''