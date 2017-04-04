import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

df = pd.read_csv('./datasets/UCIrvineCrimeData.csv');
df = df.replace('?',np.NAN)
features = [x for x in df.columns if x not in ['fold', 'state', 'community', 'communityname', 'county'
                                               ,'ViolentCrimesPerPop']]

df.dropna()
df.dropna(axis=1);
df.dropna(how='all');
df.dropna(thresh=4);
df.dropna(subset=['community']);
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df[features])
imputed_data = imr.transform(df[features]);
X, y = imputed_data, df['ViolentCrimesPerPop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0);

y_train = [[int(j == i*100) for j in range(100)] for i in y_train]
y_test = [[int(j == i*100) for j in range(100)] for i in y_test]

train_data = np.array(X_train)
train_label = np.array(y_train)
test_data = np.array(X_test)
test_lable = np.array(y_test)

learning_rate = 0.001
training_epochs = 3000
batch_size = 100
display_step = 100

n_hidden_1 = 110
n_hidden_2 = 105
n_input = 122
n_classes = 100

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def neural_network(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = neural_network(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(training_epochs):
#         _, c = sess.run([optimizer, cost], feed_dict={x: train_data,y: train_label})
#
#         if epoch % display_step == 0:
#             print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
#
#     print("Optimization Finished!")
#
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
#     print("Accuracy:", accuracy.eval({x: test_data, y: test_lable}))


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x: np.concatenate((train_data,test_data),axis = 0 ),y: np.concatenate((train_label,test_lable),axis = 0 )})

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy:", accuracy.eval({x: test_data, y: test_lable}))