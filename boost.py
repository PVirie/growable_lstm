import tensorflow as tf
import cell as Cells
import numpy as np


# the loss function is the cross-entropy
class Classification_Gradient_Boost:

    def __init__(self, RNN, boost_steps):
        print("init")

        self.inputs = tf.placeholder(shape=[None, None, RNN.get_depth()], dtype=tf.float32)
        self.targets = tf.placeholder(shape=[None], dtype=tf.int32)

        total_classes = RNN.get_num_outputs()
        one_hot_targets = tf.one_hot(self.targets, total_classes, dtype=tf.float32)

        self.gammas = []
        for i in range(boost_steps):
            self.gammas.append(tf.Variable(np.ones([1]), dtype=tf.float32))
        all_gammas = tf.stack(self.gammas, axis=0)

        self.training_ops = []

        batches = tf.shape(self.inputs)[0]
        for i in range(boost_steps):
            RNN.grow()
            cell = RNN.get_last_cell()
            outputs, _ = tf.nn.dynamic_rnn(RNN, self.inputs, initial_state=RNN.init_state(batches), time_major=False)
            outputs = outputs[:, -1, :, :]

            age = i + 1
            weighted_outputs = tf.reduce_sum(tf.reshape(all_gammas[0:age], [1, age, 1]) * outputs, axis=1)
            weighted_outputs = tf.reshape(weighted_outputs, [batches, total_classes])

            cost = tf.losses.softmax_cross_entropy(one_hot_targets, weighted_outputs)
            training_op_model = (tf.train.AdamOptimizer(0.001).minimize(cost, var_list=cell.get_variable_scope()), cost)

            current_gamma = self.gammas[i]
            training_op_weight = (tf.train.AdamOptimizer(0.0001).minimize(cost, var_list=[current_gamma]), cost)

            prediction = tf.nn.softmax(weighted_outputs, axis=-1)
            self.training_ops.append((training_op_model, training_op_weight))

        self.confidence = prediction
        self.prediction = tf.argmax(prediction, axis=-1)

    def train(self, data, target, sess=None):
        print("train")

        for i in range(len(self.training_ops)):
            for j in range(1000):
                _, c = sess.run(self.training_ops[i][0], feed_dict={self.inputs: data, self.targets: target})
                print(i, j, c)
            for j in range(1000):
                _, c = sess.run(self.training_ops[i][1], feed_dict={self.inputs: data, self.targets: target})
                print(i, j, c)

    def predict(self, data, sess=None):
        print("predict")
        p, c = sess.run((self.prediction, self.confidence), feed_dict={self.inputs: data})

        return p, c


if __name__ == "__main__":
    with tf.Session() as sess:
        rnn = Cells.Aggregated_Cell([2, 5])
        booster = Classification_Gradient_Boost(rnn, 8)

        sess.run(tf.global_variables_initializer())

        inputs = np.random.rand(20, 3, 2)
        outputs = np.random.randint(0, 5, [20])

        booster.train(inputs, outputs, sess)
        p, c = booster.predict(inputs, sess)
        print(outputs)
        print(p)
