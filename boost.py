import tensorflow as tf
import cell as Cells
import numpy as np


# the loss function is the cross-entropy
class Classification_Gradient_Boost:

    def __init__(self, RNN, boost_steps, init_rate=0.01, rate_multiplier=1.0, gradient_learning_rate=0.001, lambda_learning_rate=0.01):
        print("init")

        self.total_classes = RNN.get_num_outputs()
        self.boost_steps = boost_steps
        self.RNN = RNN

        self.init_rate = init_rate
        self.rate_multiplier = rate_multiplier
        self.gradient_learning_rate = gradient_learning_rate
        self.lambda_learning_rate = lambda_learning_rate

        self.gammas = []
        for i in range(self.boost_steps):
            self.gammas.append(tf.Variable(np.ones([1]) * self.init_rate, dtype=tf.float32))

    def build_graph(self, inputs, targets):

        one_hot_targets = tf.one_hot(targets, self.total_classes, dtype=tf.float32)
        all_gammas = tf.stack(self.gammas, axis=0)
        training_ops = []
        residues = one_hot_targets
        batches = tf.shape(inputs)[0]
        for i in range(self.boost_steps):
            self.RNN.grow()
            cell = self.RNN.get_last_cell()

            outputs, _ = tf.nn.dynamic_rnn(self.RNN, inputs, initial_state=self.RNN.init_state(batches), time_major=False)
            h = outputs[:, -1, i, :]
            # use square difference as the loss for gradient matching.
            # why don't use the true loss and optimize match to the target class instead?
            cost = tf.losses.mean_squared_error(residues, h)
            training_op_model = (tf.train.AdamOptimizer(self.gradient_learning_rate).minimize(cost, var_list=cell.get_variable_scope()), cost)

            outputs = outputs[:, -1, :, :]
            age = i + 1
            weighted_outputs = tf.reduce_sum(tf.reshape(all_gammas[0:age], [1, age, 1]) * outputs * self.rate_multiplier, axis=1)
            weighted_outputs = tf.reshape(weighted_outputs, [batches, self.total_classes])

            # true loss, for optimizing learning rate
            cost = tf.losses.softmax_cross_entropy(one_hot_targets, weighted_outputs)
            current_gamma = self.gammas[i]
            training_op_weight = (tf.train.AdamOptimizer(self.lambda_learning_rate).minimize(cost, var_list=[current_gamma]), cost)

            prediction = tf.nn.softmax(weighted_outputs, axis=-1)
            residues = one_hot_targets - prediction
            training_ops.append((training_op_model, training_op_weight))

        confidence = prediction
        results = tf.argmax(prediction, axis=-1)

        return results, confidence, training_ops


def init():
    rnn = Cells.Aggregated_Cell([2, 5])
    booster = Classification_Gradient_Boost(rnn, 8)
    inputs = tf.placeholder(shape=[None, None, rnn.get_depth()], dtype=tf.float32)
    targets = tf.placeholder(shape=[None], dtype=tf.int32)

    results, confidence, training_ops = booster.build_graph(inputs, targets)

    sess.run(tf.global_variables_initializer())
    return results, confidence, training_ops, inputs, targets


def train(data, target, model, sess=None):
    print("train")

    training_ops = model[2]
    input_ref = model[3]
    target_ref = model[4]

    for i in range(len(training_ops)):
        for j in range(1000):
            _, c = sess.run(training_ops[i][0], feed_dict={input_ref: data, target_ref: target})
            print(i, j, c)
        for j in range(1000):
            _, c = sess.run(training_ops[i][1], feed_dict={input_ref: data, target_ref: target})
            print(i, j, c)


def predict(data, model, sess=None):
    print("predict")
    prediction = model[0]
    confidence = model[1]
    input_ref = model[3]
    p, c = sess.run((prediction, confidence), feed_dict={input_ref: data})
    return p, c


if __name__ == "__main__":
    with tf.Session() as sess:
        model = init()
        inputs = np.random.rand(20, 3, 2)
        outputs = np.random.randint(0, 5, [20])

        train(inputs, outputs, model, sess)
        p, c = predict(inputs, model, sess)
        print(outputs)
        print(p)
