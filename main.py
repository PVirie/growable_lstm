import tensorflow as tf
import cell as Cells
import boost as Boost
import numpy as np
import os

root = os.path.dirname(os.path.abspath(__file__))

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


x_train = train_images.reshape(-1, 28, 28).astype(np.float32) / 255.0
x_test = test_images.reshape(-1, 28, 28).astype(np.float32) / 255.0


def init():
    rnn = Cells.Aggregated_Cell([28, 10])
    booster = Boost.Classification_Gradient_Boost(rnn, 10, init_rate=0.01, rate_multiplier=1.0, gradient_learning_rate=0.001, lambda_learning_rate=0.001)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, train_labels)).shuffle(1024 * 16).batch(100)
    testset = tf.data.Dataset.from_tensor_slices((x_test, test_labels)).batch(100)

    iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    train_init_op = iter.make_initializer(dataset)
    test_init_op = iter.make_initializer(testset)

    inputs, targets = iter.get_next()

    results, confidence, training_ops = booster.build_graph(inputs, targets)
    accuracy = tf.metrics.accuracy(targets, results)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    return accuracy, training_ops, train_init_op, test_init_op, saver


def train(model, sess=None):
    print("train")

    training_ops = model[1]
    train_init_op = model[2]
    saver = model[4]

    for i in range(len(training_ops)):
        for j in range(10):
            sess.run(train_init_op)
            while True:
                try:
                    _, c = sess.run(training_ops[i][0])
                except tf.errors.OutOfRangeError:
                    print(i, j, c)
                    break
        for j in range(10):
            sess.run(train_init_op)
            while True:
                try:
                    _, c = sess.run(training_ops[i][1])
                except tf.errors.OutOfRangeError:
                    print(i, j, c)
                    break
        saver.save(sess, os.path.join(root, "weight_sets", "model.ckpt"))


def test(data, model, sess=None):
    print("test")

    accuracy_op = model[0]
    test_init_op = model[3]
    saver = model[4]

    saver.restore(sess, os.path.join(root, "weight_sets", "model.ckpt"))

    sess.run(test_init_op)
    while True:
        accuracy = 0
        count = 0
        try:
            acc = sess.run(accuracy_op)
            accuracy = accuracy + acc
            count = count + 1
        except tf.errors.OutOfRangeError:
            print("Accuracy:", accuracy / count)
            break


if __name__ == "__main__":
    with tf.Session() as sess:
        model = init()
        train(model, sess)
        test(model, sess)
