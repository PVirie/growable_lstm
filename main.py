import tensorflow as tf
import cell as Cells
import boost as Boost
import numpy as np
import os

root = os.path.dirname(os.path.abspath(__file__))

mnist = tf.keras.datasets.fashion_mnist
# mnist.load_data(os.path.join(root, "artifacts", "fashion_mnist.npz"))
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

x_train = train_images.reshape(-1, 28, 28).astype(np.float32) / 255.0
x_test = test_images.reshape(-1, 28, 28).astype(np.float32) / 255.0
# normalize data
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train, axis=0, keepdims=True)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


y_train = train_labels
y_test = test_labels
print(x_train.shape, x_test.shape)


def init():
    rnn = Cells.Aggregated_Cell(28, 3, 10)
    booster = Boost.Classification_Gradient_Boost(rnn, 10, init_rate=1.0, rate_multiplier=1.0, gradient_learning_rate=0.001, lambda_learning_rate=0.001)

    # Smith, S. L., Kindermans, P. J., Ying, C., & Le, Q. V. (2017). Don't decay the learning rate, increase the batch size. arXiv preprint arXiv:1711.00489.
    dataset_prebatch = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024 * 16)
    dataset_batches = []
    dataset_batches.append(dataset_prebatch.batch(50))
    dataset_batches.append(dataset_prebatch.batch(100))
    dataset_batches.append(dataset_prebatch.batch(150))
    dataset_batches.append(dataset_prebatch.batch(200))

    testset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

    iter = tf.data.Iterator.from_structure(testset.output_types, testset.output_shapes)
    train_init_ops = []
    for item in dataset_batches:
        train_init_ops.append(iter.make_initializer(item))
    test_init_op = iter.make_initializer(testset)

    inputs, targets = iter.get_next()

    results, confidence, training_ops = booster.build_graph(inputs, targets)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.cast(targets, tf.int64), results), tf.float32)) / 100

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    return accuracy, training_ops, train_init_ops, test_init_op, saver


def train(model, sess=None):
    print("train")

    training_ops = model[1]
    train_init_ops = model[2]
    saver = model[4]

    total_boost = len(training_ops)
    for i in range(total_boost):
        for j in range(20):
            train_init_op = train_init_ops[int(j * len(train_init_ops) / 20)]
            sess.run(train_init_op)
            cs = 0
            t = 0
            while True:
                try:
                    _, c = sess.run(training_ops[i][0])
                    cs = cs + c
                    t = t + 1
                except tf.errors.OutOfRangeError:
                    print(i, j, cs / t)
                    break
        for j in range(20):
            train_init_op = train_init_ops[int(j * len(train_init_ops) / 20)]
            sess.run(train_init_op)
            cs = 0
            t = 0
            while True:
                try:
                    _, c = sess.run(training_ops[i][1])
                    cs = cs + c
                    t = t + 1
                except tf.errors.OutOfRangeError:
                    print(i, j, cs / t)
                    break
        saver.save(sess, os.path.join(root, "weight_sets", "model.ckpt"))


def test(model, sess=None):
    print("test")

    accuracy_op = model[0]
    test_init_op = model[3]
    saver = model[4]

    saver.restore(sess, os.path.join(root, "weight_sets", "model.ckpt"))

    sess.run(test_init_op)
    accuracy = 0
    count = 0
    while True:
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
