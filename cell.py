import tensorflow as tf


class Weak_Cell(tf.contrib.rnn.RNNCell):

    # sizes = (input_depth, age, memory_channels, output_size)
    def __init__(self, depth, age, memory_channels, output_size):

        self.depth = depth
        self.age = age
        self.memory_channels = memory_channels
        self.output_size_ = output_size
        self.name_ = "WC" + str(age)
        with tf.variable_scope(self.name_):
            self.Wif = tf.Variable(tf.random_normal([depth, memory_channels], stddev=0.01), dtype=tf.float32)
            self.Whf = tf.Variable(tf.random_normal([age * memory_channels, memory_channels], stddev=0.01), dtype=tf.float32)
            self.bf = tf.Variable(tf.ones([memory_channels]), dtype=tf.float32)

            self.Wii = tf.Variable(tf.random_normal([depth, memory_channels], stddev=0.01), dtype=tf.float32)
            self.Whi = tf.Variable(tf.random_normal([age * memory_channels, memory_channels], stddev=0.01), dtype=tf.float32)
            self.bi = tf.Variable(tf.ones([memory_channels]), dtype=tf.float32)

            self.Wic = tf.Variable(tf.random_normal([depth, memory_channels], stddev=0.01), dtype=tf.float32)
            self.Whc = tf.Variable(tf.random_normal([age * memory_channels, memory_channels], stddev=0.01), dtype=tf.float32)
            self.bc = tf.Variable(tf.zeros([memory_channels]), dtype=tf.float32)

            self.Wio = tf.Variable(tf.random_normal([depth, output_size], stddev=0.01), dtype=tf.float32)
            self.Who = tf.Variable(tf.random_normal([age * memory_channels, output_size], stddev=0.01), dtype=tf.float32)
            self.bo = tf.Variable(tf.zeros([output_size]), dtype=tf.float32)

            self.Wco = tf.Variable(tf.random_normal([memory_channels, output_size], stddev=0.01), dtype=tf.float32)
            self.bco = tf.Variable(tf.zeros([output_size]), dtype=tf.float32)

    @property
    def state_size(self):
        return self.memory_channels

    @property
    def output_size(self):
        return self.output_size_

    # cell's state = [batch, memory_channels]
    def init_state(self, batch_size):
        state = tf.random_normal([batch_size, self.memory_channels], dtype=tf.float32) * 0.01
        return state

    def build_gate(self, h, x, Wi, Wh, b, activation=tf.nn.sigmoid):
        Wix_Whh = tf.matmul(h, Wh) + tf.matmul(x, Wi)
        return activation(Wix_Whh + b)

    # states = [batch, age, memory_channels]
    def __call__(self, input, states, scope=None):

        states = tf.reshape(states[:, 0:self.age, :], [-1, self.age * self.memory_channels])

        gf = self.build_gate(states, input, self.Wif, self.Whf, self.bf)
        gi = self.build_gate(states, input, self.Wii, self.Whi, self.bi)
        C_ = self.build_gate(states, input, self.Wic, self.Whc, self.bc, tf.nn.tanh)
        go = self.build_gate(states, input, self.Wio, self.Who, self.bo)

        C = gf * states[:, -self.memory_channels:] + gi * C_
        o = go * tf.nn.tanh(tf.matmul(C, self.Wco) + self.bco)

        return o, C

    def get_variable_scope(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_)


class Aggregated_Cell(tf.contrib.rnn.RNNCell):

        # sizes = (depth, memory_channels, output_size)
    def __init__(self, depth, memory_channels, output_size):
        self.depth = depth
        self.memory_channels = memory_channels
        self.output_size_ = output_size
        self.cells_ = []
        self.current_age_ = 0

    def grow(self):
        self.current_age_ = self.current_age_ + 1
        self.cells_.append(Weak_Cell(self.depth, self.current_age_, self.memory_channels, self.output_size_))

    def get_last_cell(self):
        return self.cells_[-1]

    def get_depth(self):
        return self.depth

    def get_num_outputs(self):
        return self.output_size_

    @property
    def state_size(self):
        return self.current_age_

    @property
    def output_size(self):
        return tf.TensorShape([self.current_age_, self.output_size_])

    def init_state(self, batch_size):
        out_states = []
        for cell in self.cells_:
            state = cell.init_state(batch_size)
            out_states.append(state)
        out_states = tf.stack(out_states, axis=1)
        return out_states

    # states = [batches, age, memory_channels]
    def __call__(self, input, states, scope=None):

        outputs = []
        out_states = []
        for cell in self.cells_:
            output, state = cell(input, states)
            out_states.append(state)
            outputs.append(output)

        out_states = tf.stack(out_states, axis=1)
        outputs = tf.stack(outputs, axis=1)

        return outputs, out_states


if __name__ == "__main__":
    with tf.Session() as sess:
        cells = Aggregated_Cell(3, 1, 9)
        cells.grow()
        cells.grow()
        cells.grow()
        states = cells.init_state(20)
        rnn_inputs = tf.random_normal([20, 5, 3], 0.0, 1.0, dtype=tf.float32)

        for i in range(5):
            output, states = cells(rnn_inputs[:, i, ...], states)

        sess.run(tf.global_variables_initializer())
        out, state = sess.run((output, states), feed_dict={})
        print(out.shape, state.shape)
