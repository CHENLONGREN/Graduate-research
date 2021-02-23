import numpy as np
import tensorflow as tf


class LSTM:
    def __init__(self, input_size, state_size, hidden_sum, output_size, time_steps, batch_size, learning_rate):
        self.input_size = input_size
        self.state_size = state_size
        self.hidden_sum = hidden_sum
        self.output_size = output_size
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.learning_rate = learning_rate
        self.children = tf.placeholder(np.float32, shape=(self.input_size, self.time_steps))
        self.target = tf.placeholder(np.float32, shape=(self.batch_size, self.time_steps))

        self.network()
        self.cost()
        self.optimizer()
        self.Persistent()
        self.Persistent_optimizer()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 定义网络结构
    def network(self):
        # self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.state_size) # 定义RNN的cell
        # self.inputs = tf.placeholder(np.float32, shape=(self.batch_size, self.input_size))  # 输入层
        # self.h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
        # self.output, self.h1 = cell.call(inputs, h0)  # 一次调用call函数
        # 每调用一次这个函数就返回一个BasicRNNCell
        def get_a_cell(state_size):
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size)
        # hidden_sum层RNN,如果hidden_sum=3,它的state_size是(128, 128, 128),表示共有3个隐层状态，每个隐层状态的大小为128
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.state_size) for _ in range(self.hidden_sum)])
        # 输入层,shape = (batch_size, time_steps, input_size)
        self.inputs = tf.placeholder(np.float32, shape=(self.batch_size, self.time_steps, self.input_size))
        initial_state = lstm_cell.zero_state(self.batch_size, np.float32)
        self.outputs, self.state = tf.nn.dynamic_rnn(lstm_cell, self.inputs, initial_state=initial_state,
                                                     dtype=tf.float32)

        self.weights = tf.Variable(tf.truncated_normal([self.state_size, self.output_size], stddev=0.1))  # 正态分布，均值为0，标准差为0.1
        self.biases = tf.Variable(tf.constant(0.1, shape=[self.output_size]))   # 给定值的常量

        self.result = tf.nn.relu(tf.reshape(tf.matmul(tf.reshape(self.outputs, [self.time_steps, self.state_size]),
                                                      self.weights) + self.biases, [self.output_size, self.time_steps]))

    # 定义递归神经网络的变换。在BasicRNNCell中，state_size永远等于output_size,需要额外对输出定义新的变换。递归神经网络
    def Persistent(self):

        def add_layer(input, timestep, output_size, state_size):

            if state_size == None:
                input_reshape = tf.reshape(input, [-1, timestep])
                Weight = tf.Variable(tf.truncated_normal([timestep, timestep], stddev=0.1))
                biases = tf.Variable(tf.constant(0.1, shape=[output_size]))
                output = tf.nn.tanh(tf.matmul(input_reshape, Weight) + biases)
            else:
                input_reshape = tf.reshape(input, [timestep, state_size])
                Weight = tf.Variable(tf.truncated_normal([state_size, timestep], stddev=0.1))
                biases = tf.Variable(tf.constant(0.1, shape=[output_size]))
                output = tf.nn.tanh(tf.matmul(input_reshape, Weight) + biases)

            return output

        def parents_layer(input1, input2, timestep, output_size):
            biases1 = tf.Variable(tf.constant(0.1, shape=[output_size]))
            input = input1 + input2 + biases1
            Weight = tf.Variable(tf.truncated_normal([timestep, output_size], stddev=0.1))
            biases2 = tf.Variable(tf.constant(0.1, shape=[output_size]))
            output = tf.nn.relu(tf.matmul(input, Weight) + biases2)
            return output

        # output_reshape = tf.reshape(self.outputs, [-1, self.state_size])
        # parents = tf.concat((output_reshape, self.children), 1)
        Uh = add_layer(self.outputs, self.time_steps, self.output_size, self.state_size)
        Wx = add_layer(self.children, self.time_steps, self.output_size, None)
        self.Persistent_result = tf.reshape(parents_layer(Uh, Wx, self.time_steps, self.output_size), [self.output_size, self.time_steps])
        # self.result = tf.reshape(result0, [self.batch_size, self.time_steps])

    def cost(self):
        # self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(label, self.result), 2.0))
        self.cost = tf.losses.mean_squared_error(self.target, self.result)

    def optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def Persistent_optimizer(self):
        # self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(label, self.result), 2.0))
        self.Persistent_cost = tf.losses.mean_squared_error(self.target, self.Persistent_result)
        self.Persistent_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize( self.Persistent_cost)

    # 定义执行一步训练的函数
    def opt(self, X, Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.inputs: X, self.target: Y})
        return cost

    def Persistent_opt(self, X, Y, C):
        cost, opt = self.sess.run((self.Persistent_cost, self.Persistent_optimizer), feed_dict={self.inputs: X, self.target: Y, self.children: C})
        return cost

    # 返回输出层的结果
    def output(self, X, Y, C):
        result_tensor, opt = self.sess.run((self.Persistent_result, self.Persistent_optimizer), feed_dict={self.inputs: X, self.target: Y, self.children: C})
        return result_tensor
