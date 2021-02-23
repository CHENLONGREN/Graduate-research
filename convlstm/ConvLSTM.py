import tensorflow as tf
from cell import ConvLSTMCell
import numpy as np

class ConvLSTM():
    def __init__(self, batch_size, timesteps, shape, channels, kernel, filters, learning_rate):
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.shape = shape
        self.channels = channels
        self.kernel = kernel
        self.filters = filters
        self.learning_rate = learning_rate
        self.children = tf.placeholder(np.float32,
                                       shape=([self.batch_size, self.timesteps] + self.shape + [self.channels]))

        # Create a placeholder for videos.
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.timesteps] + self.shape + [self.channels])

        # Add the ConvLSTM step.
        cell = ConvLSTMCell(self.shape, self.filters, self.kernel)
        self.outputs, self.state = tf.nn.dynamic_rnn(cell, self.inputs, dtype=self.inputs.dtype)

        self.result = tf.nn.relu(self.outputs)

        self.label = tf.placeholder(np.float32, [self.batch_size, self.timesteps] + self.shape + [self.channels])
        # label = tf.reshape(self.target, (self.batch_size, self.time_steps))
        self.cost = tf.losses.mean_squared_error(self.label, self.result)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.Persistent()
        self.Persistent_optimizer()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def Persistent(self):

        def add_layer(input, timestep, shape):
            input_reshape = tf.reshape(input, [timestep, shape[0], shape[1]])
            Weight = tf.Variable(tf.truncated_normal([timestep, shape[1], shape[1]], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[timestep, shape[0], shape[1]]))
            output = tf.nn.tanh(tf.matmul(input_reshape, Weight) + biases)
            return output

        def parents_layer(input1, input2, timestep, shape):
            biases1 = tf.Variable(tf.constant(0.1, shape=[timestep, shape[0], shape[1]]))
            input = input1 + input2 + biases1
            Weight = tf.Variable(tf.truncated_normal([timestep, shape[1], shape[1]], stddev=0.1))
            biases2 = tf.Variable(tf.constant(0.1, shape=[timestep, shape[0], shape[1]]))
            output = tf.nn.relu(tf.matmul(input, Weight) + biases2)
            return output

        # output_reshape = tf.reshape(self.outputs, [-1, self.state_size])
        # parents = tf.concat((output_reshape, self.children), 1)
        Uh = add_layer(self.outputs, self.timesteps, self.shape)
        Wx = add_layer(self.children, self.timesteps, self.shape)
        self.Persistent_result = parents_layer(Uh, Wx, self.timesteps, self.shape)
        # self.result = tf.reshape(result0, [self.batch_size, self.time_steps])

    def Persistent_optimizer(self):
        # self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(label, self.result), 2.0))
        self.Persistent_label = tf.reshape(self.label, [self.timesteps, self.shape[0], self.shape[1]])
        self.Persistent_cost = tf.losses.mean_squared_error( self.Persistent_label, self.Persistent_result)
        self.Persistent_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize( self.Persistent_cost)

    # 定义执行一步训练的函数
    def opt(self, X, Y):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.inputs: X, self.label: Y})
        return cost

    def Persistent_opt(self, X, Y, C):
        cost, opt = self.sess.run((self.Persistent_cost, self.Persistent_optimizer), feed_dict={self.inputs: X, self.label: Y, self.children: C})
        return cost

    # 返回输出层的结果
    def output(self, X, Y, C):
        result_tensor, opt = self.sess.run((self.Persistent_result, self.Persistent_optimizer), feed_dict={self.inputs: X, self.label: Y, self.children: C})
        return result_tensor

    def output1(self, X, Y):
        result_tensor, opt = self.sess.run((self.result, self.optimizer), feed_dict={self.inputs: X, self.label: Y})
        return result_tensor











