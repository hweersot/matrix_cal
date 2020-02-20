from base.base_model import BaseModel
import tensorflow as tf


class mlp(BaseModel):
    def __init__(self, config):
        super(mlp, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.state_size[0]])
        # network architecture
#        d1 = tf.layers.dense(self.x, 1,activation=tf.nn.softmax, name="dense1")

        W1 = tf.Variable(tf.random_normal([self.config.state_size[0],1],stddev=1) , name="weight1")
        b1 = tf.constant([0.]*self.config.state_size[0])
        d1 = tf.matmul(self.x, W1)+b1

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d1))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
#                learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor,
#                                           200, 0.97, staircase=True)
#                self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy,
#                                                                                         global_step=self.global_step_tensor)

            correct_prediction = tf.reduce_mean(tf.square(d1-self.y))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

