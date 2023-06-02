import tensorflow as tf


class GazeNet(tf.keras.Model):
    """
    Multi-layer perceptron for estimating gaze depth
    """

    def __init__(self, epochs=25):

        super(GazeNet, self).__init__()
        # parameters
        self.loss = 'mean_absolute_error'
        self.epochs =  epochs
        self.lr = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.lr) 

        # layers
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.final = tf.keras.layers.Dense(1, activation='relu')


    def call(self, train_features, training=True):
        """
        override call method for tf.keras.Model
        """
        # norm = self.normalizer.adapt(train_features)
        d1 = self.dense1(train_features)
        d2 = self.dense2(d1)
        d3 = self.dense3(d2)
        output = self.final(d3) 

        return output 

        


if __name__ == "__main__":

    net = GazeNet()