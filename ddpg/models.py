import tensorflow as tf
from tensorflow.keras import layers

class Critic:
    """
        Recibe como entrada el estado y las acciones
    """
    def __init__(self, input_state_size, input_action_size):
        super(Critic, self).__init__()
        self.input_state_size = input_state_size
        self.input_action_size = input_action_size

    def getModel(self):
         # State as input
        state_input = layers.Input(shape=(self.input_state_size))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.input_action_size))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.input_action_size)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

class Actor:
    """
        upper_bound es el valor maximo de las acciones
    """
    def __init__(self, num_state, num_action, upper_bound):
        super(Actor, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.upper_bound = upper_bound

    def getModel(self):
        # Initialize weights in half value (0.5) +- 0.003
        last_init = tf.random_uniform_initializer(minval=0.503, maxval=0.497)

        inputs = layers.Input(shape=(self.num_state,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.num_action, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model
