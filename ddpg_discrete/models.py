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
        """
            Para generar un path por cada src,dst, se tendra encuenta los la tm y la tupla src,dst
            La tm va estar en una nn separa da a la tupla, para que luego se concatenen, como en:
            https://github.com/MLJejuCamp2017/DRL_based_SelfDrivingCarControl
        """
        # Initialize weights in half value (0.5) +- 0.003
        last_init = tf.random_uniform_initializer(minval=0.5-0.003, maxval=0.5+0.003)

        #Se recibe la tm y la tupla concatenadas
        inputs = layers.Input(shape=(self.num_state,))
        
        #Separamos la tm de las tuplas para tratarlas diferente
        in_tm = layers.Lambda(lambda x: x[:, :self.num_state-1])(inputs)
        out_tm = layers.Dense(256, activation="relu")(in_tm)
        out_tm = layers.Dense(256, activation="relu")(out_tm)
        # out_tm = layers.Dense(64, activation="relu")(out_tm)

        #Aqui se separa la tupla de nodos, que es el ultimo valor
        in_tuple = layers.Lambda(lambda x: x[:, -1:])(inputs)
        out_tup = layers.Dense(50, activation="relu")(in_tuple)

        #Se unen las dos nn
        concat = layers.Concatenate()([out_tm, out_tup])

        out = layers.Dense(256, activation="relu")(concat)

        outputs = layers.Dense(self.num_action, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model
