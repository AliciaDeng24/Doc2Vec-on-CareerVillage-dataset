def nn_model(que, pro, dim_list):
    '''
    Keras-based neural network that takes in question doc2vec vectors and
    outputs professional doc2vec vectorsself.

    Returns:
    Model - A keras Sequential model

    Note that this function restricts the neural network to have three dense layers.
    '''

    Model = Sequential()
    Model.add(Dense(dim_list[0], input_shape=(len(que[0]),), activation='tanh'))
    Model.add(Dense(dim_list[1], activation='tanh'))
    Model.add(Dense(dim_list[2], activation='tanh'))
    Model.add(Dense(dim_list[3], activation='tanh'))
    Model.add(Dense(len(pro[0]), activation='tanh'))

    return Model


def construct_model(input_size, dim_1, dim_2, dim_output):
    '''
    Instantiate nn model using Keras Model.
    Function parameters are used to specify the shape of input and output tensors

    Returns:
    model - a keras model object
    '''

    # Define the shape of input tensor
    inputs = Input(shape=(input_size,))

    layer_1 = Dense(units=dim_1, activation='tanh', kernel_regularizer=l1(0.1))(inputs)
    layer_2 = Dense(units=dim_2, activation='tanh', kernel_regularizer=l1(0.1))(layer_1)
    # layer_3 = Dense(units=dim_3, activation='tanh', kernel_regularizer=l1(0.1))(layer_2)

    out = Dense(units=dim_output, activation='tanh', kernel_regularizer=l1(0.1))(layer_2)

    model = Model(inputs=inputs, outputs=out)

    return model
