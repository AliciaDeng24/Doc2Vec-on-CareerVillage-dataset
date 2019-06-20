from NLP import *
from NN_Models import construct_model

# Alternative model

# Train two encoder neural network respectively on embedded questions
# and professionals vectors, having them each output encoded version
# of vectors of equal shape, and then minimize the cosine distance
# between two encoded vectors

# Define shapes of model input
pro_size = pro_input.shape[1]
ques_size = ques_input.shape[1]

# Use Keras Functional API to construct two models
dim_output = 10

model_ques = construct_model(ques_size, 32, 16, dim_output)
model_pro = construct_model(pro_size, 32, 16, dim_output)

# Use inputs and outputs of two models to define a Difference model
# Keras Lambda to construct a tensor to compute Difference
diff = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]))))([model_ques.output, model_pro.output])
diff = Lambda(lambda x: tf.reshape(x, (-1,1)))(diff)
DiffModel = Model(inputs=[model_ques.input, model_pro.input], outputs=diff)

desired_output = np.zeros((len(pro_input), 1))

# Self define a loss function (cosine proximity)
# Credit to https://github.com/keras-team/keras/issues/3031

def cosine_dist(y_1, y_2):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())
    y_1 = l2_normalize(y_1, axis=-1)
    y_2 = l2_normalize(y_2, axis=-1)
    return 1-K.mean(y_1 * y_2, axis=-1)

# Training
DiffModel.compile(optimizer=Adam(lr = 0.001), loss='mape', metrics=['accuracy'])
DiffModel.fit([ques_input, pro_input], desired_output, epochs=5, validation_split=0.1)

print(DiffModel.summary())

# Using trained professional model for prediction
test_ques_latent_vec = model_ques.predict(test_input_mtx)
pro_latent_vec = model_pro.predict(pro_input)

# Make recommendations
recommenders = find_all_closest(pro_ids, pro_input ques_ids, pro_latent_vec)
