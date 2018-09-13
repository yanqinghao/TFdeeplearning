import os
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import siamese_similarity_model as model

sess = tf.Session()
batch_size = 200
n_batches = 300
max_address_len = 20
margin = 0.25
num_features = 50
dropout_keep_prob = 0.8
def snn(address1, address2, dropout_keep_prob,
    vocab_size, num_features, input_length):
    # Define the siamese double RNN with a fully connected layer at the end
    def siamese_nn(input_vector, num_hidden):
        cell_unit = tf.nn.rnn_cell.BasicLSTMCell
        # Forward direction cell
        lstm_forward_cell = cell_unit(num_hidden, forget_bias=1.0)
        lstm_forward_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_forward_cell, output_keep_prob=dropout_keep_prob)
        # Backward direction cell
        lstm_backward_cell = cell_unit(num_hidden, forget_bias=1.0)
        lstm_backward_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_backward_cell, output_keep_prob=dropout_keep_prob)
        # Split title into a character sequence
        input_embed_split = tf.split(1, input_length, input_vector)
        input_embed_split = [tf.squeeze(x, squeeze_dims=[1]) for x
                             in input_embed_split]
        # Create bidirectional layer
        outputs, _, _ = tf.nn.bidirectional_rnn(lstm_forward_cell, lstm_backward_cell, input_embed_split, dtype = tf.float32)
        # Average The output over the sequence
        temporal_mean = tf.add_n(outputs) / input_length
        # Fully connected layer
        output_size = 10
        A = tf.get_variable(name="A", shape=[2 * num_hidden, output_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name="b", shape=[output_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        final_output = tf.matmul(temporal_mean, A) + b
        final_output = tf.nn.dropout(final_output, dropout_keep_prob)
        return (final_output)

    with tf.variable_scope("siamese") as scope:
        output1 = siamese_nn(address1, num_features)
        # Declare that we will use the same variables on the second string
        scope.reuse_variables()
        output2 = siamese_nn(address2, num_features)
    # Unit normalize the outputs
    output1 = tf.nn.l2_normalize(output1, 1)
    output2 = tf.nn.l2_normalize(output2, 1)
    # Return cosine distance
    # in this case, the dot product of the norms is the same.
    dot_prod = tf.reduce_sum(tf.mul(output1, output2), 1)
    return (dot_prod)
def get_predictions(scores):
    predictions = tf.sign(scores, name="predictions")
    return(predictions)
def loss(scores, y_target, margin):
    # Calculate the positive losses
    pos_loss_term = 0.25 * tf.square(tf.sub(1., scores))
    pos_mult = tf.cast(y_target, tf.float32)
    # Make sure positive losses are on similar strings
    positive_loss = tf.mul(pos_mult, pos_loss_term)
    # Calculate negative losses, then make sure on dissimilar strings
    neg_mult = tf.sub(1., tf.cast(y_target, tf.float32))
    negative_loss = neg_mult*tf.square(scores)
    # Combine similar and dissimilar losses
    loss = tf.add(positive_loss, negative_loss)
    # Create the margin term. This is when the targets are 0., and the scores are less than m, return 0.
    # Check if target is zero (dissimilar strings)
    target_zero = tf.equal(tf.cast(y_target, tf.float32), 0.)
    # Check if cosine outputs is smaller than margin
    less_than_margin = tf.less(scores, margin)
    # Check if both are true
    both_logical = tf.logical_and(target_zero, less_than_margin)
    both_logical = tf.cast(both_logical, tf.float32)
    # If both are true, then multiply by (1-1)=0.
    multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
    total_loss = tf.mul(loss, multiplicative_factor)
    # Average loss over batch
    avg_loss = tf.reduce_mean(total_loss)
    return(avg_loss)
def accuracy(scores, y_target):
    predictions = get_predictions(scores)
    correct_predictions = tf.equal(predictions, y_target)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return(accuracy)
def create_typo(s):
    rand_ind = random.choice(range(len(s)))
    s_list = list(s)
    s_list[rand_ind]=random.choice(string.ascii_lowercase + '0123456789')
    s = ''.join(s_list)
    return(s)
street_names = ['abbey', 'baker', 'canal', 'donner', 'elm',
'fifth', 'grandvia', 'hollywood', 'interstate', 'jay', 'kings']
street_types = ['rd', 'st', 'ln', 'pass', 'ave', 'hwy', 'cir',
'dr', 'jct']
test_queries = ['111 abbey ln', '271 doner cicle',
'314 king avenue', 'tensorflow is fun']
test_references = ['123 abbey ln', '217 donner cir', '314 kingsave', '404 hollywood st', 'tensorflow is so fun']
def get_batch(n):
    # Generate a list of reference addresses with similar addresses that have
    # a typo.
    numbers = [random.randint(1, 9999) for i in range(n)]
    streets = [random.choice(street_names) for i in range(n)]
    street_suffs = [random.choice(street_types) for i in range(n)]
    full_streets = [str(w) + ' ' + x + ' ' + y for w,x,y in
    zip(numbers, streets, street_suffs)]
    typo_streets = [create_typo(x) for x in full_streets]
    reference = [list(x) for x in zip(full_streets, typo_streets)]
    # Shuffle last half of them for training on dissimilar addresses
    half_ix = int(n/2)
    bottom_half = reference[half_ix:]
    true_address = [x[0] for x in bottom_half]
    typo_address = [x[1] for x in bottom_half]
    typo_address = list(np.roll(typo_address, 1))
    bottom_half = [[x,y] for x,y in zip(true_address, typo_address)]
    reference[half_ix:] = bottom_half
    # Get target similarities (1's for similar, -1's for nonsimilar)
    target = [1]*(n-half_ix) + [-1]*half_ix
    reference = [[x,y] for x,y in zip(reference, target)]
    return(reference)
vocab_chars = string.ascii_lowercase + '0123456789 '
vocab2ix_dict = {char:(ix+1) for ix, char in enumerate(vocab_chars)}
vocab_length = len(vocab_chars) + 1
# Define vocab one-hot encoding
def address2onehot(address, vocab2ix_dict = vocab2ix_dict, max_address_len = max_address_len):
    # translate address string into indices
    address_ix = [vocab2ix_dict[x] for x in list(address)]
    # Pad or crop to max_address_len
    address_ix = (address_ix + [0]*max_address_len)[0:max_address_len]
    return(address_ix)
address1_ph = tf.placeholder(tf.int32, [None, max_address_len],
name="address1_ph")
address2_ph = tf.placeholder(tf.int32, [None, max_address_len],
name="address2_ph")
y_target_ph = tf.placeholder(tf.int32, [None], name="y_target_ph")
dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")
# Create embedding lookup
identity_mat = tf.diag(tf.ones(shape=[vocab_length]))
address1_embed = tf.nn.embedding_lookup(identity_mat, address1_ph)
address2_embed = tf.nn.embedding_lookup(identity_mat, address2_ph)
# Define Model
text_snn = model.snn(address1_embed, address2_embed, dropout_keep_prob_ph,
vocab_length, num_features, max_address_len)
# Define Accuracy
batch_accuracy = model.accuracy(text_snn, y_target_ph)
# Define Loss
batch_loss = model.loss(text_snn, y_target_ph, margin)
# Define Predictions
predictions = model.get_predictions(text_snn)
# Declare optimizer
optimizer = tf.train.AdamOptimizer(0.01)
# Apply gradients
train_op = optimizer.minimize(batch_loss)
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)
train_loss_vec = []
train_acc_vec = []
for b in range(n_batches):
    # Get a batch of data
    batch_data = get_batch(batch_size)
    # Shuffle data
    np.random.shuffle(batch_data)
    # Parse addresses and targets
    input_addresses = [x[0] for x in batch_data]
    target_similarity = np.array([x[1] for x in batch_data])
    address1 = np.array([address2onehot(x[0]) for x in input_addresses])
    address2 = np.array([address2onehot(x[1]) for x in input_addresses])
    train_feed_dict = {address1_ph: address1,
    address2_ph: address2,
    y_target_ph: target_similarity,
    dropout_keep_prob_ph: dropout_keep_prob}
    _, train_loss, train_acc = sess.run([train_op, batch_loss, batch_accuracy], feed_dict=train_feed_dict)
    # Save train loss and accuracy
    train_loss_vec.append(train_loss)
    train_acc_vec.append(train_acc)
test_queries_ix = np.array([address2onehot(x) for x in test_queries])
test_references_ix = np.array([address2onehot(x) for x in test_references])
num_refs = test_references_ix.shape[0]
best_fit_refs = []
for query in test_queries_ix:
    test_query = np.repeat(np.array([query]), num_refs, axis=0)
    test_feed_dict = {address1_ph: test_query,address2_ph: test_references_ix, y_target_ph: target_similarity, dropout_keep_prob_ph: 1.0}
    test_out = sess.run(text_snn, feed_dict=test_feed_dict)
    best_fit = test_references[np.argmax(test_out)]
    best_fit_refs.append(best_fit)
print('Query Addresses: {}'.format(test_queries))
print('Model Found Matches: {}'.format(best_fit_refs))

# Plot the loss and accuracy
plt.plot(train_loss_vec, 'k-', lw=2, label='Batch Loss')
plt.plot(train_acc_vec, 'r:', label='Batch Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy and Loss')
plt.title('Accuracy and Loss of Siamese RNN')
plt.grid()
plt.legend(loc='lower right')
plt.show()
