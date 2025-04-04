#!/usr/bin/env python
# coding: utf-8


# Define the list of regions
regions = [1,2,3,4,5,6,7,8,9]
version = '9_12'
biome = 'amazonia'  # ['pantanal', 'pampa', 'caatinga', 'cerrado', 'mata_atlantica']
folder = '../dados'  # Main folder where the data is located
folder_modelo = '../../../mnt/Files-Geo/Arquivos/modelos_monitor'
#ee.Initialize(project='workspace-ipam')

def load_image(image):
    return gdal.Open(image, gdal.GA_ReadOnly)

def convert_to_array(dataset):
    nbr = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.stack(nbr, 2)

for region in regions:
    region = str(region)
    sub_region = '1'  # Assuming sub_region is the same as region
    folder_amostras = f'../../../mnt/Files-Geo/Arquivos/amostras_monitor/{biome}_r{sub_region}'
    images_train_test = [
        f'train_test_fire_nbr_{biome}_r{sub_region}_sentinel_v{version}_*.tif'
    ]

    # Initialize list to hold all training data
    all_data_train_test_vector = []

    # ### TRAINING AND TEST IMAGES ###
    for index, images in enumerate(images_train_test):
        images_name = glob.glob(f'{folder_amostras}/{images}')

        for image in images_name:
            dataset_train_test = load_image(image)
            data_train_test = convert_to_array(dataset_train_test)

            # Reshape training and test data
            vector = data_train_test.reshape([data_train_test.shape[0] * data_train_test.shape[1], data_train_test.shape[2]])
            dataclean = vector[~np.isnan(vector).any(axis=1)]
            all_data_train_test_vector.append(dataclean)

    # ### PREPARE DATA FOR THE MODEL ###
    def filter_valid_data_and_shuffle(data_train_test_vector): 
        np.random.shuffle(data_train_test_vector)
        return data_train_test_vector

    # Concatenate training and test data
    data_train_test_vector = np.concatenate(all_data_train_test_vector)

    # Select only valid data and shuffle
    valid_data_train_test = filter_valid_data_and_shuffle(data_train_test_vector)

    bi = [0,1,2,3]  # Index of NBR bands
    li = 4  # Index of label

    # SPLIT DATA INTO TRAINING AND VALIDATION SETS
    TRAIN_FRACTION = 0.7

    training_size = int(valid_data_train_test.shape[0] * TRAIN_FRACTION)
    training_data= valid_data_train_test[0:training_size,:]
    validation_data = valid_data_train_test[training_size:-1,:]

    # Compute per-band means and standard deviations of the input NBR
    data_mean = training_data[:,bi].mean(0)
    data_std = training_data[:,bi].std(0)

    # HYPERPARAMETERS
    lr = 0.001  # Learning rate
    BATCH_SIZE = 1000
    N_ITER = 7000
    NUM_INPUT = len(bi)
    NUM_N_L1 = 7
    NUM_N_L2 = 14
    NUM_N_L3 = 7
    NUM_N_L4 = 14
    NUM_N_L5 = 7
    NUM_CLASSES = 2

    # BUILD THE MODEL
    def fully_connected_layer(input, n_neurons, activation=None):
        # Layer variables
        input_size = input.get_shape().as_list()[1]
        W = tf.Variable(tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))))
        b = tf.Variable(tf.zeros([n_neurons]))

        # Linear operation
        layer = tf.matmul(input, W) + b

        # Apply non-linearity
        if activation == 'relu':
            layer = tf.nn.relu(layer)

        return layer

    graph = tf.Graph()
    with graph.as_default():
        # Input layers
        x_input = tf.placeholder(tf.float32, shape=[None, NUM_INPUT])
        y_input = tf.placeholder(tf.int64, shape=[None])

        normalized = (x_input - data_mean) / data_std
        hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')

        logits = fully_connected_layer(hidden5, n_neurons=NUM_CLASSES)

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input, name='error'))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

        outputs = tf.argmax(logits, 1)

        correct_prediction = tf.equal(outputs, y_input)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initializer
        init = tf.global_variables_initializer()

        # Saver to save the trained model
        saver = tf.train.Saver()

    # EXECUTE THE MODEL
    start_time = time.time()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        validation_dict = {
            x_input: validation_data[:,bi],
            y_input: validation_data[:,li]
        }

        for i in range(N_ITER + 1):
            batch = training_data[np.random.choice(training_size, BATCH_SIZE, False), :]
            # Create the training feed_dict
            feed_dict = {
              x_input: batch[:,bi], 
              y_input: batch[:,li]
            }
            # Run the training iteration
            optimizer.run(feed_dict=feed_dict)

            if i % 100 == 0:
                # Calculate accuracy
                acc = accuracy.eval(validation_dict) * 100

                # Save the model variables
                model_path = f'{folder_modelo}/monitor_{biome}_r{region}_v{version}_rnn_lstm_ckpt'
                saver.save(sess, model_path)
                print('Accuracy %.2f%% at step %s' %(acc, i))

    end_time = time.time()
    training_time = end_time - start_time
    print(colored('Spent time: {0}'.format(time.strftime("%H:%M:%S", time.gmtime(training_time))), 'yellow'))
    print(colored(f'Model saved at: {model_path}','green'))
