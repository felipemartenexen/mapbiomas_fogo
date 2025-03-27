#!/usr/bin/env python
# coding: utf-8

# -- PARÂMETROS BÁSICOS --
version = 4
biome = 'amazonia'  # ['pantanal', 'pampa', 'caatinga', 'cerrado', 'mata_atlantica']
regions = [6,7,8,9]  # Lista de regiões
folder = '../../dados'  # Pasta principal onde ficam os dados
folder_modelo = '../../../mnt/Files-Geo/Arquivos/modelos_col3'

# Função para carregar a imagem
def load_image(image_path):
    return gdal.Open(image_path, gdal.GA_ReadOnly)

# Função para converter a imagem em um array NumPy
def convert_to_array(dataset):
    nbr = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.stack(nbr, 2)

# Filtra dados válidos e os embaralha
def filter_valid_data_and_shuffle(data_train_test_vector):
    np.random.shuffle(data_train_test_vector)
    return data_train_test_vector

# Construção de camadas totalmente conectadas
def fully_connected_layer(input, n_neurons, activation=None):
    input_size = input.get_shape().as_list()[1]
    W = tf.Variable(tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))))
    b = tf.Variable(tf.zeros([n_neurons]))
    layer = tf.matmul(input, W) + b
    if activation == 'relu':
        layer = tf.nn.relu(layer)
    return layer

# Loop principal para treinar o modelo em múltiplas regiões
for region in regions:
    print(colored(f'Processando região {region}', 'yellow'))
    folder_amostras = f'../../../mnt/Files-Geo/Arquivos/amostras_col4/{biome}_r5'
    images_train_test = [
        f'train_test_fire_nbr_{biome}_r5_l89_v1_*.tif'
    ]

    # Coleta de dados de treino e teste
    all_data_train_test_vector = []
    for index, images in enumerate(images_train_test):
        images_name = glob.glob(f'{folder_amostras}/{images}')
        if not images_name:
            print(colored(f'Nenhum arquivo encontrado para o padrão {images}', 'red'))
            continue

        for image in images_name:
            try:
                dataset_train_test = load_image(image)
                if dataset_train_test is None:
                    print(colored(f'Erro ao carregar a imagem: {image}', 'red'))
                    continue
                data_train_test = convert_to_array(dataset_train_test)
                vector = data_train_test.reshape([data_train_test.shape[0] * data_train_test.shape[1], data_train_test.shape[2]])
                dataclean = vector[~np.isnan(vector).any(axis=1)]
                if dataclean.size == 0:
                    print(colored(f'A imagem {image} não contém dados válidos.', 'red'))
                    continue
                all_data_train_test_vector.append(dataclean)
            except Exception as e:
                print(colored(f'Erro ao processar a imagem {image}: {e}', 'red'))
                continue

    # Verificar se há dados válidos
    if not all_data_train_test_vector:
        print(colored(f'Nenhum dado válido encontrado para a região {region}. Pulando...', 'red'))
        continue

    try:
        # Concatena os dados e filtra os válidos
        data_train_test_vector = np.concatenate(all_data_train_test_vector)
    except ValueError:
        print(colored(f'Erro ao concatenar dados na região {region}. Pulando...', 'red'))
        continue

    valid_data_train_test = filter_valid_data_and_shuffle(data_train_test_vector)

    # Define índices para as bandas e rótulos
    bi = [0, 1, 2, 3]  # Índices das bandas
    li = 4  # Índice do rótulo

    # Define a fração de treino
    TRAIN_FRACTION = 0.7
    training_size = int(valid_data_train_test.shape[0] * TRAIN_FRACTION)
    training_data = valid_data_train_test[:training_size, :]
    validation_data = valid_data_train_test[training_size:, :]

    # Calcula médias e desvios padrão para normalização
    data_mean = training_data[:, bi].mean(0)
    data_std = training_data[:, bi].std(0)

    # -- HIPERPARÂMETROS --
    lr = 0.001  # Taxa de aprendizado
    BATCH_SIZE = 1000
    N_ITER = 7000
    NUM_INPUT = len(bi)
    NUM_CLASSES = 2
    NUM_N_L1 = 7
    NUM_N_L2 = 14
    NUM_N_L3 = 7
    NUM_N_L4 = 14
    NUM_N_L5 = 7

    # Construção do modelo
    graph = tf.Graph()
    with graph.as_default():
        x_input = tf.placeholder(tf.float32, shape=[None, NUM_INPUT])
        y_input = tf.placeholder(tf.int64, shape=[None])

        normalized = (x_input - data_mean) / data_std
        hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')
        logits = fully_connected_layer(hidden5, n_neurons=NUM_CLASSES)

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input))
        optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

        outputs = tf.argmax(logits, 1)
        correct_prediction = tf.equal(outputs, y_input)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    # Treinamento do modelo
    start_time = time.time()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        validation_dict = {
            x_input: validation_data[:, bi],
            y_input: validation_data[:, li]
        }

        for i in range(N_ITER + 1):
            batch = training_data[np.random.choice(training_size, BATCH_SIZE, False), :]
            feed_dict = {
                x_input: batch[:, bi],
                y_input: batch[:, li]
            }
            optimizer.run(feed_dict=feed_dict)

            if i % 100 == 0:
                acc = accuracy.eval(validation_dict) * 100
                model_path = f'{folder_modelo}/col3_{biome}_r{region}_v{version}_rnn_lstm_ckpt'
                saver.save(sess, model_path)
                print(f'Região {region}: Precisão {acc:.2f}% na iteração {i}')

    elapsed_time = time.time() - start_time
    print(colored(f'Tempo gasto para região {region}: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}', 'yellow'))
    print(colored(f'Modelo salvo em: {model_path}', 'green'))

print(colored('Treinamento completo para todas as regiões.', 'green'))
