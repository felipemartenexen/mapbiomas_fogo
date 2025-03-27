
# Initialize Earth Engine
#ee.Initialize(project='workspace-ipam')

# -- BASIC PARAMETERS --
# ------------------------------------------------
regions = [7,8,9]
version = '9_13'
biome = 'amazonia'
resolution = '30'  # '10' or '30'
folder = '../dados'
folder_modelo = '../../../mnt/Files-Geo/Arquivos/modelos_monitor'
folder_mosaic = f'../../../mnt/Files-Geo/Arquivos/mosaics_monitor_sentinel_{resolution}m/vrt/{biome}'
years = [2025]  # You can add more years here
meses = [2]  # You can add more months here
sulfix = ''

# Set the GPU memory growth to limit memory usage
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Reshape into a single vector of pixels for data classification
def reshape_single_vector(data_classify):
    data_classify_vector = data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])
    return data_classify_vector

def classify(data_classify_vector, folder_modelo, region):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Restore variable values
        saver.restore(sess, f'{folder_modelo}/monitor_{biome}_r{region}_v{version}_rnn_lstm_ckpt')

        # Perform classification - divide into parts to avoid memory issues
        output_data_classify_parts = []
        indices = [0, 4000000, 8000000, 12000000, len(data_classify_vector)]
        for i in range(len(indices)-1):
            part = outputs.eval({x_input: data_classify_vector[indices[i]:indices[i+1], bi]})
            output_data_classify_parts.append(part)

        output_data_classify = np.concatenate(output_data_classify_parts)

    # Clear TensorFlow session resources
    tf.keras.backend.clear_session()
    return output_data_classify

def reshape_image_output(output_data_classified, data_classify):
    output_image_data = output_data_classified.reshape(
        [data_classify.shape[0], data_classify.shape[1]])
    return output_image_data

def filter_spatial(output_image_data):
    binary_image = output_image_data > 0

    # Remove small white regions
    open_image = ndimage.binary_opening(
        binary_image, structure=np.ones((4, 4)))
    # Remove small black holes
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))
    return close_image

def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    cols = dataset_classify.RasterXSize
    rows = dataset_classify.RasterYSize
    ds = dataset_classify

    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Float32)
    outDs.GetRasterBand(1).WriteArray(image_data_scene)

    # Add GeoTransform and Projection
    geotrans = ds.GetGeoTransform()
    proj = ds.GetProjection()
    outDs.SetGeoTransform(geotrans)
    outDs.SetProjection(proj)
    outDs.FlushCache()
    outDs = None

def render_classify(dataset_classify, folder_modelo, region):
    data_classify = convert_to_array(dataset_classify)

    data_classify_vector = reshape_single_vector(data_classify)

    output_data_classified = classify(data_classify_vector, folder_modelo, region)

    output_image_data = reshape_image_output(
        output_data_classified, data_classify)

    return filter_spatial(output_image_data)

def read_grid_landsat(biome, region):
    grid = ee.FeatureCollection(f'users/geomapeamentoipam/AUXILIAR/grid_regions_monitor/grid-{biome}-{region}')
    grid_features = grid.getInfo()['features']
    return grid_features

# Clip the image by the grid within the biome limit
def clip_image_by_grid(geom, image, output):
    with rasterio.open(image) as src:
        print(f"Image path: {image}")
        print(f"Output path: {output}")
        try:
            out_image, out_transform = mask(
                src, geom, crop=True, nodata=np.nan, filled=True)
        except ValueError as e:
            print(f'Skipping image: {image} - {str(e)}')
            return
    out_meta = src.meta.copy()

    # Save the resulting raster
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)

for region in regions:
    region = str(region)
    grid_landsat = read_grid_landsat(biome, region)
    start_time = time.time()

    for year in years:
        for mes in meses:
            if mes < 10:
                 vrt_NBR = f'{folder_mosaic}/{biome}_filter_{year}_0{mes}.vrt'
            else:
                 vrt_NBR = f'{folder_mosaic}/{biome}_filter_{year}_{mes}.vrt'  

            input_scenes = []
            total_scenes_done = 0
            for grid in grid_landsat:

                orbit = grid['properties']['ORBITA']
                point = grid['properties']['PONTO']

                output_image_name = f'{folder}/image_monitor_{biome}_r{region}_v{version}_{orbit}_{point}_{year}.tif'

                if not os.path.isfile(output_image_name):

                    geometry_cena = [grid['geometry']]
                    feature_grid = ee.Feature(grid)
                    area_grid = feature_grid.area().divide(1000 * 1000).getInfo()  # km²

                    # Clip the NBR image
                    print(colored(f'Clipping image for scene {orbit}/{point}', 'cyan'))
                    NBR_clipped = f'{folder}/image_mosaic_monitor_{biome}_r{region}_v{version}_{orbit}_{point}_clipped_{year}.tif'

                    try:
                        clip_image_by_grid(geometry_cena, vrt_NBR, NBR_clipped)
                    except ValueError as e:
                        print(colored(f'Error clipping image: {image}', 'red'))

                    dataset_classify = None
                    try:
                        image = NBR_clipped

                        if os.path.isfile(image):
                            dataset_classify = load_image(image)
                        else:
                            print(colored(f'Image not found: {image}', 'red'))
                    except:
                        print(colored(f'Image not found: {image}', 'red'))

                    if dataset_classify:
                        try:
                            image_data = render_classify(dataset_classify, folder_modelo, region)
                            convert_to_raster(dataset_classify, image_data, output_image_name)
                        except:
                            print(colored(f'Error classifying image: {image}', 'red'))

                total_scenes_done += 1

                if os.path.isfile(output_image_name):
                    input_scenes.append(output_image_name)

                    print(colored(f'Done in {year}, {total_scenes_done} scenes of {len(grid_landsat)} scenes ({total_scenes_done/len(grid_landsat)*100:.2f}% complete)', 'green'))

            if len(input_scenes) > 0:
                input_scenes_str = " ".join(input_scenes)

                image_name = f"queimada_{biome}_v{version}_region{region}_{year}_{mes}_{sulfix}"
                output_image = f"{folder}/{image_name}.tif"

                print(colored('Merging all scenes', 'yellow'))
                os.system(f'gdal_merge.py -n 0 -co COMPRESS=PACKBITS -co BIGTIFF=YES -of gtiff {input_scenes_str} -o {output_image}')

                os.system(f'gsutil -m cp {output_image} gs://tensorflow-fire-cerrado1/result_classified_sentinel/{biome}')            

                outputAssetID = f'users/geomapeamentoipam/COLECAO_FOGO_SENTINEL/CLASSIFICACAO/{biome.upper()}/{image_name}'
                bucket = f'gs://tensorflow-fire-cerrado1/result_classified_sentinel/{biome}/{image_name}.tif'            

                os.system('earthengine upload image --asset_id={0} {1}'.format(outputAssetID, bucket))

                os.system(f'rm -rf {folder}/image_*')
                os.system(f'rm -rf {folder}/train_*')            

                print(colored('Done {0}'.format(year), 'green'))
                print(colored('Done {0}'.format(mes), 'green'))

                elapsed_time = time.time() - start_time
                print(colored('Spent time: {0}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))), 'yellow'))

    print(colored('Done all for region {0}.'.format(region), 'green'))

print(colored('Done all regions.', 'green'))
