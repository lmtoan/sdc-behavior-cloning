import json

from utils import *

from sklearn.model_selection import train_test_split

from keras.models import Sequential # Import model class
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten # Import layers
from keras.optimizers import Adam # Import optimizer
from keras.callbacks import ModelCheckpoint # Import checkpoint


def create_model():
    """Create Keras model
    
    According to Nvidia literature
    """
    
    # Starting shape: (_, 160, 320, 3)
    # Neuron layers sequence: (3, 24, 36, 48, 64)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=INPUT_SHAPE, name='Normalization')) # Standardize, no need for sklearn Standardizer?
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu', name='Conv_L1')) 
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu', name='Conv_L2'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu', name='Conv_L3'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu', name='Conv_L4')) # Reduce stride to have more fine-grain detail
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu', name='Conv_L5'))
    model.add(Dropout(rate=0.5, name='Dropout_L1'))
    model.add(Flatten(name='Flatten')) # Fully-connected layers
    model.add(Dense(units=100, activation='elu', name='Dense_L1'))
    model.add(Dense(units=50, activation='elu', name='Dense_L2'))
    model.add(Dense(units=10, activation='elu', name='Dense_L3'))
    model.add(Dense(units=1, name='Dense_L4'))
    
    print(model.summary())
    
    return model


def train_model(model, optimizer, train_gen, val_gen, verbose=True, **config):
    """Perform training operations given defined model"""
    
    storage_path = config.get('model_dir', 'models')
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
        
    # Checkpoint
    checkpoint = ModelCheckpoint(os.path.join(storage_path, '%s_{epoch:03d}.h5' %config.get('model_name', str(time.time()))), monitor='val_loss',
                                 save_best_only=config.get('save_best_only', True), 
                                 verbose=int(verbose), mode='auto') # Auto infer min_loss or max_acc of val set
    
    if optimizer is None:
        model.compile(optimizer='adagrad', loss='mse')
    else:
        model.compile(optimizer, loss='mse')
    
    history = model.fit_generator(train_gen, steps_per_epoch=config.get('train_steps', 50), epochs=config.get('num_epoch', 1),
                        validation_data=val_gen, validation_steps=config.get('val_steps', 10),
                        callbacks=[checkpoint], max_queue_size=config.get('max_queue_size', 10),
                        verbose=int(verbose),
                        workers=config.get('workers', 1), use_multiprocessing=config.get('multi_processing', False))
    
    return history


def main(data_dir, log_path, config):
    """Main thread"""
    
    log_df = pd.read_csv(log_path)
    
    X_train, X_valid, y_train, y_valid = train_test_split(log_df.loc[:, IMAGE_COLS], log_df.loc[:, STEER_COLS], test_size=0.2)
    num_train = len(X_train)
    num_valid = len(X_valid)
    
    print("Num_train = {0}. Num_valid = {1}".format(num_train, num_valid))
    
    if 'train_steps' not in config:
        config['train_steps'] = num_train // config.get('batch_size', 64)
    if 'val_steps' not in config:
        config['val_steps'] = num_valid // config.get('batch_size', 64)
    
    augment_pipeline = config.get('augment_pipeline', False)
    if augment_pipeline:
        train_gen = batch_generator(data_dir, X_train, y_train, is_training=True, **config)
    else:
        train_gen = batch_generator(data_dir, X_train, y_train, is_training=False, **config)

    valid_gen = batch_generator(data_dir, X_valid, y_valid, is_training=False, **config)
    
    model = create_model()
    
    optimizer = Adam(lr=config.get('lr', 0.0001))
    
    stat = train_model(model, optimizer, train_gen, valid_gen, **config)
    
    return stat


if __name__ == '__main__':
    """ Main thread
    
    Usage: python model.py <json_config_path>
    """
    
    if len(sys.argv) < 2:
        raise Exception("Please enter either config.json, 1, or 0")

    if '.json' in sys.argv[1]:
        try:
            print("Reading config from %s" %sys.argv[1])
            config_dict = json.load(sys.argv[1])
        except:
            raise Exception("Can't load json dictionary")
    elif sys.argv[1] == '1':
        config_dict = {
            'data': 'data/0929_data', 'augment_pipeline': True, 'model_dir': 'models',
            'batch_size': 64, 'correction': 0.2, 'model_name': '1006_0929_augment', 'num_epoch': 20, 
            'train_steps': 2000, 'val_steps': 50, 'save_best_only': False
        }
    else:
        config_dict = {
            'data': 'data/0929_data', 'augment_pipeline': False, 'model_dir': 'models',
            'batch_size': 64, 'correction': 0.2, 'model_name': '1006_0929_no_augment', 'num_epoch': 20, 
            'train_steps': 2000, 'val_steps': 50, 'save_best_only': False
        }

    print(config_dict)
    data_dir = config_dict['data']
    log_path = os.path.join(data_dir, 'driving_log.csv')
    
    print("Proceed...")
    _ = main(data_dir, log_path, config_dict)