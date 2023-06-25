import dataset
import training
import tensorflow

train_data_dir = 'image/train'
val_data_dir = 'image/valid'

with tensorflow.device('/device:GPU:0'):
    print('Generating dataset from image...')
    train_data, val_data, class_num = dataset.generate(train_data_dir, val_data_dir)
    print('Training model...')
    model = training.train_model(train_data, 
                                    val_data, 
                                    class_num, 
                                    epochs = 50)

