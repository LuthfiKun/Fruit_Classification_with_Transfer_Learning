import plot
import model
import time
from keras.callbacks import EarlyStopping

def train_model(train_data, 
                val_data, 
                class_num, 
                epochs = 50, 
                batch_size = 64, 
                input_shape = (224, 224, 3), 
                optimizer='adam',
                save_plot=True):
    train_step = train_data.samples // batch_size
    val_step = val_data.samples // batch_size

    #Create model
    vgg_model = model.vgg_tf_model(input_shape, class_num, optimizer)

    #Configure early stopping
    early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

    tic = time.perf_counter() 

    #Training model
    vgg_history = vgg_model.fit(train_data,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=val_data,
                                steps_per_epoch=train_step,
                                validation_steps=val_step,
                                callbacks=[early_stop],
                                verbose=1)
                                
    toc = time.perf_counter() 
    print('Time taken to train model (s): ', toc-tic)

    tensorflow.keras.saving.save_model(vgg_model, 'model/VGG16_TF_model.h5')

    #Evaluasi model
    score = vgg_model.evaluate(validgen, verbose=0)
    print('Test accuracy:', score[1])
    print('Model saved on model dir')

    if save_plot:
        plot.plot_training()

    return vgg_model