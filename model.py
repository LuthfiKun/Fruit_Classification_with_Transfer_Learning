from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten

def vgg_tf_model(input_shape, n_classes, optimizer='adam'):
    vgg_extractor = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    for layer in vgg_extractor.layers:
      layer.trainable = False

    top_model = vgg_extractor.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    model = Model(inputs=vgg_extractor.input, outputs=output_layer)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model