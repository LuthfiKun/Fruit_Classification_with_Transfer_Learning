import matplotlib.pyplot as plt

def plot_training():
    plt.plot(vgg_history.history['accuracy'], label='Training')
    plt.plot(vgg_history.history['val_accuracy'], label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("VGG16 Accuracy")
    plt.legend()
    plt.savefig('plot/training_accuracy.png')

    plt.plot(range(1,27), vgg_history.history['loss'][1:], label='Training')
    plt.plot(range(1,27), vgg_history.history['val_loss'][1:], label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VGG16 Loss")
    plt.legend()
    plt.savefig('plot/training_loss.png')

    print('Plot saved on plot dir')