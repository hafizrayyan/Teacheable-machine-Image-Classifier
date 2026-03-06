from tensorflow.keras.preprocessing.image import ImageDataGenerator
  
def load_data(data_dir):
    datagen = ImageDataGenerator(rescale = 1./255 , 
        validation_split = 0.2,
        rotation_range=30,
        zoom_range=0.3,
        brightness_range=[0.5, 1.5],
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")
    
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size = (128,128),
        batch_size = 32 ,
        class_mode = "categorical",
        subset = "training"
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size = (128,128),
        batch_size = 32 ,
        class_mode = "categorical",
        subset = "validation"
    )

    return train_data , val_data
