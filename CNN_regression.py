from tensorflow.keras import losses
from tensorflow.keras import optimizers
import deeplake
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from keras.layers import *
from keras.models import *
import numpy as np
from datetime import datetime
from sklearn import metrics
import matplotlib.lines as lines
import wandb
import coremltools as ct

EPOCHS = 300
lr = 0.001
OPTIMIZER = tf.keras.optimizers.Adam(lr)
LOSS = 'mean_squared_error'
METRICS = [tf.keras.metrics.RootMeanSquaredError(name='rmse'), tf.keras.metrics.MeanAbsoluteError(name='mae')]
# Using Deeplake to store and load datasets
dataset_train = 'hub://earthshot-labs/DBH_Depth_Map_meters'
dataset_test = 'hub://earthshot-labs/DBH_Depth_Map_meters_test_set'
ds = deeplake.load(dataset_train)
ds_test = deeplake.load(dataset_test)
input_shape = (192, 192, 1)
weights = 'imagenet'  # None

models = ['MobileNet', 'MobileNetV2', 'DenseNet121', 'EfficientNetB0'] # < 35MB
BATCH_SIZE = 32  # 8, 16, 64
seed = random.randint(0, 10000)

def to_model_fit(item):
    x = item['depth_maps']/255
    y = item['DBH']
    return (x, y)

print(f'Batch size: {BATCH_SIZE}')
for model_type in models:
    from wandb.keras import WandbCallback
    print(f'model_type: {model_type}')
    current_date = datetime.now().strftime('%Y-%m-%d')
    model_name = f'{model_type}_1channel_Depth_map_imagenet_decimeters'
    wandb.init(project="DBH-Depth-Map-CNN-Regression-mmf-January-2023_new_test_set_Jan17_normalized_reparted", 
               name=f"Meters{model_name}_{datetime.now().month}_{datetime.now().day}_{datetime.now().year}") 

    wandb.config.model_name = model_name
    wandb.config.batch_size = BATCH_SIZE
    wandb.config.epochs = EPOCHS
    wandb.config.optimizer = OPTIMIZER
    wandb.config.loss = LOSS
    wandb.config.normalized = True

    # Dataset preparation
    wandb.config.dataset_train = dataset_train

    # Deeplake ds to tensorflow ds
    ds_tf = ds.tensorflow()
    ds_tf = ds_tf.map(lambda x: to_model_fit(x))

    image_count = len(ds)
    print(f"Images count: {image_count}")
    wandb.config.image_count = image_count

    # split train/val sets
    train_size = int(0.8 * image_count)
    val_size = int(0.2 * image_count)
    print(f"{train_size} training images and {val_size} validation images. Batch size of {BATCH_SIZE}")

    wandb.config.seed = seed
    wandb.config.shuffle = 'shuffle pre train/val split'
    ds_tf = ds_tf.shuffle(image_count, seed=seed).repeat(-1)

    val_ds = ds_tf.take(val_size)
    train_ds = ds_tf.skip(val_size)

    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)

    # Model 
    if input_shape[-1] == 1 or input_shape[-1] == 4:
        # mapping grayscale to RGB 
        inputs = tf.keras.Input(shape=input_shape)
        inputs_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(inputs)
    else:
        inputs = tf.keras.Input(shape=input_shape)
        inputs_conv = inputs

    if model_type == 'MobileNetV2':
        MobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=weights, include_top=False)
        MobileNetV2.trainable = True
        x = MobileNetV2(inputs_conv)
    elif model_type == 'MobileNet':
        MobileNet = tf.keras.applications.MobileNet(weights=weights,include_top=False)
        MobileNet.trainable = True
        x = MobileNet(inputs_conv)
    elif model_type == 'EfficientNetB0':
        EfficientNetB0 = tf.keras.applications.EfficientNetB0(weights=weights,include_top=False)
        EfficientNetB0.trainable = True
        x = EfficientNetB0(inputs_conv)
    elif model_type == 'EfficientNetB1':
        EfficientNetB1 = tf.keras.applications.EfficientNetB1(weights=weights,include_top=False)
        EfficientNetB1.trainable = True
        x = EfficientNetB1(inputs_conv)
    elif model_type == 'EfficientNetV2B0':
        EfficientNetV2B0 = tf.keras.applications.EfficientNetV2B0(weights=weights,include_top=False)
        EfficientNetV2B0.trainable = True
        x = EfficientNetV2B0(inputs_conv)
    elif model_type == 'EfficientNetV2B1':
        EfficientNetV2B1 = tf.keras.applications.EfficientNetV2B0(weights=weights,include_top=False)
        EfficientNetV2B1.trainable = True
        x = EfficientNetV2B1(inputs_conv)
    elif model_type == 'DenseNet201':
        DenseNet201 = tf.keras.applications.densenet.DenseNet201(weights=weights,include_top=False)
        DenseNet201.trainable = True
        x = DenseNet201(inputs_conv)
    elif model_type == 'DenseNet121':
        DenseNet121 = tf.keras.applications.densenet.DenseNet121(weights=weights,include_top=False)
        DenseNet121.trainable = True
        x = DenseNet121(inputs_conv)
    elif model_type == 'ResNet152V2':
        ResNet152V2 = tf.keras.applications.resnet_v2.ResNet152V2(weights=weights,include_top=False)
        ResNet152V2.trainable = True
        x = ResNet152V2(inputs_conv)
    elif model_type == 'Xception':
        xception = tf.keras.applications.xception.Xception(weights=weights,include_top=False)
        xception.trainable = True
        x = xception(inputs_conv)

        
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out_dbh = tf.keras.layers.Dense(1)(x)  

    model = tf.keras.Model(inputs=inputs, outputs=[out_dbh])

    model.compile(
            optimizer=optimizers.get(OPTIMIZER), 
            loss=losses.get(LOSS),
            metrics=METRICS)

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=26, verbose=1,
                    mode='min', restore_best_weights=True
                )
    checkpoint_folder = f'{model_type}_{BATCH_SIZE}bs_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    wandb.config.checkpoints_folder = checkpoint_folder
    print(checkpoint_folder)
    # checkpoint_filepath = os.path.join(checkpoint_folder, f"{model_type}_{BATCH_SIZE}bs_checkpoint_tree_regression_depthmap_" + current_date + ".h5")
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.99,
                                  patience=19,verbose=1)

    # Training
    model.fit(
        x=train_ds, 
        steps_per_epoch=train_size//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_ds,
        validation_steps=val_size//BATCH_SIZE,
        callbacks=[reduce_lr, early_stop, WandbCallback()]) # model_checkpoint_callback

    # Saving whole model
    # using tf here because h5 saving models in a loop doesn't work here: "ValueError: Unable to create group (name already exists)"
    model.save(os.path.join(checkpoint_folder, f"model_{model_type}_{BATCH_SIZE}bs_DBH_regression_depthmap_" + current_date + ".h5"))
    # Wandb save
    wandb.save(os.path.join(checkpoint_folder, f"model_{model_type}_{BATCH_SIZE}bs_DBH_regression_depthmap_" + current_date + ".h5"))


    # Inference / Metrics computation
    wandb.config.dataset_test = dataset_test

    ds_tf_test = ds_test.tensorflow()
    ds_tf_test = ds_tf_test.map(lambda x: to_model_fit(x))
    test_ds = ds_tf_test.batch(BATCH_SIZE)
    predictions = model.predict(test_ds)
    predictions[predictions<0] = 0  # replace negative results with 0 
    y_true_test = []
    for img, dbh in test_ds:
        for b in range(dbh.shape[0]):
            y_true_test.append(dbh[b].numpy()[0])
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rmse = round(sqrt(mean_squared_error(predictions, y_true_test)),2)

    list_error = []
    for img, dbh in test_ds:
        for b in range(0, dbh.shape[0]):
            list_error.append(abs(dbh[b].numpy()-predictions[b]))

    mse_test, root_mean_squared_error_test, mae_test = model.evaluate(test_ds)
    print(f'RMSE DBH Regression and DBH Tape: {round(sqrt(mean_squared_error(np.array(y_true_test), np.array(predictions)[:,0])),2)} cm')

    import pandas as pd
    df_DBH = pd.DataFrame({'y_true_test': np.array(y_true_test),
                          'predictions': np.array(predictions)[:,0]})

    max_value = np.max([np.max(y_true_test), np.max(predictions)])
    plt.figure(dpi=1000, figsize=(9,6))
    ax = df_DBH.plot(x = 'y_true_test',y='predictions', kind = 'scatter', figsize=(9,6), color='blue')
    line = lines.Line2D([0,np.max([np.max(y_true_test), np.max(predictions)])], [0,np.max([np.max(y_true_test), np.max(predictions)])],lw=2, color='green', axes=ax)
    ax.add_line(line)
    plt.title('DBH measured with tape versus DBH estimated with CNN')
    plt.ylabel('DBH estimated with CNN (cm)')
    plt.xlabel('DBH measured with tape (cm)')
    plt.legend(['Linear Function', 'DBH estimated with CNN'], loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(f"{checkpoint_folder}/{model_name}_{datetime.now().month}_{datetime.now().day}_{datetime.now().year}.png", dpi=100)
    wandb.log({"Scatter plot": wandb.Image(f"{checkpoint_folder}/{model_name}_{datetime.now().month}_{datetime.now().day}_{datetime.now().year}.png")})


    r2 = metrics.r2_score(y_true_test, predictions)

    wandb.log({"mean_error": np.mean(list_error),
              "mse_test": mse_test,
              "mae_test": mae_test,
              "root_mean_squared_error_test": root_mean_squared_error_test,
              "r2": r2
              })

    # Tensorflow to CoreMl Conversion
    image_input = ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE, scale=1/255.0) 

    # Set input as ImageType so CoreML can automatically resize it using Vision framework
    coreml_model = ct.convert(model, inputs=[image_input])
    print("Tensorflow model converted to CoreML.")

    # Save CoreML Model
    file_name = f"model_{model_type}_{BATCH_SIZE}bs_DBH_regression_depthmap_" + current_date + ".mlmodel"
    local_file_path = os.path.join(checkpoint_folder, file_name)
    coreml_model.save(local_file_path)
    print("Core ML model named {} saved in {}.".format(file_name, local_file_path))
    wandb.save(local_file_path)

    # TFLite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(f"{checkpoint_folder}/DBH_regression_depthmap_{current_date}.tflite", "wb").write(tflite_model)
    wandb.save(f"{checkpoint_folder}/DBH_regression_depthmap_{current_date}.tflite")

    wandb.finish()

    print("Done! Moving to next training...")
    
print("All trainings done!")
exit()
