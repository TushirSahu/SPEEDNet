import tensorflow as tf
from tensorflow.keras.models import Model
from dataset.dataloader import getDataLoader
from loss import *
from Networks.model import build_model
from utils import *
from keras_flops import get_flops

model = build_model(8)

flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")


train_gen,val_gen,adeno_gen,nor_gen,pol_gen,ser_gen,hgr_gen,lgr_gen = getDataLoader(4)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("model_{epoch:02d}.h5", monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min'),
tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',factor = 0.1, patience = 12, min_lr = 0.00001)
]

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=diceCoefLoss, metrics=['accuracy', diceCoef, jacard_coef, keras.metrics.Precision(), keras.metrics.Recall()])
history = model.fit(train_gen, validation_data=val_gen, epochs=120, verbose=1, steps_per_epoch=100, callbacks=callbacks)
plot(history)   
FPS(model, val_gen)