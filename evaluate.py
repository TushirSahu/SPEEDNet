from tensorflow.keras.models import load_model
from loss import *
from train import *

img_dim=224

model = load_model('/kaggle/working/model_73.h5',
                   custom_objects={'focal_tversky_loss':focal_tversky_loss,
                    'jacard_coef':jacard_coef,'diceCoef':diceCoef})
model.evaluate(nor_gen), model.evaluate(pol_gen), model.evaluate(hgr_gen),model.evaluate(lgr_gen), model.evaluate(adeno_gen), model.evaluate(ser_gen), model.evaluate(val_gen)