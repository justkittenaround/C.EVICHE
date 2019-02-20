###DNN
##http://tflearn.org/models/dnn/


import numpy as np
import tensorflow as tf
import tflearn.models as models



#DNN
tflearn.models.dnn.DNN (network, clip_gradients=5.0, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/', checkpoint_path=None, best_checkpoint_path=None, max_checkpoints=None, session=None, best_val_accuracy=0.0)

#train/validate
fit (x, y, n_epoch=10, validation_set=None, show_metric=False, batch_size=None, shuffle=None, snapshot_epoch=True, snapshot_step=None, excl_trainops=None, validation_batch_size=None, run_id=None, callbacks=[])

#test
evaluate(x,y, batch_size=128)
predict (X)

#saliency
get_weights (weight_tensor)
save (model_file)
