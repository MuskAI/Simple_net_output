from __future__ import print_function
import os
from data_parser_down_stage_1024 import DataParser
# from EightAndDualNet_withImage import superPixNet
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
from keras import losses
import numpy as np
# from accum_optimizer import AccumOptimizer
# import pdb
# from model.rcf_model_resnet import resnet_rcf
from loss_functions import wce_huber_loss_eight,wce_huber_loss,wce_huber_loss_new,cross_entropy_balanced,pixel_error,huber_loss,acc,sensitivity,precision,specificity,f1_socre
from keras.optimizers import Adam,SGD
from keras.callbacks import ReduceLROnPlateau
from keras.losses import mean_squared_error
# from SuperAndSubPixelNet import superPixNet
# from EightAndDualNet_withImage import superPixNet,load_weights_from_hdf5_group_by_name
from U_Net_Down_stage_output import superPixNet  
from keras_preprocessing.image import ImageDataGenerator,ImageEnhance
import matplotlib.pyplot as plt
def generate_minibatches(dataParser, train=True):
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.X_train, dataParser.batch_size)
            ims, ems, double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, chanel_fuse, edgemaps_4, edgemaps_8, edgemaps_16, _ = dataParser.get_batch(
                batch_ids)
        else:
            batch_ids = np.random.choice(dataParser.X_test, dataParser.batch_size)
            ims, ems, double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, chanel_fuse, edgemaps_4, edgemaps_8, edgemaps_16, _ = dataParser.get_batch(
                batch_ids, train=False)

        # datagen.flow()

        # yield(ims,[chanel1,chanel2,chanel3,chanel4,chanel5,chanel6,chanel7,chanel8,ems,ems,ems,ems])
        yield(ims,[chanel1,chanel2,chanel3,chanel4,chanel5,chanel6,chanel7,chanel8,edgemaps_4,edgemaps_8,edgemaps_16,double_edge])


######
if __name__ == "__main__":
    # params
    K.clear_session()
    # 全局设置
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model_name = '10.25_new_out'
    model_dir     = os.path.join('checkpoints', model_name)
    csv_fn        = os.path.join(model_dir, 'train_log.csv')
    checkpoint_fn = os.path.join(model_dir,
                                 'checkpoint.{epoch:02d}-{val_loss:.4f}-{val_fuse_acc:.4f}-{val_fuse_sensitivity:.4f}-{val_fuse_precision:.4f}-{val_fuse_f1_socre:.4f}.hdf5')

    batch_size_train = 3

    # environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    # os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    if not os.path.isdir(model_dir): os.makedirs(model_dir)

    # prepare data
    dataParser = DataParser(batch_size_train)


    # model
    # model = superPixNet()
    model =superPixNet()
    # model.load_weights('/home/liu/libiao/Simple_net_output/checkpoints/10.16_new_out/checkpoint.03-0.0362-0.9964-0.9336-0.7332-0.8023.hdf5',by_name=True)
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-4)
    checkpointer = callbacks.ModelCheckpoint(period=1,mode='min',monitor='val_loss',filepath=checkpoint_fn, verbose=1, save_best_only=False)
    csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')
    tensorboard = callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, batch_size=32,
                                        write_graph=False, write_grads=True, write_images=False)
    # earlystopping_callback = callbacks.EarlyStopping(mode='min',monitor='val_loss', verbose=1, patience=1500, baseline=None)
    # callback_list = [lr_decay, checkpointer, tensorboard,csv_logger,earlystopping_callback]
    callback_list = [lr_decay, checkpointer, tensorboard, csv_logger]

    # optimizer = SGD(lr=1e-3, momentum=args["momentum"], nesterov=False)
    optimizer = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999)
    # learning_rate = 0.0001
    # optimizer=SGD(lr=learning_rate, momentum=0.9, clipnorm=5.0)
    model.compile(loss={
                        'new1_e_chanel0_1': wce_huber_loss_eight,
        'new1_e_chanel01':wce_huber_loss_eight,
        'new1_e_chanel1_1':wce_huber_loss_eight,
        'new1_e_chanel10': wce_huber_loss_eight,
        'new1_e_chanel11': wce_huber_loss_eight,
        'new1_e_chanel_1_1': wce_huber_loss_eight,
        'new1_e_chanel_10': wce_huber_loss_eight,
        'new1_e_chanel_11': wce_huber_loss_eight,
        'X_out4':wce_huber_loss_eight,
        'X_out8': wce_huber_loss_eight,
        'X_out16': wce_huber_loss_eight,
        'fuse':wce_huber_loss_eight,

                        },loss_weights={
        'new1_e_chanel0_1': 1,
        'new1_e_chanel01': 1,
        'new1_e_chanel1_1': 1,
        'new1_e_chanel10': 1,
        'new1_e_chanel11': 1,
        'new1_e_chanel_1_1': 1,
        'new1_e_chanel_10': 1,
        'new1_e_chanel_11': 1,
        'X_out4':1,
        'X_out8': 1,
        'X_out16': 1,
        'fuse':10,

                        },
        metrics={
            # 'fuse':[acc, pixel_error, sensitivity, specificity, precision, f1_socre],
            'new1_e_chanel_10': [acc, pixel_error, sensitivity, specificity, precision, f1_socre],
            'new1_e_chanel_11': [acc, pixel_error, sensitivity, specificity, precision, f1_socre],
            'X_out4': [acc, pixel_error, sensitivity, specificity, precision, f1_socre],
            'X_out8': [acc, pixel_error, sensitivity, specificity, precision, f1_socre],
            'fuse': [acc, pixel_error, sensitivity, specificity, precision, f1_socre],
                 },
                  optimizer=optimizer)
    # print(generate_minibatches(dataParser,True))
    print(dataParser)

    train_history = model.fit_generator(
                        generate_minibatches(dataParser,True),
                        # max_q_size=40, workers=1,
                        steps_per_epoch=dataParser.steps_per_epoch,  #batch size
                        epochs=1000,initial_epoch=0,
                        validation_data=generate_minibatches(dataParser, train=True),
                        validation_steps=dataParser.val_steps,
                        callbacks=callback_list,verbose=1)

