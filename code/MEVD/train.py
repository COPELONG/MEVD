

# import tensorflow as tf
# from tensorflow import keras
import tensorflow.keras as keras
# import tensorflow._api.v1.keras as keras
from keras.callbacks import CSVLogger
from keras.layers import Dense, concatenate,Flatten
from keras.models import Model
from keras.optimizers import Adam
from get_train_test import get_train_test
from get_dataset import get_df_1
from get_dataset import get_df_2,get_df_3
from Parser import parameter_parser
#from models.cnn import get_CNN
# from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model
import datetime
#from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

#使用基础模块
from models.gru import get_gru
from models.lstm import get_lstm
from models.blstm import get_blstm
from models.BiGRU import get_bigru

#使用编码器+BLSTM模块
from models.ENBlstm import get_encoder_with_blstm

#使用BLSTM+ATT模块
from models.blstm_attention import get_blstm_attention

#使用编码器+BLSTM+ATT模块
from models.En_Blstm_ATT import get_encoder_with_blstm_att

#使用CNN+Resnet模块
from models.CNN_Resnet import get_cnn_with_resnet

#使用PCA降维+CNN模块
from models.cnn_pca import get_cnn_with_pca

#SFN+BTE+DRSN模块
from models.Re_Trans_cnn import get_re_trans_cnn

#SFN+DCE+DRSN模块
from models.Detail_cnn import get_detail_cnn

#去除SFN模块
from models.No_SFN_Detail_cnn import get_No_SFN_detail_cnn
from models.No_SFN_Trans_cnn import get_No_SFN_Trans_cnn


#去除DRSN模块
from models.No_DRSN_Detail_cnn import get_No_DRSN_detail_cnn
from models.No_DRSN_Trans_cnn import get_No_DRSN_Trans_cnn

#去除双分支模块
from models.No_Two_Detail_cnn import get_No_Two_detail_cnn
from models.No_Two_Trans_cnn import get_No_Two_trans_cnn

import numpy as np
import time
import datetime
import keras.backend as K
import tensorflow as tf
from keras.layers import Input
from sklearn.decomposition import PCA
import numpy as np
from keras.layers import Add
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU,Activation,SpatialDropout1D, Lambda,GlobalAveragePooling1D
import os
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示ERROR级别的日志消息


num_classes = 2
args = parameter_parser()



def ClassiFilerNet(INPUT_SIZE, TIME_STEPS):
    # 左边输入
    #完整模型input1 = get_re_trans_cnn()
    input1 = get_re_trans_cnn()

    #去除SFN
    #input1 = get_No_SFN_Trans_cnn()

    #去除DRSN
    #input1 = get_No_DRSN_Trans_cnn()

    #去除TWO
    #input1 = get_No_Two_trans_cnn()
    input1_name = input1.name

    #消融实验：各种变种基础模块对比
    #input1 = get_cnn_with_pca()
    #input1 = get_cnn()
    #input1 = get_cnn_with_resnet()
    
    #input2 = get_blstm_with_encoder_with_ATT(INPUT_SIZE, TIME_STEPS)
    #input1 = get_lstm()
    #input1 = merge_model(INPUT_SIZE, TIME_STEPS)
    #------------------------------------------------------------------------------------------
    # 右边输入
    
    #input2 = get_bigru(INPUT_SIZE, TIME_STEPS)
    
    #完整模型input2 = get_detail_cnn()
    input2 = get_detail_cnn()

    #去除SFN
    #input2 = get_No_SFN_detail_cnn()

    #去除DRSN
    #input2 = get_No_DRSN_detail_cnn()

    #去除TWO
    #input2 = get_No_Two_detail_cnn()

    #消融实验：各种变种基础模块对比

    #input2 = get_blstm(INPUT_SIZE, TIME_STEPS)
    #input2 = get_encoder_with_blstm_att(INPUT_SIZE, TIME_STEPS)
    #input2 = get_blstm_with_encoder_with_ATT(INPUT_SIZE, TIME_STEPS)
    #input2 = get_blstm_attention(INPUT_SIZE, TIME_STEPS)
    input2_name = input2.name
    for layer in input2.layers:
        layer._name = layer.name + str("_2")
    inp1 = input1.input
    inp2 = input2.input

    print("input1.output shape:", input1.output.shape)
    print("input2.output shape:", input2.output.shape)

    # 绘制每个模型结构
    # model = get_cnn_with_resnet()
    # tf.keras.utils.plot_model(model, to_file='resnet_model.png', show_shapes=True)
    merge_layers = concatenate([input1.output, input2.output])

    #fc1 = Dense(300, activation='relu',name="dense_a")(merge_layers)
    #fc2 = Dense(300, activation='relu',name="dense_b")(fc1)
    


    # #Re_Trans_cnn模块和Detail_cnn模块融合，随时修改filters=300
    # #因为此时向量形状都是（none，100，300）
    # # 1x1 卷积层用来调整特征通道数
    # merge_layers_1x1 = Conv1D(filters=600, kernel_size=1, strides=1, padding='same', activation='relu')(merge_layers)
    
    # #对DRSN输入也是三维形状：比如（none，100，300）格式，否则需要调整
    # #-----------------------------DRSN软阈值模块----------------------------------#
    # residual = merge_layers
    # residual_abs = Lambda(abs)(merge_layers)
    # abs_mean = GlobalAveragePooling1D()(residual_abs)

    # channels = merge_layers.get_shape().as_list()[-1]

    # scales = Dense(channels, activation=None, kernel_initializer='he_normal',
    #                    kernel_regularizer=l2(1e-4))(abs_mean)


    # scales = BatchNormalization()(scales)
    # scales = Activation('relu')(scales)
    # scales = Dense(channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)

    # thres = keras.layers.multiply([abs_mean, scales])

    # sub = keras.layers.subtract([residual_abs, thres])
    # zeros = keras.layers.subtract([sub, sub])
    # n_sub = keras.layers.maximum([sub, zeros])
    # residual = keras.layers.multiply([Lambda(K.sign)(residual), n_sub]) 
    # residual = Activation('relu')(residual)  
    # merged_res = Add()([merge_layers_1x1, residual])

    # #例如：(none，100，300)--->(none,30000)
    # merged_res = Flatten()(merged_res) 
    # #--------------------------------------------------------------------#
    
    

    #fc1 = Dense(512, name="dense_a")(merged_res)
    fc1 = Dense(512, name="dense_a")(merge_layers)
    bn1 = BatchNormalization()(fc1)
    leakyrelu1 = LeakyReLU(alpha=0.01)(bn1)  # alpha是负值范围的斜率

    # 减少损失信息并加入LeakyReLU
    fc2 = Dense(256, activation='linear', name="dense_b")(leakyrelu1)
    bn2 = BatchNormalization()(fc2)
    leakyrelu2 = LeakyReLU(alpha=0.01)(bn2)

    fc3 = Dense(num_classes, activation='softmax',name="dense_c")(leakyrelu2)
   
    class_models = Model([inp1, inp2], [fc3])

     
    model_name = input1_name + input2_name+ '_best_model.h5'
    return class_models,model_name


import os

def train():
    df1, base = get_df_1()#word2Vec模型
    #df2, BASE2 = get_df_2()#FastText模型
    #df3, BASE3 = get_df_3()#Doc2Vec模型
    x_train, x_test, y_train, y_test = get_train_test(df1)
    #x_train_2, x_test_2, y_train_2, y_test_2 = get_train_test(df2)
    #x_train_3, x_test_3, y_train_3, y_test_3 = get_train_test(df3)
    model,model_name = ClassiFilerNet(x_train.shape[1], x_train.shape[2])        # 模型确定
    
    Png_filename = model_name + ".png"
    plot_model(model, to_file=Png_filename)
    #plot_model(model, to_file="model.png")
    print(model.summary())


    adam = Adam(lr=args.lr)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 创建log文件夹（如果不存在）
    if not os.path.exists('log'):
     os.makedirs('log')

    # 日志文件路径
    log_file_path = 'log/' + base + '_log.txt'

    #csv_logger = CSVLogger('log\\' + base + '_log.txt', append=True, separator=',')
    csv_logger = CSVLogger('log/' + base + '_log.txt', append=True, separator=',')

    dataset_name = os.path.basename(args.dataset).split('.')[0]
    
    save_dir = os.path.join("model_h5", dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    model_name = os.path.join(save_dir, model_name)

    # 定义回调函数，保存每个时期中最佳的权重
    checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    # history = model.fit(
    #     x=[x_train, x_train_2],
    #     y=y_train,
    #     validation_data=([x_test, x_test_2], y_test),
    #     batch_size=args.batch_size,
    #     epochs=args.epochs,
    #     callbacks=[csv_logger, checkpoint]
    # )
    start = time.perf_counter()
    history = model.fit(
        x=[x_train, x_train],
        y=y_train,
        validation_data=([x_test, x_test], y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[csv_logger, checkpoint]
    ) 
    end = time.perf_counter()
    train_time = datetime.timedelta(seconds=end - start) 

    
    model.load_weights(model_name)
    print("加载完毕权重文件")
    # loss, accuracy = model.evaluate([x_test, x_test_2],
    #                                 y_test,
    #                                 batch_size=y_test.shape[0],
    #                                 verbose=False)
    start = time.perf_counter()
    loss, accuracy = model.evaluate([x_test, x_test],
                                    y_test,
                                    batch_size=y_test.shape[0],
                                    verbose=False)
    end = time.perf_counter()
    test_time = datetime.timedelta(seconds=end - start)  
                                   
    print("test loss:", str(loss))

    print("test accuracy", str(accuracy))
    
    # predictions = (model.predict([x_test, x_test_2], batch_size=y_test.shape[0])).round()
    predictions = (model.predict([x_test, x_test], batch_size=y_test.shape[0])).round()
    tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
    print('tn :',tn)
    print('fp :',tn)
    print('fn :',tn)
    print('tp :',tn)

    print('False positive rate(FP): ', fp / (fp + tn))
    print('False negative rate(FN): ', fn / (fn + tp))
    recall = tp / (tp + fn)
    print('Recall: ', recall)
    precision = tp / (tp + fp)
    print('Precision: ', precision)
    print('F1 score: ', (2 * precision * recall) / (precision + recall))
    print("训练时间：", train_time)
    print("测试时间：", test_time)




    with open(log_file_path, mode='a') as f:
        f.write("df: " + base + '\n')
        f.write("Loss:" + str(loss) + "\n")
        f.write("Accuracy:" + str(accuracy) + "\n")
        f.write('False positive rate(FP): ' + str(fp / (fp + tn)) + "\n")
        f.write('False negative rate(FN): ' + str(fn / (fn + tp)) + "\n")
        f.write('Recall: ' + str(recall) + "\n")
        f.write('Precision: ' + str(precision) + "\n")
        f.write('F1 score: ' + str((2 * precision * recall) / (precision + recall)) + "\n")

        # f.write('训练时间：', str(train_time) + "\n")
        # f.write('测试时间：', str(test_time) + "\n")
        
        f.write("-------------------------------" + "\n")
  


if __name__ == '__main__':
    for i in range(4):
      print("目前循环次数：",i+1)
      train()
