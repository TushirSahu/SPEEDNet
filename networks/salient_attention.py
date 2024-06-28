from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Add, Multiply, Lambda, concatenate, add
import tensorflow.keras.backend as K


def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

def SalientAttentionBlock(f_maps, sal_ins, pool_maps,avg_pool, num_fmaps,number):
#     conv1_salins0,_= Involution(channel=3,  kernel_size=1, stride=1,dilation_rate=1,name=f"inv_{1+number}")(sal_ins)
#     conv1_salins1,_ = Involution(channel=3,  kernel_size=3, stride=1,dilation_rate=2,name=f"inv_{2+number}")(sal_ins)
    conv1_salins0 = Conv2D(3,(1,1), dilation_rate=1, padding = 'same')(sal_ins)
    conv1_salins1 = Conv2D(3,(3,3),dilation_rate=2, padding = 'same')(sal_ins)
#     conv1_salins  = multiply([conv1_salins1,conv1_salins0])
    conv1_salins01 = add([conv1_salins0,conv1_salins1])
    

#     conv1_salins2,_ =Involution(channel=3,  kernel_size=3, stride=1,dilation_rate=4,name=f"inv_{3+number}")(sal_ins)
    conv1_salins2 = Conv2D(3,(3,3),dilation_rate=4, padding = 'same')(sal_ins)
#     conv1_salins_  = multiply([conv1_salins1,conv1_salins])
    conv1_salins201 = add([conv1_salins01,conv1_salins2])
    
#     conv1_salins3,_= Involution(channel=3, kernel_size=3, stride=1,dilation_rate=8,name=f"inv_{4+number}")(sal_ins)
    conv1_salins3 = Conv2D(3,(3,3),dilation_rate=8, padding = 'same')(sal_ins)
#     conv1_salins_  = multiply([conv1_salins3,conv1_salins])
    conv1_salins3102 = add([conv1_salins3,conv1_salins201])
    
    conv1_salins = concatenate([conv1_salins01,conv1_salins201,conv1_salins3102,conv1_salins0])
    conv1_salins  = Conv2D(3,(1,1),activation="relu")(conv1_salins)
    conv1_fmaps0 = Conv2D(3,(1,1),activation="relu")(f_maps)
#     conv1_fmaps0 ,_= Involution(channel=3,  kernel_size=1, stride=2,dilation_rate=1,name=f"inv_{5+number}")(conv1_fmaps0)
    conv1_fmaps0 = Conv2D(3,(1,1),dilation_rate=1, strides= 2, padding = 'same')(conv1_fmaps0)
    conv1_fmaps1 = Conv2D(3,(1,1),dilation_rate=1, padding = 'same')(conv1_fmaps0)
#     conv1_fmaps1,_ = Involution(channel=3, kernel_size=3, stride=1,dilation_rate=2,name=f"inv_{6+number}")(conv1_fmaps0)
    conv1_fmaps1 = add([conv1_fmaps1,conv1_fmaps0])
#     conv1_fmaps2,_ = Involution(channel=3,  kernel_size=3, stride=1,dilation_rate=4,name=f"inv_{7+number}")(conv1_fmaps1)
    conv1_fmaps2 = Conv2D(3,(3,3),dilation_rate=4, padding = 'same')(conv1_fmaps1)
    conv1_fmaps2 = add([conv1_fmaps1,conv1_fmaps2])
    
    
    attn_add = add([conv1_fmaps2,conv1_salins])

    attn_add = Conv2D(64, (1, 1), padding='same', activation='relu',dilation_rate=2)(attn_add)
# #     attn_add, _ = Involution(channel=3,  kernel_size=3, stride=1,dilation_rate=1,name=f"inv_{8+number}")(attn_add)
    
# #     conv_1d, _ = Involution(channel=3,  kernel_size=3, stride=1,dilation_rate=1,name=f"inv_{9+number}")(attn_add)
    conv_1d = Conv2D(64, (1, 1), padding='same', activation='relu',dilation_rate=4)(attn_add)
# 
    conv_1d = Conv2D(1, (1, 1), activation='relu')(conv_1d)
    conv_1d = expend_as(conv_1d,64)
#     conv_1d = expend_as(attn_add,64)
    conv_nd = Conv2D(num_fmaps * 2, (1, 1), activation='relu')(conv_1d)
    attn_act = Activation('sigmoid')(conv_nd)
#     pool_maps = concatenate([avg_pool,pool_maps])
#     attn = multiply([attn_act, pool_maps])
    return attn_act