from tensorflow.keras.layers import Conv2D, Activation, add, Lambda, concatenate
import tensorflow.keras.backend as K

def expand_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)

def SalientAttentionBlock(feature_maps, saliency_input, num_feature_maps):
    """
    Salient Attention Block
    input: fmaps, saliency_input, num_fmaps
    output: attention_activation
    """

    # Saliency Input Pathway
    conv1_saliency_0 = Conv2D(3, (1, 1), dilation_rate=1, padding='same')(saliency_input)
    conv1_saliency_1 = Conv2D(3, (3, 3), dilation_rate=2, padding='same')(saliency_input)
    conv1_saliency_01 = add([conv1_saliency_0, conv1_saliency_1])

    conv1_saliency_2 = Conv2D(3, (3, 3), dilation_rate=4, padding='same')(saliency_input)
    conv1_saliency_201 = add([conv1_saliency_01, conv1_saliency_2])

    conv1_saliency_3 = Conv2D(3, (3, 3), dilation_rate=8, padding='same')(saliency_input)
    conv1_saliency_3102 = add([conv1_saliency_3, conv1_saliency_201])

    conv1_saliency_combined = concatenate([conv1_saliency_01, conv1_saliency_201, conv1_saliency_3102, conv1_saliency_0])
    conv1_saliency_combined = Conv2D(3, (1, 1), activation="relu")(conv1_saliency_combined)

    # Feature Maps Pathway
    conv1_feature_maps_0 = Conv2D(3, (1, 1), activation="relu")(feature_maps)
    conv1_feature_maps_0 = Conv2D(3, (1, 1), dilation_rate=1, strides=2, padding='same')(conv1_feature_maps_0)

    conv1_feature_maps_1 = Conv2D(3, (1, 1), dilation_rate=1, padding='same')(conv1_feature_maps_0)
    conv1_feature_maps_1 = add([conv1_feature_maps_1, conv1_feature_maps_0])

    conv1_feature_maps_2 = Conv2D(3, (3, 3), dilation_rate=4, padding='same')(conv1_feature_maps_1)
    conv1_feature_maps_2 = add([conv1_feature_maps_1, conv1_feature_maps_2])

    # Attention Addition
    attention_addition = add([conv1_feature_maps_2, conv1_saliency_combined])
    attention_addition = Conv2D(64, (1, 1), padding='same', activation='relu', dilation_rate=2)(attention_addition)

    # 1D Convolution
    conv_1d = Conv2D(64, (1, 1), padding='same', activation='relu', dilation_rate=4)(attention_addition)
    conv_1d = Conv2D(1, (1, 1), activation='relu')(conv_1d)
    conv_1d_expanded = expand_as(conv_1d, 64)

    # Final Attention Map
    conv_nd = Conv2D(num_feature_maps * 2, (1, 1), activation='relu')(conv_1d_expanded)
    attention_activation = Activation('sigmoid')(conv_nd)

    return attention_activation
