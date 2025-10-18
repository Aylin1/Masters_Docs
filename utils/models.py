from tensorflow.keras import layers, models


def build_unet(input_shape):
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    return models.Model(inputs, outputs)
def build_hrnet_with_attention_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder with Squeeze-and-Excitation (SE) blocks
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    c1 = squeeze_excitation_block(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    c2 = squeeze_excitation_block(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    c3 = squeeze_excitation_block(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c4)
    c4 = squeeze_excitation_block(c4)

    # Decoder
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c4)
    u3 = layers.Concatenate()([u3, c3])
    c5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u3)
    c5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c5)
    c5 = squeeze_excitation_block(c5)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u2)
    c6 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c6)
    c6 = squeeze_excitation_block(c6)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u1 = layers.Concatenate()([u1, c1])
    c7 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u1)
    c7 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c7)
    c7 = squeeze_excitation_block(c7)

    # Final Output
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c7)

    return models.Model(inputs, outputs)


# Squeeze-and-Excitation Block
def squeeze_excitation_block(input_tensor, ratio=16):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation="relu", use_bias=False)(se)
    se = layers.Dense(filters, activation="sigmoid", use_bias=False)(se)
    return layers.Multiply()([input_tensor, se])

def build_improved_fusionnet_unet(input_shape):
    inputs = layers.Input(input_shape)

    def residual_block(x, filters):
        """A simple residual block with shape matching."""
        shortcut = x
        # Adjust the shortcut if the number of filters does not match
        if x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding="same")(shortcut)

        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Add()([shortcut, x])
        return x

    # Encoder
    c1 = residual_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = residual_block(p2, 256)
    c3 = residual_block(c3, 256)

    # Decoder
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = residual_block(u1, 128)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = residual_block(u2, 64)

    # Final Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return models.Model(inputs, outputs)
