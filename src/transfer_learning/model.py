import tensorflow as tf


def build_model():

    base_model = tf.keras.applications.EfficientNetB0(

        include_top=False,

        weights="imagenet",

        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = tf.keras.Sequential([

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            4,
            activation="softmax"
        )

    ])

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()