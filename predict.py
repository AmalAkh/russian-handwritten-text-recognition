from tensorflow.keras.applications import VGG16

import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import ResNet50
import tensorflow.keras
import os
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Bidirectional, LSTM, Reshape, Dropout
import numpy as np
import json
def load_image(path):

    img = tf.io.read_file(path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_image(img, channels=3)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [50, 200])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    img = img.numpy()
    return img

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
vocab = dict()


path_to_images = "your path here"

vocab = [ "\u0425", "!", "\u043b", "N", "\u0414", "c", "\u041a", "'", "a", "5", "6", "s", "\u044b", "\u0417", "\u044e", "\u0445", ":", "\u041e", "\u0422", "\u0449", "\u0401", " ", "\u043a", "\u0441", "=", "+", "\u0432", "\u0426", "\u0444", "\u0447", "\u042b", "[", "\u0418", "B", "\u0433", "4", "\u0435", "\u0443", "7", "?", "\u044a", ")", "\u0442", "\u044c", "\u0427", "\u0424", "\u0411", "\u0437", "\u043c", "\u041c", "I", "O", "9", "\u0416", "\u042e", "}", "\u0429", "\u043d", "n", "3", ",", "\u0439", "\u044f", "]", "\u041f", "\u0438", "\u2116", "\u0421", "\"", "t", "V", "(", "\u043f", "\u0440", "e", "l", "r", "\u0448", "\u0431", "M", "/", "\u0415", "2", "\u042d", "\u0434", "\u0436", "_", "\u042f", "|", "\u0410", "0", "\u041b", "\u0420", "8", ";", "1", "-", "<", "\u0451", "\u0430", "z", "\u044d", "b", "\u0423", "\u0446", "\u0428", "\u0412", "\u043e", ">", ".", "\u041d", "\u0413", "T", "p", "*", "k", "y", "F", "A", "H", "u", "v", "g", "K", "f", "D", "d", "R", "L", "q", "\u042c", "Y", "X", "C", "i", "o", "S", "J", "G", "%", "w", "x", "U", "E", "j", "h", "m", "W", "P"]




images = []
normal_images = []
for path in os.listdir(path_to_images):
    if ".png" in path:
        images.append(load_image(os.path.join(path_to_images, path)))
        normal_images.append(load_image(os.path.join(path_to_images, path)))

images = np.array(images)



char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocab, mask_token=None)
num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True,mask_token=None)





vgg = VGG16(include_top=False, input_shape=(200,50,3))

conv1 = vgg.get_layer("block1_conv1")
conv2 = vgg.get_layer("block1_conv2")
pool1 = vgg.get_layer("block1_pool")

conv3 = vgg.get_layer("block2_conv1")
conv4 = vgg.get_layer("block2_conv2")
pool2 = vgg.get_layer("block2_pool")


img_input = Input(shape=(200, 50, 3), name="image_input",dtype="float32" )
lbl_input = Input(shape=(None,),dtype="float32")

x = conv1(img_input)

x = conv2(x)

x = pool1(x)

x = layers.BatchNormalization()(x)




x = conv3(x)
x = conv4(x)



x = pool2(x)


x = layers.BatchNormalization()(x)





x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(x)

x = layers.BatchNormalization()(x)

x = Reshape(((200 // 4), (50 // 4) * 64))(x)

x = Dense(64, activation="relu",kernel_initializer="he_normal")(x)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)

x = Dense(151, activation="softmax", name="target_dense")(x)
output = CTCLayer()(lbl_input, x)

model = Model([img_input, lbl_input], output)
class CERMetric(tf.keras.metrics.Metric):

    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(len(y_true))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_states(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)

model.compile(optimizer=tf.keras.optimizers.Adam())
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model.summary()

model.load_weights("best-model.h5")




prediction_model = tf.keras.models.Model(
    model.get_layer(name="image_input").input, model.get_layer(name="target_dense").output
)
prediction_model.summary()


prs = prediction_model.predict(images)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :23
    ]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res.replace("[UNK]", ""))
    return output_text
pred_texts = decode_batch_predictions(prs)

print(pred_texts)


