import tensorflow as tf
import random
from utils import Contrast_loss, CapsuleLayer,Query_merge
import bert


class Contrast_induction(tf.keras.models.Model):
    def __init__(self, d_model, number_class, max_query_lenth):
        super(Contrast_induction, self).__init__()
        self.d_model = d_model
        self.cls_layer = tf.keras.layers.Lambda(lambda x: x[:, 0])
#         self.dense_query = tf.keras.layers.Dense(self.d_model, activation='relu', name="dense_query")
#         self.dense_doc = tf.keras.layers.Dense(self.d_model, activation='relu')
#         self.dense_doc2 = tf.keras.layers.Dense(self.d_model, activation='relu', name="dense_doc")
        self.max_sentence_len = max_query_lenth
        self.number_class = number_class
        self.capsule = CapsuleLayer(dim_capsule=self.d_model)
#         self.capsule=Query_merge()
        self.loss_layer = Contrast_loss(num_class=self.number_class)
        self.bert = self.build_bert()

    def build_bert(self, ):
        l_input_ids = tf.keras.layers.Input(shape=(self.max_sentence_len,), dtype='int32')
        l_token_type_ids = tf.keras.layers.Input(shape=(self.max_sentence_len,), dtype='int32')
        model_dir = './tiny_roberta/'
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        output = l_bert([l_input_ids, l_token_type_ids])
        model = tf.keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
#         model.build(input_shape=(None, max_seq_len))
#         model.build(input_shape=[(None, self.max_sentence_len), (None, self.max_sentence_len)])
        return model

    def label_model(self, inputs):
        doc = self.bert(inputs)
        doc = self.cls_layer(doc)
        doc = self.capsule(doc)
#         doc = self.dense_doc2(doc)
        doc = tf.math.l2_normalize(doc, axis=-1, name="label_norm")
        return doc

    def call(self, inputs):
        char_query_input = inputs[0]
        pos_query_input = inputs[1]
        query = self.bert([char_query_input, pos_query_input])
        query = self.cls_layer(query)
        query = tf.math.l2_normalize(query, axis=-1, name="query_norm")
        doc_merge = tf.concat(
            [self.label_model([inputs[2 * i], inputs[2 * i + 1]]) for i in range(1, self.number_class + 1)], axis=0)
        loss = self.loss_layer([query, doc_merge])
        return query, doc_merge, loss

    def bulid_graph(self, ):
        input_palce = []
        input_palce.append(tf.keras.layers.Input(shape=[self.max_sentence_len, ], name="query_char"))
        input_palce.append(tf.keras.layers.Input(shape=[self.max_sentence_len, ], name="query_pos"))
        input_palce.append(tf.keras.layers.Input(shape=[self.max_sentence_len, ], name="doc_char"))
        input_palce.append(tf.keras.layers.Input(shape=[self.max_sentence_len, ], name="doc_pos"))
        for i in range(1, self.number_class):
            input_palce.append(tf.keras.layers.Input(shape=[self.max_sentence_len, ]))
            input_palce.append(tf.keras.layers.Input(shape=[self.max_sentence_len, ]))
        return tf.keras.models.Model(input_palce, self.call(input_palce))

#[512,] query_char
#[16,32,] class = [32,] [7,] [5,] [20,] 