import tensorflow as tf
import math
import numpy as np
class BidirTrans:
    def __init__(self, config):
        self.graph = tf.Graph()
        self.NlLen = config.NlLen
        self.CodeLen = config.CodeLen
        self.WoLen = config.WoLen
        self.embedding_size = config.embedding_size
        self.Nl_Vocsize = config.Nl_Vocsize
        self.Vocsize = config.Vocsize
        self.max_step = config.max_step
        self.margin = config.margin
        self.CodeVocsize = config.Code_Vocsize
        #self.keep_prob = config.keep_prob
    def weight_nonzero(self, labels):
        return tf.to_float(tf.not_equal(labels, 0))
    def weight_zero(self, labels):
        return tf.to_float(tf.equal(labels, 0))
    def mask_from_embbeding(self, emb):
        return self.weight_nonzero(tf.reduce_sum(tf.abs(emb), asix=3, keep_dims=True))
    def layer_norm(self, vec, na=None, axis=2):
        return tf.contrib.layers.layer_norm(vec, scope=na, begin_norm_axis=axis, reuse=None)
    def drop(self, input):
        return tf.nn.dropout(input, self.keep_prob)
    def max_height_pooling(self, input):
        height = int(input.get_shape()[1])
        width = int(input.get_shape()[2])
        input = tf.expand_dims(input, -1)
        output = tf.nn.max_pool(input, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output
    def headAttention(self, Q, K, V, mask):
        d = int(Q.shape[2])
        d = math.sqrt(float(d))
        matrix = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / d
        mask = tf.expand_dims(mask, -2)
        a = matrix * mask
        ma = self.weight_zero(a) * (-1e18)
        a += ma
        a = tf.nn.softmax(a)
        a *= mask
        return tf.matmul(a, V)
    def multiheadAttention_QKV(self, Q, K, V, mask, name):
        m = int(V.shape[1])
        d = int(V.shape[2])
        list_concat = []
        heads = 8
        for i in range(heads):
            W_q = tf.layers.dense(Q, d//heads, name=name + "QKV_Wq" + str(i), use_bias=False)
            W_k = tf.layers.dense(K, d//heads, name=name + "QKV_Wk" + str(i), use_bias=False)
            W_v = tf.layers.dense(V, d//heads, name=name + "QKV_Wv" + str(i), use_bias=False)
            list_concat.append(self.headAttention(W_q, W_k, W_v, mask))
        concat_head = tf.concat(list_concat, -1)
        W_o = tf.layers.dense(concat_head, d, name=name + "res_Att", use_bias=False)
        return W_o
    def get_timing_signal_1d(self, length,
                             channels,
                             min_timescale=1.0,
                             max_timescale=1.0e4,
                             start_index=0):
          position = tf.to_float(tf.range(length) + start_index)
          num_timescales = channels // 2
          log_timescale_increment = (
              math.log(float(max_timescale) / float(min_timescale)) /
              tf.maximum(tf.to_float(num_timescales) - 1, 1))
          inv_timescales = min_timescale * tf.exp(
              tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
          scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
          signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
          signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
          signal = tf.reshape(signal, [1, length, channels])
          return signal
    def multiheadCombine_QKV(self, Query, Keys, Values):
        d = int(Query.shape[2])
        list_concat = []
        heads = 8
        qd = math.sqrt(float(d // heads))
        for i in range(heads):
            W_q = tf.layers.dense(Query, d // heads, name="qkv2headq" + str(i),
                                  use_bias=False)  # self.weight_variable(shape=[d, d // k])
            W_kv = tf.layers.dense(Keys, d // heads, name="qkv2headkv" + str(i), use_bias=False)
            W_k = tf.layers.dense(Keys, d // heads, name="qkv2headk" + str(i),
                                  use_bias=False)  # self.weight_variable(shape=[d, d // k])
            W_v = tf.layers.dense(Values, d // heads, name="qkv2headv" + str(i),
                                  use_bias=False)  # self.weight_variable(shape=[d, d // k])
            W_vv = tf.layers.dense(Values, d // heads, name="qkv2headvv" + str(i),
                                   use_bias=False)  # self.weight_variable(shape=[d, d // k])
            QK = tf.reduce_sum(W_q * W_k, -1, keepdims=True) / qd
            QV = tf.reduce_sum(W_q * W_v, -1, keepdims=True) / qd
            QK_1 = QK - tf.maximum(QK, QV)
            QV_1 = QV - tf.maximum(QK, QV)
            self.probe = QV
            QK = tf.exp(QK_1)
            QV = tf.exp(QV_1)
            QK_S = QK / (QK + QV)
            QV_S = QV / (QK + QV)
            QK_S *= W_kv
            QV_S *= W_vv
            list_concat.append(QK_S + QV_S)
        concat_head = tf.concat(list_concat, -1)
        W_o = tf.layers.dense(concat_head, d, name="qkv2head",
                              use_bias=False)
        return W_o
    def sepconv(self, state, size, mask):
        state = self.drop(tf.layers.separable_conv1d(tf.expand_dims(mask, -1) * self.drop(tf.layers.separable_conv1d(state, size, 3, activation=self.gelu, padding="SAME", name="conv")), size, 3, padding="SAME", name="dense_2") + state)
        return state
    def gelu(self, x):
        #return tf.nn.tanh(x)
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf
    def encoder_left(self, inputState, mask, name, em_Char):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            state = inputState
            for i in range(3):
                with tf.variable_scope("Left" + str(i), reuse=tf.AUTO_REUSE):
                    state += self.get_timing_signal_1d(tf.shape(state)[1], tf.shape(state)[2]) + self.get_timing_signal_1d(tf.shape(state)[1], tf.shape(state)[2], start_index=i)
                    state = self.layer_norm(self.drop(self.multiheadAttention_QKV(state, state, state, mask, "Left1") + state), "LeftNorm1")
                    #char em
                    state = self.layer_norm(self.drop(self.multiheadCombine_QKV(state, state, em_Char) + state), "LeftNorm2")
                    state *= tf.expand_dims(mask, -1)
                    state = self.sepconv(state, self.embedding_size, mask)
                    state = self.layer_norm(state, "LeftNorm3")
            return state
    def encoder_right(self, inputState, mask, name, em_Char, inputLeft, leftmask):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            state = inputState
            for i in range(1):
                with tf.variable_scope("Right" + str(i), reuse=tf.AUTO_REUSE):
                    state += self.get_timing_signal_1d(tf.shape(state)[1], tf.shape(state)[2]) + self.get_timing_signal_1d(tf.shape(state)[1], tf.shape(state)[2], start_index=i)

                    state = self.layer_norm(self.drop(self.multiheadAttention_QKV(state, state, state, mask, "right1") + state), "RightNorm1")
                    # char em
                    state = self.layer_norm(self.drop(self.multiheadCombine_QKV(state, state, em_Char) + state), "RightNorm2")
                    state = self.layer_norm(self.drop(self.multiheadAttention_QKV(state, inputLeft, inputLeft, leftmask, "right2") + state), "RightNorm3")
                    state *= tf.expand_dims(mask, -1)
                    state = self.sepconv(state, self.embedding_size, mask)
                    state = self.layer_norm(state, "RightNorm4")
            return state
    def getCos(self, q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.div(mul, tf.multiply(q1, a1))
        return cosSim


    def getLoss(self, posSim, negSim, margin):
        zero = tf.fill(tf.shape(posSim), 0.0)
        tfMargin = tf.fill(tf.shape(negSim), margin)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(posSim, negSim)))
            loss = tf.reduce_mean(losses)
        return loss

    def AveragePool(self, state, mask):
        validNum = tf.reduce_sum(mask, reduction_indices=[-1])
        mask = tf.to_float(tf.equal(validNum, 0))
        validNum += mask
        validNum = tf.expand_dims(validNum, axis=-1)
        stateSum = tf.reduce_sum(state, reduction_indices=[-2])
        stateSum = tf.div(stateSum, validNum)
        return stateSum
    def transEmModel(self, inputNl, inputCode, inputNlCharEm, inputCodeCharEm, NlMask, CodeMask, inputNlOverlap, inputCodeOverlap):
        #em_Nl = tf.nn.embedding_lookup(self.embedding, inputNl)
        em_Nl = Nl_overlap = tf.nn.embedding_lookup(self.overlap_embedding, inputNlOverlap)
        #em_Nl = tf.concat([em_Nl, Nl_overlap], axis=-1)

        #em_Code = tf.nn.embedding_lookup(self.Code_embedding, inputCode)
        em_Code = Code_overlap = tf.nn.embedding_lookup(self.overlap_embedding, inputCodeOverlap)
        #em_Code = tf.concat([em_Code, Code_overlap], axis=-1)

        em_Nl_Char = tf.nn.embedding_lookup(self.char_embedding, inputNlCharEm)
        em_Code_Char = tf.nn.embedding_lookup(self.char_embedding, inputCodeCharEm)
        with tf.variable_scope("char_embedding", reuse=tf.AUTO_REUSE):
            em_Nl_Char = self.drop(tf.layers.conv2d(em_Nl_Char, self.embedding_size, [1, 3], name="Nl1", padding="SAME"))
            em_Nl_Char = self.drop(tf.layers.conv2d(em_Nl_Char, self.embedding_size, [1, 5], name="Nl2", padding="SAME"))
            em_CharConv_Nl = self.drop(tf.layers.conv2d(em_Nl_Char, self.embedding_size, [1, self.WoLen], name="Nl"))
            em_CharConv_Nl = self.layer_norm(tf.reduce_max(em_CharConv_Nl, reduction_indices=[-2]), "LayerNorm1")
            em_Code_Char = self.drop(tf.layers.conv2d(em_Code_Char, self.embedding_size, [1, 3], name="Code1", padding="SAME"))
            em_Code_Char = self.drop(tf.layers.conv2d(em_Code_Char, self.embedding_size, [1, 5], name="Code2", padding="SAME"))
            em_CharConv_Code = self.drop(tf.layers.conv2d(em_Code_Char, self.embedding_size, [1, self.WoLen], name="Code"))
            em_CharConv_Code = self.layer_norm(tf.reduce_max(em_CharConv_Code, reduction_indices=[-2]), "LayerNorm2")
            leftOutput = self.encoder_left(em_Nl, NlMask, "nl", em_CharConv_Nl)
            leftOutput1 = self.encoder_left(em_Code, CodeMask, "code", em_CharConv_Code)
            att1 = self.layer_norm(self.multiheadAttention_QKV(leftOutput, leftOutput1, leftOutput1, CodeMask, "att1"))
            att2 = self.layer_norm(self.multiheadAttention_QKV(leftOutput1, leftOutput, leftOutput, NlMask, "att2"))
            
            tmp = tf.layers.conv1d(leftOutput, 256, 3)
            leftOutput = self.max_height_pooling(tmp)#self.AveragePool(leftOutput, CodeMask)
            tmp = tf.layers.conv1d(leftOutput1, 256, 3)
            rightOutput = self.max_height_pooling(tmp)
            tmp = tf.layers.conv1d(att1, 256, 3)
            leftOutputcmb = self.max_height_pooling(tmp)#self.AveragePool(leftOutput, CodeMask)
            tmp = tf.layers.conv1d(att2, 256, 3)
            rightOutputcmb = self.max_height_pooling(tmp)
            #self.AveragePool(rightOutput, NlMask)#self.max_height_pooling(rightOutput)
            #leftOutput = self.encoder_left(em_Code, CodeMask, "CNLeft", em_CharConv_Code)
            #rightOutput = self.encoder_right(em_Nl, NlMask, "CNRight", em_CharConv_Nl, leftOutput, CodeMask)
            #Code_Nl = self.AveragePool(rightOutput, NlMask)#self.max_height_pooling(rightOutput)
            all = tf.concat([leftOutput, rightOutput, leftOutputcmb, rightOutputcmb], -1)
            all = tf.layers.dense(all, 1024)
            all = self.drop(all)
            all = tf.layers.dense(all, 2)
            return all#self.getCos(Nl_Code, Code_Nl)

    def build(self):
        with self.graph.as_default():
            self.keep_prob = tf.placeholder(tf.float32)
            self.inputNl = tf.placeholder(tf.int32, shape=[None, self.NlLen])
            self.inputNl_Overlap = tf.placeholder(tf.int32, shape=[None, self.NlLen])
            self.inputNl_Overlap_Neg = tf.placeholder(tf.int32, shape=[None, self.NlLen])

            self.inputCode_Overlap = tf.placeholder(tf.int32, shape=[None, self.CodeLen])
            self.inputCode = tf.placeholder(tf.int32, shape=[None, self.CodeLen])
            self.inputCodeNeg = tf.placeholder(tf.int32, shape=[None, 2])
            self.inputCode_Overlap_Neg = tf.placeholder(tf.int32, shape=[None, self.CodeLen])

            self.inputNlChar = tf.placeholder(tf.int32, shape=[None, self.NlLen, self.WoLen])
            self.inputCodeChar = tf.placeholder(tf.int32, shape=[None, self.CodeLen, self.WoLen])
            self.inputCodeCharNeg = tf.placeholder(tf.int32, shape=[None, self.CodeLen, self.WoLen])

            self.NlMask = self.weight_nonzero(self.inputNl)
            self.CodeMask = self.weight_nonzero(self.inputCode)
            self.CodeMaskNeg = self.weight_nonzero(self.inputCodeNeg)

            #self.embedding = tf.get_variable("embedding", [self.Nl_Vocsize, self.embedding_size - 5], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
            #self.Code_embedding = tf.get_variable("code_embedding", [self.CodeVocsize, self.embedding_size - 5], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
            self.char_embedding = tf.get_variable("char_embedding", [self.Vocsize, self.embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
            self.overlap_embedding = tf.get_variable("overlap_embedding", [111, self.embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
            #self.code_embedding = tf.get_variable("code_embedding", [self.CodeVocsize, self.embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))

            with tf.name_scope("possSim"):
                self.positiveSim = self.transEmModel(self.inputNl, self.inputCode, self.inputNlChar, self.inputCodeChar, self.NlMask, self.CodeMask, self.inputNl_Overlap, self.inputCode_Overlap)
                self.result = tf.nn.softmax(self.positiveSim)
                correct_prediction = tf.equal(tf.argmax(self.result, 1), tf.argmax(self.inputCodeNeg, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                tf.summary.scalar("possSim", accuracy)
            '''with tf.name_scope("negSim"):
                self.negativeSim = self.transEmModel(self.inputNl, self.inputCodeNeg, self.inputNlChar, self.inputCodeCharNeg, self.NlMask, self.CodeMaskNeg, self.inputNl_Overlap_Neg, self.inputCode_Overlap_Neg)
                tf.summary.scalar("negSim", tf.reduce_mean(self.negativeSim, reduction_indices=[-1]))
            with tf.name_scope("loss"):
                self.loss = self.getLoss(self.positiveSim, self.negativeSim, self.margin)
                tf.summary.scalar("loss", self.loss)'''

            self.loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.inputCodeNeg, logits=self.positiveSim), reduction_indices=[-1])
            tf.summary.scalar("loss", self.loss)
            self.merge = tf.summary.merge_all()
            self.optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
