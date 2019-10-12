from Model import *
import os
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':20,
    'CodeLen':120,
    'batch_size':320,
    'embedding_size':256,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100
})
class ModelWrapper:
    def __init__(self, sessConfig):
        self.model = BidirTrans(args)
        self.model.build()
        self.sess = tf.Session(config=sessConfig, graph=self.model.graph)
        with tf.Session() as tmp_sess:
            tmp_sess.run(tf.global_variables_initializer())
            self.sess.run(tf.variables_initializer(self.model.graph.get_collection('variables')))
        self.saver = None
    def save_checkpoint(self, folder='checkpoint', filename="save.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver == None:
            self.saver = tf.train.Saver(self.model.graph.get_collection('variables'))
        with self.model.graph.as_default():
            self.saver.save(self.sess, filepath)
    def load_checkpoint(self, folder='checkpoint', filename="save.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath + ".meta"):
            raise ("No model in path {}".format(filepath))
        with self.model.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)
#ModelWrapper()
