from nltk.tokenize import word_tokenize
import pickle
import os
import numpy as np
import sys
import random
from ModelWrapper import *
from vocab import *
#use shared vocabulary
class DataSet:
    def __init__(self, config, dataName = "train"):
        self.train_path = "train.txt"
        self.val_path = "valid.txt"
        self.test_path = "test.txt"
        self.dev_path = "dev.txt"
        self.eval_path = "eval.txt"
        self.Nl_Voc = {"pad":0, "Unknown" : 1}
        self.Code_Voc = {"pad":0, "Unknown": 1}
        self.Char_Voc = {"pad":0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        if not os.path.exists("nl_voc.pkl"):
            self.init_dic()
        self.Load_Voc()
        if dataName == "train":
            if os.path.exists("data.pkl"):
                self.data = pickle.load(open("data.pkl", "rb"))
                self.Nls = pickle.load(open("Nls.pkl", "rb"))
                self.Codes = pickle.load(open("Codes.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.train_path, "r", encoding='utf-8'))
        elif dataName == "val":
            if os.path.exists("valdata.pkl"):
                self.data = pickle.load(open("valdata.pkl", "rb"))
                self.Nls = pickle.load(open("valNls.pkl", "rb"))
                self.Codes = pickle.load(open("valCodes.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.val_path, "r", encoding='utf-8'))
        elif dataName == "test":
            if os.path.exists("testdata.pkl"):
                self.data = pickle.load(open("testdata.pkl", "rb"))
                self.Nls = pickle.load(open("testNls.pkl", "rb"))
                self.Codes = pickle.load(open("testCodes.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.test_path, "r", encoding='utf-8'))
        elif dataName == "dev":
            if os.path.exists("devdata.pkl"):
                self.data = pickle.load(open("devdata.pkl", "rb"))
                self.Nls = pickle.load(open("devNls.pkl", "rb"))
                self.Codes = pickle.load(open("devCodes.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.dev_path, "r", encoding='utf-8'))
        else:
            if os.path.exists("evaldata.pkl"):
                self.data = pickle.load(open("evaldata.pkl", "rb"))
                self.Nls = pickle.load(open("evalNls.pkl", "rb"))
                self.Codes = pickle.load(open("evalCodes.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.eval_path, "r", encoding='utf-8'))
    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
    def init_dic(self):
        print("initVoc")
        f = open(self.train_path, "r", encoding='utf-8')
        lines = f.readlines()
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for i in range(int(len(lines) / 2)):
            Nl = lines[2 * i].strip()
            Code = lines[2 * i + 1].strip()
            Nl_tokens = word_tokenize(Nl.lower())
            Code_Tokens = Code.lower().split()
            Nls.append(Nl_tokens)
            #Nls.append(Code_Tokens)
            Codes.append(Code_Tokens)
            maxNlLen = max(maxNlLen, len(Nl_tokens))
            maxCodeLen = max(maxCodeLen, len(Code_Tokens))
        #print(Nls)
        #print("------------------")
        nl_voc = VocabEntry.from_corpus(Nls, size=7500, freq_cutoff=3)
        code_voc = VocabEntry.from_corpus(Codes, size=7500, freq_cutoff=3)
        self.Nl_Voc = nl_voc.word2id
        self.Code_Voc = code_voc.word2id

        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        for x in self.Code_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        open("nl_voc.pkl", "wb").write(pickle.dumps(self.Nl_Voc))
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
        open("char_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))
        print(self.Nl_Voc)
        print(self.Code_Voc)
        print(maxNlLen, maxCodeLen, maxCharLen)
    def Get_Em(self, WordList, NlFlag=True):
        ans = []
        for x in WordList:
            if NlFlag:
                if x not in self.Nl_Voc:
                    ans.append(1)
                else:
                    ans.append(self.Nl_Voc[x])
            else:
                if x not in self.Code_Voc:
                    ans.append(1)
                else:
                    ans.append(self.Code_Voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def get_overlap_indices(self, question, answer):
        a = []
        b = []
        for x in question:
            isOverlap = False
            ma = 0
            for y in answer:
           #     ma = 0
                if x in y:
                    isOverlap = True
                    ma = max(ma, int(100 * (len(x) / len(y))))
                    #break
            a.append(ma)
            #if not isOverlap:
            #    a.append(0)
        for x in answer:
            isOverlap = False
            mb = 0
            for y in question:
                #mb = 0
                if x in y:
                    isOverlap = True
                    mb = max(mb, int(100 * (len(x) / len(y))))
                    #break
            b.append(mb)
            #if not isOverlap:
            #    b.append(0)
        a, _ = self.pad_seq(a, self.Nl_Len)
        b, _ = self.pad_seq(b, self.Code_Len)
        return a, b
    def preProcessData(self, datafile):
        lines = datafile.readlines()
        Nl_Sentences = []
        Code_Sentences = []
        Nl_Chars = []
        Code_Chars = []
        Nl_Overlap = []
        Code_Overlap = []
        res = []
        for i in range(int(len(lines) / 2)):
            Nl = lines[2 * i].strip()
            Code = lines[2 * i + 1].strip()
            if len(Code) == 0:
                continue
            Nl_tokens = word_tokenize(Nl.lower())
            Code_Tokens = Code.lower().split()
            self.Nls.append(Nl_tokens)
            self.Codes.append(Code_Tokens)
            Nl_Sentences.append(self.Get_Em(Nl_tokens))
            Code_Sentences.append(self.Get_Em(Code_Tokens, False))
            Nl_Chars.append(self.Get_Char_Em(Nl_tokens))
            Code_Chars.append(self.Get_Char_Em(Code_Tokens))
            res.append([0, 1])
            a, b = self.get_overlap_indices(Nl_tokens, Code_Tokens)
            Nl_Overlap.append(a)
            Code_Overlap.append(b)
        for i in range(len(Nl_Sentences)):
            Nl_Sentences[i], _ = self.pad_seq(Nl_Sentences[i], self.Nl_Len)
            Code_Sentences[i], _ = self.pad_seq(Code_Sentences[i], self.Code_Len)
            for j in range(len(Nl_Chars[i])):
                Nl_Chars[i][j], _ = self.pad_seq(Nl_Chars[i][j], self.Char_Len)
            for j in range(len(Code_Chars[i])):
                Code_Chars[i][j], _ = self.pad_seq(Code_Chars[i][j], self.Char_Len)
            Nl_Chars[i] = self.pad_list(Nl_Chars[i], self.Nl_Len, self.Char_Len)
            Code_Chars[i] = self.pad_list(Code_Chars[i], self.Code_Len, self.Char_Len)
        Nl_Sentences = np.array(Nl_Sentences, np.int32)
        Code_Sentences = np.array(Code_Sentences, np.int32)
        Nl_Chars = np.array(Nl_Chars, np.int32)
        Code_Chars = np.array(Code_Chars, np.int32)
        Nl_Overlap = np.array(Nl_Overlap, np.int32)
        Code_Overlap = np.array(Code_Overlap, np.int32)
        res = np.array(res)
        #Nl_Overlap = np.array(Nl_Overlap, np.int32)
        #Code_Overlap = np.array(Code_Overlap, np.int32)
        batchs = [Nl_Sentences, Nl_Chars, Code_Sentences, Code_Chars, Nl_Overlap, Code_Overlap, res]
        if self.dataName == "train":
            open("data.pkl", "wb").write(pickle.dumps(batchs))
            open("Nls.pkl", "wb").write(pickle.dumps(self.Nls))
            open("Codes.pkl", "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "val":
            open("valdata.pkl", "wb").write(pickle.dumps(batchs))
            open("valNls.pkl", "wb").write(pickle.dumps(self.Nls))
            open("valCodes.pkl", "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "test":
            open("testdata.pkl", "wb").write(pickle.dumps(batchs))
            open("testNls.pkl", "wb").write(pickle.dumps(self.Nls))
            open("testCodes.pkl", "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "dev":
            open("devdata.pkl", "wb").write(pickle.dumps(batchs))
            open("devNls.pkl", "wb").write(pickle.dumps(self.Nls))
            open("devCodes.pkl", "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "eval":
            open("evaldata.pkl", "wb").write(pickle.dumps(batchs))
            open("evalNls.pkl", "wb").write(pickle.dumps(self.Nls))
            open("evalCodes.pkl", "wb").write(pickle.dumps(self.Codes))
        return batchs
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq, act_len
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def Get_Train(self, batch_size, data="train"):
        data = self.data
        loaddata = []
        if self.dataName == "train":

            NegId = []
            for i in range(len(data[0])):
                tmp = []
                for j in range(5):
                    rand_offset = random.randint(0, len(data[0]) - 1)
                    while rand_offset == i:
                        rand_offset = random.randint(0, len(data[0]) - 1)
                    tmp.append(rand_offset)
                NegId.append(tmp)
            maxlen = len(data[0])
            tmp = []
            for i in range(len(data)):
                tmp.append([])
            for i in range(maxlen):
                for x in NegId[i]:
                    tmp[0].append(data[0][i])
                    tmp[1].append(data[1][i])
                    tmp[2].append(data[2][x])
                    tmp[3].append(data[3][x])
                    a, b = self.get_overlap_indices(self.Nls[i], self.Codes[x])
                    tmp[4].append(np.array(a, np.int32))
                    tmp[5].append(np.array(b, np.int32))
                    tmp[6].append([1, 0])
            for i in range(len(data)):
                loaddata.append(np.append(data[i], tmp[i], axis=0))
            shuffle = np.random.permutation(range(len(loaddata[0])))
            for i in range(len(data)):
                loaddata[i] = loaddata[i][shuffle]
        if self.dataName == "val" or self.dataName == "test" or self.dataName == "dev" or self.dataName == "eval":
            loaddata = data
        batch_nums = int(len(loaddata[0]) / batch_size)
        print(batch_nums)
        for i in range(batch_nums):
            ans = []
            for j in range(len(loaddata)):
                ans.append(loaddata[j][batch_size * i:batch_size * (i + 1)])
            yield ans
            #yield Nl_Sentences[batch_size * i: batch_size * (i + 1)],Nl_Chars[batch_size * i: batch_size * (i + 1)],Code_Sentences[batch_size * i: batch_size * (i + 1)],Code_Chars[batch_size * i: batch_size * (i + 1)],Neg_Code_Sentences[batch_size * i: batch_size * (i + 1)], Neg_Code_Chars[batch_size * i: batch_size * (i + 1)]
