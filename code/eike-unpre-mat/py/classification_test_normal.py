import pandas as pd
import numpy as np
import timeit
from tqdm import tqdm
import time


class TestNormal:
    def __init__(self, data_set, score_func, epoch):
        self.data_set = data_set
        self.epoch = epoch
        self.score_func = score_func
        self.dim = 100
        self.entity_vec = []
        self.relation_vec = []
        self.delta_relation = []
        self.max_min_relation = []
        self.wrong_triple = []
        self.right_triple = []
        self.valid = True
        self.get_min_max = False
        self.valid_num = 0
        self.test_num = 0
        self.relation_num = 0

    def run(self):
        print('-----start valid normal triple----')
        start_time = timeit.default_timer()
        print('start time:{}'.format(time.asctime(time.localtime(time.time()))))
        self.prepare()
        triple_out_dict = self.run_valid()
        print('end time:{}'.format(time.asctime(time.localtime(time.time()))))
        print('cost_time:{}s'.format(timeit.default_timer() - start_time))
        return triple_out_dict
    
    def prepare(self, final_test=False):
        if self.valid:
            triple_negative_df = pd.read_csv("../../data/" + self.data_set + "/Valid/triple2id_negative.txt", header= None, sep=' ', skiprows=[0])
            triple_positive_df = pd.read_csv("../../data/" + self.data_set + "/Valid/triple2id_positive.txt", header=None, sep=' ', skiprows=[0])
            self.valid_num = len(triple_negative_df)
        else:
            triple_negative_df = pd.read_csv("../../data/" + self.data_set + "/Test/triple2id_negative.txt", header= None, sep=' ', skiprows=[0])
            triple_positive_df = pd.read_csv("../../data/" + self.data_set + "/Test/triple2id_positive.txt", header=None, sep=' ', skiprows=[0])
            self.test_num = len(triple_negative_df)
        
        relation_df = pd.read_csv("../../data/" + self.data_set + "/Train/relation2id.txt", header=None, skiprows=[0], sep='\t')
        self.relation_num = len(relation_df)
        entity_df = pd.read_csv("../../data/" + self.data_set + "/Train/instance2id.txt", header=None, skiprows=[0], sep='\t')
        entity_num = len(entity_df)

        if not final_test:
            self.delta_relation = [0 for _ in range(self.relation_num)]
        self.max_min_relation = [[-1, 1000000] for _ in range(self.relation_num)]
        self.wrong_triple = []
        self.right_triple = []
        for i in range(len(triple_negative_df)):
            self.wrong_triple.append(list(triple_negative_df.loc[i]))
            self.right_triple.append(list(triple_positive_df.loc[i]))
        
        f1_df = pd.read_csv("vector/" + self.data_set + "/instance_vec_ex2vec" + str(self.epoch) + ".vec", header=None, sep='\t')
        f2_df = pd.read_csv("vector/" + self.data_set + "/relation2vec" + str(self.epoch) + ".vec", header=None, sep='\t')
        self.entity_vec = []
        for i in range(len(f1_df)):
            self.entity_vec.append(f1_df.loc[i][:-1])
        self.relation_vec = []
        for i in range(len(f2_df)):
            self.relation_vec.append(f2_df.loc[i][:-1])

    def test(self):
        TP, TN, FP, FN = 0, 0, 0, 0
        ans = [[0, 0, 0, 0] for _ in range(self.relation_num)]
        
        if self.valid:
            input_size = self.valid_num
        else:
            input_size = self.test_num
        
        for i in range(input_size):
            if self.check(self.right_triple[i]):
                TP += 1
                ans[self.right_triple[i][2]][0] += 1
            else:
                FN += 1
                ans[self.right_triple[i][2]][1] += 1
            if not self.check(self.wrong_triple[i]):
                TN += 1
                ans[self.wrong_triple[i][2]][2] += 1
            else:
                FP += 1
                ans[self.wrong_triple[i][2]][3] += 1
        # print(ans)
        if self.valid:
            return_ans = []
            for i in range(self.relation_num):
                if ans[i] != [0, 0, 0, 0]:
                    return_ans.append((ans[i][0] + ans[i][2]) * 100 / (ans[i][0] + ans[i][1] + ans[i][2] + ans[i][3]))
                else:
                    return_ans.append(0)
            return return_ans
        else:
            with open('triple_relation_classification.csv', 'w') as f:
                f.write("relation" + ',' + "accuracy" + ',' + "precision" + ',' +  "recall" + ',' +  "f1-score" + ',' +  
                        "TP" + ',' +  "FN" + ',' +  "TN" + ',' +  "FP" + "\n")
                for i, item in enumerate(ans):
                    accuracy, precision, recall, f1_score = list(map(lambda x: str(x), self.calculate_triple_classification(item[0], item[1], item[2], item[3])))
                    # print("accuracy:{}, precision:{}, recall:{}, f1-score:{}".format(accuracy, precision, recall, f1_score) + 
                    #       "TP:{}, FN:{}, TN:{}, FP:{}".format(item[0], item[1], item[2], item[3]))
                    f.write(str(i) + ',' + accuracy + ',' + precision + ',' + recall + ',' + f1_score + ',' +
                            str(item[0]) + ',' + str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]) + '\n')

            triple_out_dict = {}
            print("Triple classification:")
            print("TP:{}, TN:{}, FP:{}, FN:{}".format(TP, TN, FP, FN))
            accuracy, precision, recall, f1_score = self.calculate_triple_classification(TP, FN, TN, FP)
            print("accuracy:{:.2f}%".format(accuracy))
            print("precision:{:.2f}%".format(precision))
            print("recall:{:.2f}%".format(recall))
            print("F1-score:{:.2f}%".format(f1_score))
            triple_out_dict['accuracy'] = accuracy
            triple_out_dict['precision'] = precision
            triple_out_dict['recall'] = recall
            triple_out_dict['F1'] = f1_score
            return triple_out_dict
    
    def calculate_triple_classification(self, tp, fn, tn, fp):
        if tp == 0:
            tp = 1
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # print("recall: {}, precision:{}".format(recall, precision))
        f1_score = 2 * precision * recall / (precision + recall)
        return list(map(lambda x: round(x*100, 2), [accuracy, precision, recall, f1_score]))

    def run_valid(self):
        self.get_min_max = True
        self.test()
        self.get_min_max = False
        best_ans_relation = [0 for _ in range(self.relation_num)]
        best_delta_relation = [0 for _ in range(self.relation_num)]
        # print(self.max_min_relation)
        for i in range(self.dim):
            for j in range(self.relation_num):
                self.delta_relation[j] = self.max_min_relation[j][1] + (self.max_min_relation[j][0] - self.max_min_relation[j][1])*i / 100
            
            ans = self.test()
            
            for k in range(self.relation_num):
                if ans[k] > best_ans_relation[k]:
                    best_ans_relation[k] = ans[k]
                    best_delta_relation[k] = self.delta_relation[k]
  
        for i in range(self.relation_num):
            self.delta_relation[i] = best_delta_relation[i]
        # print('best_ans_relation:{}'.format(self.delta_relation))
        print('-----start test nomal triples----')
        self.valid = False
        self.prepare(final_test=True)
        triple_out_dict = self.test()
        return triple_out_dict

    def check(self, triple):
        tmp = self.entity_vec[triple[0]] + self.relation_vec[triple[2]]
        if self.score_func == 1:
            dis = np.sum(np.abs(tmp - self.entity_vec[triple[1]]))
        else:
            dis = np.linalg.norm((tmp - self.entity_vec[triple[1]]), ord=2, axis=0)

        if self.get_min_max:
            if dis > self.max_min_relation[triple[2]][0]:
                self.max_min_relation[triple[2]][0] = dis

            if dis < self.max_min_relation[triple[2]][1]:
                self.max_min_relation[triple[2]][1] = dis
        return dis < self.delta_relation[triple[2]]


def test_triple(data_set, score_func, epoch):
    test_normal = TestNormal(data_set, score_func, epoch)
    triple_out_dict = test_normal.run()
    return triple_out_dict


if __name__ == '__main__':
    test_triple('YAGO39K',1, 1000)
