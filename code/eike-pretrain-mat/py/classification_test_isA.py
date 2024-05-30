# encoding=UTF-8
import numpy as np
import math
import pandas as pd
import timeit

cosine_similarity = lambda vec1, vec2: np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class TestIsA:
    def __init__(self, data_set, epoch, in_rate):
        self.data_set = data_set
        self.epoch = epoch
        self.in_rate = in_rate
        self.dim = 0
        self.delta_sub_max = 0
        self.delta_sub_min = 0
        self.delta_ins_max = 0
        self.delta_ins_min = 0
        self.delta_ins = 0
        self.delta_sub_max = 0 #
        self.delta_sub_min = 0 #
        self.delta_sub = 0
        self.get_max_min = True
        self.delta_sub_dim = []
        self.delta_ins_dim = []
        self.ins_test_num = 0
        self.sub_test_num = 0
        self.ins_wrong = []
        self.ins_right = []
        self.sub_wrong = []
        self.sub_right = []
        self.instance_vec = []
        self.concept_ext_vec = []
        self.concept_r = []
        self.concept_int_vec = []
        self.mix = False
        self.valid = True

    def load_vector(self):
        f1 = open("vector/" + self.data_set + "/instance_vec_ex2vec" + str(self.epoch) + ".vec", 'r', encoding='utf-8')
        f4 = open("vector/" + self.data_set + "/instance_vec_in2vec" + str(self.epoch) + ".vec", 'r', encoding='utf-8')
        f2 = open("vector/" + self.data_set + "/concept_vec_ex2vec" + str(self.epoch) + ".vec", 'r', encoding='utf-8')
        f3 = open("vector/" + self.data_set + "/concept_vec_in2vec" + str(self.epoch) + ".vec", 'r', encoding='utf-8')
        
        self.instance_vec_ex = list()
        while True:
            line = f1.readline()
            if not line:
                break
            line = line.strip('\n').split('\t')
            line_list = list(map(float, line))
            self.instance_vec_ex.append(line_list)
        self.dim = len(self.instance_vec_ex[0])

        self.instance_vec_in = list()
        while True:
            line = f4.readline()
            if not line:
                break
            line = line.strip('\n').split('\t')
            line_list = list(map(float, line))
            self.instance_vec_in.append(line_list)
        self.dim = len(self.instance_vec_in[0])    

        self.concept_ext_vec = list()
        self.concept_r = list()
        while True:
            line_concept = f2.readline().strip('\n')
            line_r = f2.readline().strip('\n')
            if not line_r:
                break
            line_concept = line_concept.split('\t')
            line_concept_list = list(map(float, line_concept))
            line_r = line_r.split('\t')
            line_r_list = list(map(float, line_r))
            self.concept_ext_vec.append(line_concept_list)
            self.concept_r.append(line_r_list)

        self.concept_int_vec = list()
        while True:
            line = f3.readline()
            if not line:
                break
            line = line.strip('\n').split('\t')
            line_list = list(map(float, line))
            self.concept_int_vec.append(line_list)

        self.concept_ext_vec = np.array(self.concept_ext_vec)
        self.concept_r = np.array(self.concept_r)
        self.concept_int_vec = np.array(self.concept_int_vec)
        self.instance_vec = np.array(self.instance_vec)

        return True


    def prepare(self):
        print('-----prepare data -----')
        if self.valid:
            if self.mix:
                fin = open("../../data/" + self.data_set + "/M-Valid/instanceOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/M-Valid/instanceOf2id_positive.txt", 'r',
                                 encoding='utf-8')
            else:
                fin = open("../../data/" + self.data_set + "/Valid/instanceOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/Valid/instanceOf2id_positive.txt", 'r',
                                 encoding='utf-8')
        else:
            if self.mix:
                fin = open("../../data/" + self.data_set + "/M-Test/instanceOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/M-Test/instanceOf2id_positive.txt", 'r',
                                 encoding='utf-8')
            else:
                fin = open("../../data/" + self.data_set + "/Test/instanceOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/Test/instanceOf2id_positive.txt", 'r',
                                 encoding='utf-8')

        self.ins_test_num = int(fin.readline().strip('\n'))
        self.ins_test_num = int(fin_right.readline().strip('\n'))

        self.ins_wrong = []
        self.ins_right = []
        for i in range(self.ins_test_num):
            tmp = list(map(int, fin.readline().strip('\n').split(' ')))
            self.ins_wrong.append((tmp[0], tmp[1]))
            tmp = list(map(int, fin_right.readline().strip('\n').split(' ')))
            self.ins_right.append((tmp[0], tmp[1]))

        fin.close()
        fin_right.close()

        if self.valid:
            if self.mix:
                fin = open("../../data/" + self.data_set + "/M-Valid/subClassOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/M-Valid/subClassOf2id_positive.txt", 'r',
                                 encoding='utf-8')
            else:
                fin = open("../../data/" + self.data_set + "/Valid/subClassOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/Valid/subClassOf2id_positive.txt", 'r',
                                 encoding='utf-8')
        else:
            if self.mix:
                fin = open("../../data/" + self.data_set + "/M-Test/subClassOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/M-Test/subClassOf2id_positive.txt", 'r',
                                 encoding='utf-8')
            else:
                fin = open("../../data/" + self.data_set + "/Test/subClassOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/Test/subClassOf2id_positive.txt", 'r',
                                 encoding='utf-8')

        self.sub_test_num = int(fin.readline().strip('\n'))
        self.sub_test_num = int(fin_right.readline().strip('\n'))

        self.sub_wrong = []
        self.sub_right = []
        for i in range(self.sub_test_num):
            tmp = list(map(int, fin.readline().strip('\n').split(' ')))
            self.sub_wrong.append((tmp[0], tmp[1]))
            tmp = list(map(int, fin_right.readline().strip('\n').split(' ')))
            self.sub_right.append((tmp[0], tmp[1]))

        fin.close()
        fin_right.close()

    def run_valid(self):
        print('-----start valid-----')

        self.get_max_min = True
        current_ans = self.test()
        self.get_max_min = False

        ins_best_answer = 0
        ins_best_delta = 0
        sub_best_answer = 0
        sub_best_delta = 0
        for i in range(100):
            f = i / 100
            self.delta_ins = self.delta_ins_min + (self.delta_ins_max - self.delta_ins_min) * i / 100
            self.delta_sub = f * 2
            #self.delta_sub = self.delta_sub_min + (self.delta_sub_max - self.delta_sub_min) * i / 100
            ans = self.test()
            if ans[0] > ins_best_answer:
                ins_best_answer = ans[0]
                ins_best_delta = self.delta_ins
            if ans[1] > sub_best_answer:
                sub_best_answer = ans[1]
                sub_best_delta = f * 2
                #sub_best_delta = self.delta_sub
        print("delta_ins is " + str(ins_best_delta) + ". The best ins accuracy on valid data is " + str(ins_best_answer)
              + "%")
        print("delta_sub is " + str(sub_best_delta) + ". The best sub accuracy on valid data is " + str(sub_best_answer)
              + "%")
        self.delta_ins = ins_best_delta
        self.delta_sub = sub_best_delta

    def test(self):
        TP_ins, TN_ins, FP_ins, FN_ins = 0, 0, 0, 0
        TP_sub, TN_sub, FP_sub, FN_sub = 0, 0, 0, 0
        TP_ins_dict, TN_ins_dict, FP_ins_dict, FN_ins_dict = dict(), dict(), dict(), dict()
        concept_set = dict()

        def check_instance(instance, concept):
            dis_ext = np.sum(np.square(np.true_divide(self.instance_vec_ex[instance] - self.concept_ext_vec[concept],
                                           self.concept_r[concept][:-1]))) - np.square(self.concept_r[concept][-1])
            dis_ext = dis_ext if dis_ext > 0 else 0
            dis_int =  1 - cosine_similarity(self.instance_vec_in[instance], self.concept_int_vec[concept])
            score = dis_ext  + dis_int * self.in_rate
            if self.get_max_min:
                if score > self.delta_ins_max:
                    self.delta_ins_max = score
                if score < self.delta_ins_min:
                    self.delta_ins_min = score
            return score < self.delta_ins

        def check_sub_class(concept1, concept2):
            dis_ext = np.sum(np.square(np.true_divide(self.concept_ext_vec[concept1], self.concept_r[concept1][:-1]) -
                            np.true_divide(self.concept_ext_vec[concept2], self.concept_r[concept2][:-1])))  + \
                             np.square(self.concept_r[concept1][-1]) - np.square(self.concept_r[concept2][-1])
            dis_ext = dis_ext if dis_ext > 0 else 0
            dis_int = 1 - cosine_similarity(self.concept_int_vec[concept1], self.concept_int_vec[concept2]) +  \
                             np.linalg.norm(self.concept_int_vec[concept1], axis=-1) - np.linalg.norm(self.concept_int_vec[concept2], axis=-1)
            dis = dis_ext + dis_int * self.in_rate
            score = dis
            if np.sqrt(dis_ext) < np.fabs(self.concept_r[concept1][-1] - self.concept_r[concept2][-1]) and \
                    self.concept_r[concept1][-1] < self.concept_r[concept2][-1]:
                return True
            if np.sqrt(dis_ext) < self.concept_r[concept1][-1] + self.concept_r[concept2][-1]:
                tmp = (self.concept_r[concept1][-1] + self.concept_r[concept2][-1] - np.sqrt(dis_ext) - dis_int * self.in_rate) / self.concept_r[concept1][-1]
                if tmp > self.delta_sub:
                    return True
            return False
            #if self.get_max_min:
            #    if score > self.delta_sub_max:
            #        self.delta_sub_max = score
            #    if score < self.delta_sub_min:
            #        self.delta_sub_min = score
            #return score < self.delta_sub


        for i in range(self.ins_test_num):
            if check_instance(self.ins_right[i][0], self.ins_right[i][1]):
                TP_ins += 1
                if self.ins_right[i][1] in TP_ins_dict:
                    TP_ins_dict[self.ins_right[i][1]] += 1
                else:
                    TP_ins_dict[self.ins_right[i][1]] = 1
            else:
                FN_ins += 1
                if self.ins_right[i][1] in FN_ins_dict:
                    FN_ins_dict[self.ins_right[i][1]] += 1
                else:
                    FN_ins_dict[self.ins_right[i][1]] = 1

            if not check_instance(self.ins_wrong[i][0], self.ins_wrong[i][1]):
                TN_ins += 1
                if self.ins_wrong[i][1] in TN_ins_dict:
                    TN_ins_dict[self.ins_wrong[i][1]] += 1
                else:
                    TN_ins_dict[self.ins_wrong[i][1]] = 1
            else:
                FP_ins += 1
                if self.ins_wrong[i][1] in FP_ins_dict:
                    FP_ins_dict[self.ins_wrong[i][1]] += 1
                else:
                    FP_ins_dict[self.ins_wrong[i][1]] = 1
            concept_s = self.ins_right[i][1]
            concept_m = self.ins_wrong[i][1]
            concept_set[concept_s] = None
            concept_set[concept_m] = None

        for i in range(self.sub_test_num):
            if check_sub_class(self.sub_right[i][0], self.sub_right[i][1]):
                TP_sub += 1
            else:
                FN_sub += 1
            if not check_sub_class(self.sub_wrong[i][0], self.sub_wrong[i][1]):
                TN_sub += 1
            else:
                FP_sub += 1

        if self.valid:
            ins_ans = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins)
            sub_ins = (TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FN_sub + FP_sub)
            tmp_tuple = (ins_ans, sub_ins)
            return tmp_tuple
        else:
            instance_out_dict = {}
            print("instanceOf triple classification:")
            print("TP:{}, TN:{}, FP:{}, FN:{}".format(TP_ins, TN_ins, FP_ins, FN_ins))
            if TP_ins == 0:
                TP_ins = 1
            accuracy_ins = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins)
            precision_ins = TP_ins * 100 / (TP_ins + FP_ins)
            recall_ins = TP_ins * 100 / (TP_ins + FN_ins)
            F1_ins = 2 * precision_ins * recall_ins / (precision_ins + recall_ins)
            print("accuracy: {:.2f}%".format(accuracy_ins))
            print("precision: {:.2f}%".format(precision_ins))
            print("recall: {:.2f}%".format(recall_ins))
            print("F1-score: {:.2f}%".format(F1_ins))
            instance_out_dict['accuracy'] = round(accuracy_ins, 2)
            instance_out_dict['precision'] = round(precision_ins, 2)
            instance_out_dict['recall'] = round(recall_ins, 2)
            instance_out_dict['F1'] = round(F1_ins, 2)

            subclass_out_dict = {}
            print("subClassOf triple classification:")
            print("TP:{}, TN:{}, FP:{}, FN:{}".format(TP_sub, TN_sub, FP_sub, FN_sub))
            if TP_sub == 0:
                TP_sub = 1
            accuracy_sub = (TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FN_sub + FP_sub)
            precision_sub = TP_sub * 100 / (TP_sub + FP_sub)
            recall_sub = TP_sub * 100 / (TP_sub + FN_sub)
            F1_sub = 2 * precision_sub * recall_sub / (precision_sub + recall_sub)
            print("accuracy: {:.2f}%".format(accuracy_sub))
            print("precision: {:.2f}%".format(precision_sub))
            print("recall: {:.2f}%".format(recall_sub))
            print("F1-score: {:.2f}%".format(F1_sub))

            subclass_out_dict['accuracy'] = round(accuracy_sub, 2)
            subclass_out_dict['precision'] = round(precision_sub, 2)
            subclass_out_dict['recall'] = round(recall_sub, 2)
            subclass_out_dict['F1'] = round(F1_sub, 2)

            # don't understand

            """for item in sorted(concept_set):
                index = item
                TP_ins = TP_ins_dict[item]
                TN_ins = TN_ins_dict[item]
                FN_ins = FN_ins_dict[item]
                FP_ins = FP_ins_dict[item]
                accuracy = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins)
                precision = TP_ins * 100 / (TP_ins + FP_ins)
                recall = TP_ins * 100 / (TP_ins + FN_ins)
                p = TP_ins * 100 / (TP_ins + FP_ins)
                r = TP_ins * 100 / (TP_ins + FN_ins)
                f1 = 2 * p * r / (p + r)"""

            tmp_tuple = (instance_out_dict, subclass_out_dict, self.delta_ins, self.delta_sub)
            return tmp_tuple

    def run(self):
        self.load_vector()
        # prepare for valid to load valid data

        self.mix = False
        self.valid = True
        self.prepare()
        self.run_valid()
        # test
        self.valid = False
        self.prepare()
        instance_out_dict, subclass_out_dict, delta_ins, delta_sub = self.test()

        # m数据集
        self.mix = True
        self.valid = True
        self.prepare()
        self.run_valid()
        # test
        self.valid = False
        self.prepare()
        m_instance_out_dict, m_subclass_out_dict, m_delta_ins, m_delta_sub = self.test()
        return (instance_out_dict, subclass_out_dict, m_instance_out_dict, m_subclass_out_dict, delta_ins, delta_sub,
                m_delta_ins, m_delta_sub)


def test_isA(data_set, epoch, ex_rate):
    test_isa_example = TestIsA(data_set, epoch, ex_rate)
    result_tuple = test_isa_example.run()
    return result_tuple


if __name__ == '__main__':
    test_isA('YAGO39K', 1000, 0.3)