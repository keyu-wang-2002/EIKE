import math
import timeit
import numpy as np
import pandas as pd 
# import tensorflow as tf
import multiprocessing as mp
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class KnowledgeGraph:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_num = 0
        self.relation_num = 0
        self.concept_num = 0
        self.entity_dict = {}
        self.relation_dict = {}
        self.concept_dict = {}
        self.entities = []
        self.concepts = []
        self.triples = []
        self.triples_num = 0
        self.instance_of_num = 0
        self.instance_of = []
        self.instance_of_ok = {}
        self.subclass_of_num = 0
        self.subclass_of = []
        self.subclass_of_ok = {}
        self.train_num = 0

        self.concept_instance = []
        self.instance_concept = []
        self.sub_up_concept = []
        self.up_sub_concept = []
        self.instance_brother = []
        self.concept_brother = []
        self.test_triples = []
        self.valid_triples = []
        self.test_triple_num = 0
        self.valid_triple_num = 0
        self.triple_head_instance_brother = []
        self.triple_tail_instance_brother = []
        self.instance_of_head_instance_brother = []
        self.instance_of_tail_instance_brother = []
        self.subclass_of_head_instance_brother = []
        self.subclass_of_tail_instance_brother = []
        '''load dicts train data'''
        self.load_dicts()
        self.load_train_data()
        '''construct pools after loading'''
        self.triples_pool = set(self.triples)
        self.golden_triple_pool = set(self.triples) | set(self.valid_triples) | set(self.test_triples)
        self.instance_of_pool = set(self.instance_of)
        self.subclass_of_pool = set(self.subclass_of)


    def load_dicts(self):
        entity_dict_file = "instance2id.txt"
        relation_dict_file = "relation2id.txt"
        concept_dict_file = "concept2id.txt"
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join('../../data/', self.data_dir,'Train', entity_dict_file), header=None, skiprows=[0])
        # print(entity_df[0])
        # print('------')
        # print(entity_df[1])
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.entity_num = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.entity_num))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', relation_dict_file), header=None, skiprows=[0])
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.relation_num = len(self.relation_dict)
        print('#relation: {}'.format(self.relation_num))
        print('-----Loading concept dict-----')
        concept_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', concept_dict_file), header=None, skiprows=[0])
        self.concept_dict = dict(zip(concept_df[0], concept_df[1]))
        self.concept_num = len(self.concept_dict)
        self.concepts = list(self.concept_dict.values())
        print('#concept: {}'.format(self.concept_num))

        self.concept_instance = [[] for _ in range(self.concept_num)]
        self.instance_concept = [[] for _ in range(self.entity_num)]
        self.sub_up_concept = [[] for _ in range(self.concept_num)]
        self.up_sub_concept = [[] for _ in range(self.concept_num)]
        self.instance_brother = [[] for _ in range(self.entity_num)]
        self.concept_brother = [[] for _ in range(self.concept_num)]

    def load_train_data(self):
        instance_of_file = "instanceOf2id.txt"
        subclass_of_file = "subClassOf2id.txt"
        triple_file = "triple2id.txt"

        print('-----Loading instance_of triples-----')
        instance_of_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', instance_of_file), header=None, sep=' ')
        # print(instance_of_df[0])
        # print(instance_of_df[1])
        self.instance_of = list(zip(instance_of_df[0], instance_of_df[1]))
        self.instance_of_num = len(self.instance_of)
        print('#instance of :{}'.format(self.instance_of_num))
        self.instance_of_ok = dict(zip(self.instance_of, [1 for i in range(len(self.instance_of))]))
        for instance_of_item in self.instance_of:
            self.instance_concept[instance_of_item[0]].append(instance_of_item[1])
            self.concept_instance[instance_of_item[1]].append(instance_of_item[0])

        print('-----Loading training triples-----')
        triple_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', triple_file), header=None, sep=' ', skiprows=[0])
        self.triples = list(zip(triple_df[0], triple_df[1], triple_df[2]))
        self.triples_num = len(self.triples)
        print('#triples:{}'.format(self.triples_num))

        print('-----Loading subclass_of triples-----')
        subclass_of_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', subclass_of_file), header=None, sep=' ')
        self.subclass_of = list(zip(subclass_of_df[0], subclass_of_df[1]))
        self.subclass_of_num = len(self.subclass_of)
        print('#subclass of:{}'.format(self.subclass_of_num))
        self.subclass_of_ok = dict(zip(self.subclass_of, [1 for i in range(len(self.subclass_of))]))
        for subclass_of_item in self.subclass_of:
            self.sub_up_concept[subclass_of_item[0]].append(subclass_of_item[1])
            self.up_sub_concept[subclass_of_item[1]].append(subclass_of_item[0])
        
        self.train_num = self.triples_num + self.instance_of_num + self.subclass_of_num
        print('#train_num:{}'.format(self.train_num))

        print('-----Loading test triples data-----')
        test_df = pd.read_csv('../../data/' + self.data_dir + '/Test/triple2id_positive.txt', header=None, sep=' ', skiprows=[0])
        self.test_triples = list(zip(test_df[0], test_df[1], test_df[2]))
        self.test_triple_num = len(self.test_triples)

        print('-----Loading valid triples data-----')
        valid_df = pd.read_csv('../../data/' + self.data_dir + '/Valid/triple2id_positive.txt', header=None, sep=' ', skiprows=[0])
        self.valid_triples = list(zip(valid_df[0], valid_df[1], valid_df[2]))
        self.valid_triple_num = len(self.valid_triples)



class LinkPrediction:
    def __init__(self, kg: KnowledgeGraph, score_func,
                 embedding_dim, n_rank_calculator, epoch):
        self.kg = kg
        self.epoch = epoch
        self.embedding_dim = embedding_dim
        self.n_rank_calculator = n_rank_calculator
        self.score_func = score_func
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        self.r_trainable = True
        self.concept_trainable = True
        '''读入数据'''
        f1_df = pd.read_csv("vector/"+ kg.data_dir +"/entity2vec" + str(self.epoch) + ".vec", header=None, sep='\t')
        f2_df = pd.read_csv("vector/"+ kg.data_dir +"/relation2vec" + str(self.epoch) + ".vec", header=None, sep='\t')
        entity_vec = []
        for i in range(len(f1_df)):
            entity_vec.append(f1_df.loc[i][:-1])
        relation_vec = []
        for i in range(len(f2_df)):
            relation_vec.append(f2_df.loc[i][:-1])

        
        '''embeddings'''
        self.entity_embedding = tf.Variable(tf.convert_to_tensor(np.array(entity_vec)), dtype=tf.float64, trainable=False)
        self.relation_embedding = tf.Variable(tf.convert_to_tensor(np.array(relation_vec)), dtype=tf.float64, trainable=False)
        
        self.build_eval_graph()

        
        
    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)


    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 1:  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                        k=self.kg.entity_num)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                        k=self.kg.entity_num)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.entity_num)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.entity_num)
        return idx_head_prediction, idx_tail_prediction

    def launch_evaluation(self, session):

        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            #print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
            #                                                   n_used_eval_triple,
            #                                                   self.kg.test_triple_num), end='\r')
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_meanrank_reciprocal_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_meanrank_reciprocal_raw = 0

        tail_hits10_raw = 0

        '''Filter'''
        head_meanrank_filter = 0
        head_meanrank_reciprocal_filter = 0
        head_hits10_filter = 0
        head_hits5_filter = 0
        head_hits3_filter = 0
        head_hits1_filter = 0
        tail_meanrank_filter = 0
        tail_meanrank_reciprocal_filter = 0
        tail_hits10_filter = 0
        tail_hits5_filter = 0
        tail_hits3_filter = 0
        tail_hits1_filter = 0

        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            head_meanrank_reciprocal_raw += 1 / (head_rank_raw + 1)
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            tail_meanrank_reciprocal_raw += 1 / (tail_rank_raw + 1)
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            head_meanrank_reciprocal_filter += 1 / (head_rank_filter + 1)
            if head_rank_filter < 10:
                head_hits10_filter += 1
            if head_rank_filter < 5:
                head_hits5_filter += 1
            if head_rank_filter < 3:
                head_hits3_filter += 1
            if head_rank_filter < 1:
                head_hits1_filter += 1
            tail_meanrank_filter += tail_rank_filter
            tail_meanrank_reciprocal_filter += 1 / (tail_rank_filter + 1)
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
            if tail_rank_filter < 5:
                tail_hits5_filter += 1
            if tail_rank_filter < 3:
                tail_hits3_filter += 1
            if tail_rank_filter < 1:
                tail_hits1_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_meanrank_reciprocal_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_meanrank_reciprocal_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        # print('-----Head prediction-----')
        # print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        # print('-----Tail prediction-----')
        # print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        mean_rank_raw = round((head_meanrank_raw + tail_meanrank_raw) / 2, 3)
        mean_rank_reciprocal_raw = round((head_meanrank_reciprocal_raw + tail_meanrank_reciprocal_raw) / 2, 3)
        hits10_raw = round((head_hits10_raw + tail_hits10_raw) / 2, 3)*100
        print('------Average------')
        print('MeanRank: {:.3f}, MeanRankReciprocal:{:.3f}, Hits@10: {:.3f}'
              .format((head_meanrank_raw + tail_meanrank_raw) / 2,
                      (head_meanrank_reciprocal_raw + tail_meanrank_reciprocal_raw) / 2,
                      (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_meanrank_reciprocal_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        head_hits5_filter /= n_used_eval_triple
        head_hits3_filter /= n_used_eval_triple
        head_hits1_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_meanrank_reciprocal_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        tail_hits5_filter /= n_used_eval_triple
        tail_hits3_filter /= n_used_eval_triple
        tail_hits1_filter /= n_used_eval_triple
        # print('-----Head prediction-----')
        # print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        # print('-----Tail prediction-----')
        # print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        mean_rank_filter = round((head_meanrank_filter + tail_meanrank_filter) / 2, 3)
        meanrank_reciprocal_filter = round((head_meanrank_reciprocal_filter + tail_meanrank_reciprocal_filter) / 2, 3)
        hits10_filter = round((head_hits10_filter + tail_hits10_filter) / 2, 3)*100
        hits5_filter = round((head_hits5_filter + tail_hits5_filter) / 2, 3)*100
        hits3_filter = round((head_hits3_filter + tail_hits3_filter) / 2, 3)*100
        hits1_filter = round((head_hits1_filter + tail_hits1_filter) / 2, 3)*100
        
        print('-----Average-----')
        print('MeanRank: {:.3f}, MeanRankReciprocal: {:.3f}, Hits@10: {:.3f}, Hits@5: {:.3f}, Hits@3: {:.3f}, '
              'Hits@1: {:.3f}'.format(mean_rank_filter,
                                      meanrank_reciprocal_filter,
                                      hits10_filter,
                                      hits5_filter,
                                      hits3_filter,
                                      hits1_filter)
              )
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

        return (mean_rank_raw, mean_rank_reciprocal_raw, hits10_raw, mean_rank_filter, meanrank_reciprocal_filter,
                hits10_filter, hits5_filter, hits3_filter, hits1_filter)
        
        # with open('mrr.csv', 'a', encoding='utf-8') as f:
        #     f.write(str(mean_rank_raw) + ',' +
        #             str(mean_rank_reciprocal_raw) + ',' +
        #             str(hits10_raw) + ',' +
        #             str(mean_rank_filter) + ',' +
        #             str(meanrank_reciprocal_filter) + ',' +
        #             str(hits10_filter) + ',' +
        #             str(hits5_filter) + ',' +
        #             str(hits3_filter) + ',' +
        #             str(hits1_filter) + '\n'
        #             )

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

def test_begain(data_set, score_func, embedding_dim, epoch, n_rank_calculator=32):
    kg=KnowledgeGraph(data_set)
    lk = LinkPrediction(kg, score_func, embedding_dim, n_rank_calculator, epoch)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        tf.global_variables_initializer().run()
        result = lk.launch_evaluation(sess)
    return result

if __name__ == "__main__":
    test_begain('YAGO39K', 1, 100, 1000)