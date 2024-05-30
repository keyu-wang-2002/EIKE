import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
from collections import Counter
import pickle as pkl
import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def norm(x, pnorm=0):
    if pnorm == 1:
        return torch.sum(torch.abs(x), -1)
    else:
        return torch.sum(x**2,-1)

def normalize_emb(x):
    # return  x/float(length)
    veclen = torch.clamp_min_(torch.norm(x, 2, -1, keepdim=True), 1.0)
    ret = x/veclen
    return ret.detach()

def normalize_radius(x):
    return torch.clamp(x,min=-1.0,max=1.0) # 不能置0否则无法反向传播

def cos_dist(x, y):
    return sum(x*y) / ((sum(x*x)*sum(y*y))**0.5)

class Dataset(object):
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.args = args
        self.instance_num, self.instance2id = self.read_file(self.dataset_name, "instance2id")
        self.relation_num, self.relation2id = self.read_file(self.dataset_name, "relation2id")
        self.concept_num, self.concept2id = self.read_file(self.dataset_name, "concept2id")
        self.triple_num, self.triples = self.read_triples(self.dataset_name, "triple2id")

        self.fb_h, self.fb_t, self.fb_r = [], [], []
        self.relation_vec,self.instance_vec,self.concept_vec = [],[],[]
        self.relation_tmp, self.instance_tmp, self.concept_tmp = [], [], []
        self.concept_r, self.concept_r_tmp = [], []
        self.ok = {}
        self.subClassOf_ok = {}
        self.instanceOf_ok = {}
        self.subClassOf = []
        self.instanceOf = []
        self.instance_concept = [[] for i in range(self.instance_num)]
        self.concept_instance = [[] for i in range(self.concept_num)]
        self.sub_up_concept = [[] for i in range(self.concept_num)]
        self.up_sub_concept = [[] for i in range(self.concept_num)]



    def read_file(self, dataset,filename,split = 'Train'):
        with open("../../data/" + dataset + "/" + split+"/"+filename + ".txt", encoding='utf-8') as file:
            L = file.readlines()
            num = int(L[0].strip())
            contents = [[x for x in line.strip().split()] for line in L[1:]]
        return num, contents

    def read_triples(self, dataset,filename,split = 'Train'):
        with open("../../data/" + dataset + "/" + split+"/"+filename + ".txt", encoding='utf-8') as file:
            L = file.readlines()
            num = int(L[0].strip())
            contents = [[int(x) for x in line.strip().split()] for line in L[1:]]
        return num, contents

    def read_biples(self, dataset, filename,split = 'Train'):
        with open("../../data/" + dataset + "/" + split+"/"+filename + ".txt", encoding='utf-8') as file:
            L = file.readlines()
            contents = [[int(x) for x in line.strip().split()] for line in L[1:]]
        return contents

    def addHrt(self, x, y, z):  # x: head ,y: tail, z:relation
        self.fb_h.append(x)
        self.fb_r.append(z)
        self.fb_t.append(y)
        if (x, z) not in self.ok:
            self.ok[(x, z)] = {y: 1}
        else:
            self.ok[(x, z)][y] = 1

    def addSubClassOf(self, sub, parent):
        self.subClassOf.append([sub, parent])
        self.subClassOf_ok[(sub, parent)] = 1

    def addInstanceOf(self, instance, concept):
        self.instanceOf.append([instance, concept])
        self.instanceOf_ok[(instance, concept)] = 1

    def setup(self):
        self.left_entity = [Counter() for i in range(self.relation_num)]
        self.right_entity = [Counter() for i in range(self.relation_num)]

        for h, t, r in self.triples:
            self.addHrt(h, t, r)
            self.left_entity[r][h] += 1
            self.right_entity[r][t] += 1

        self.left_num = [float(sum(c.values())) / float(len(c)) for c in self.left_entity]
        self.right_num = [float(sum(c.values())) / float(len(c)) for c in self.right_entity]

        self.instanceOf_contents = self.read_biples(self.args.dataset, "instanceOf2id")
        self.subClassOf_contents = self.read_biples(self.args.dataset, "subClassOf2id")

        for a, b in self.instanceOf_contents:
            self.addInstanceOf(a,b)
            self.instance_concept[a].append(b)
            self.concept_instance[b].append(a)

        for a, b in self.subClassOf_contents:
            self.addSubClassOf(a,b)
            self.sub_up_concept[a].append(b)
            self.up_sub_concept[b].append(a)


        self.instance_brother = [[ins for concept in concepts
                                for ins in self.concept_instance[concept]
                                if ins != instance_out]
                                for instance_out, concepts
                                in enumerate(self.instance_concept)]

        self.concept_brother = [[sub for up in ups
                                for sub in self.up_sub_concept[up]
                                if sub != sub_out]
                                for sub_out, ups
                                in enumerate(self.sub_up_concept)]

        self.trainSize = len(self.fb_h) + len(self.instanceOf) + len(self.subClassOf)

        print("train size {} {} {} {}".format(self.trainSize, len(self.fb_h),len(self.instanceOf),len(self.subClassOf)))

    def save(self,):
        with open("../../data/" + self.dataset_name + "/" + self.args.split + "/processed.pkl",'wb') as file:
            pkl.dump(self, file)
def load_processed(dataset_name,split):
    with open("../../data/" + dataset_name + "/" + split + "/processed.pkl",'rb') as file:
        res = pkl.load(file)
    return res


class Train(nn.Module):
    def __init__(self,args,dataset):
        super(Train, self).__init__()
        self.args = args
        self.D = dataset
        self.instance_vec_ex = nn.Embedding(self.D.instance_num, args.emb_dim)
        self.concept_vec_ex = nn.Embedding(self.D.concept_num, args.emb_dim)
        self.concept_r = nn.Embedding(self.D.concept_num, 1)
        self.concept_axis = nn.Embedding(self.D.concept_num, args.emb_dim)
        self.relation_vec = nn.Embedding(self.D.relation_num, args.emb_dim)
        self.concept_vec_in = nn.Embedding.from_pretrained(self.pretrained_vector())
        self.concept_in_vec_linear = nn.Linear(self.pre_dim, args.emb_dim)
        self.instance_map_ex2in = torch.rand(args.emb_dim, args.emb_dim)
        #self.instance_map_ex2in = nn.Embedding(self.D.instance_num, args.emb_dim)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr)

        self.instance_vec_ex.weight.requires_grad = True
        self.concept_vec_ex.weight.requires_grad = True
        self.concept_vec_in.weight.requires_grad = True
        self.relation_vec.weight.requires_grad = True
        self.concept_r.weight.requires_grad = True
        self.concept_axis.weight.requires_grad = False
        self.concept_in_vec_linear.weight.requires_grad = True
        self.instance_map_ex2in.requires_grad = True


        # initialize
        nn.init.normal_(self.instance_vec_ex.weight.data, 0.0, 1.0 / args.emb_dim)
        nn.init.normal_(self.relation_vec.weight.data, 0.0, 1.0 / args.emb_dim)
        nn.init.normal_(self.concept_vec_ex.weight.data, 0.0, 1.0 / args.emb_dim)
        nn.init.normal_(self.concept_vec_in.weight.data, 0.0, 1.0 / args.emb_dim)
        nn.init.normal_(self.instance_map_ex2in, 0.0, 1.0 / args.emb_dim)
        nn.init.uniform_(self.concept_r.weight.data, 0.0, 1.0)
        nn.init.constant_(self.concept_axis.weight.data, 1.000000)

    def pretrained_vector(self):
        def read_file(dataset,filename,split = 'Train'):
            with open("../../data/" + dataset + "/" + split+"/"+filename + ".txt", encoding='utf-8') as file:
                L = file.readlines()
                num = int(L[0].strip())
                contents = [[x for x in line.strip().split()] for line in L[1:]]
            return num, contents
        dataset_name="YAGO39K"
        entity_num, conept2id = read_file(dataset_name, "concept2id")
        #concept_id2mention = {}
        concept_mention_list = []
        #pretrained_tensor = torch.ones(len(conept2id), self.pre_dim)
        #embedder = pipeline("feature-extraction", model="nlpconnect/dpr-ctx_encoder_bert_uncased_L-2_H-128_A-2") 
        embedder = SentenceTransformer('/root/transformers_hub/models--sentence-transformers--all-MiniLM-L6-v2')
        print("-----get pretrained concept mention vector-------")
        #for i in tqdm(range(len(conept2id))):
        for i in range(len(conept2id)):
            concept = conept2id[i][0].strip('<').strip('>').split('_')
            concept_mention = " ".join(concept)
            concept_mention = concept_mention
            #concept_id2mention[i] = concept_mention
            concept_mention_list.append(concept_mention)
            #pretrained_tensor[i] = torch.tensor(embedder(concept_mention)[0][0])
        embeddings = torch.tensor(embedder.encode(concept_mention_list))
        return torch.tensor(embeddings)
        #return pretrained_tensor

    def doTrain(self):
        nbatches = self.args.nbatches
        nepoch = self.args.nepoch
        batchSize = int(self.D.trainSize / nbatches)
        allreadyindex = 0

        dis_a_L, dis_b_L = [], []
        dis_count = 0
        for epoch in range(nepoch):
            res = 0
            if epoch >= 1000:
                if epoch % 200 < 100:
                    self.instance_vec_ex.weight.requires_grad = False
                    self.concept_vec_ex.weight.requires_grad = False
                    self.relation_vec.weight.requires_grad = False
                    self.concept_r.weight.requires_grad = True
                    self.concept_axis.weight.requires_grad = True
                    self.concept_vec_in.weight.requires_grad = True
                    self.instance_map_ex2in.requires_grad = True
                else:
                    self.instance_vec_ex.weight.requires_grad = True
                    self.concept_vec_ex.weight.requires_grad = True
                    self.relation_vec.weight.requires_grad = True
                    self.concept_r.weight.requires_grad = False
                    self.concept_axis.weight.requires_grad = False
                    self.concept_vec_in.weight.requires_grad = True
                    self.instance_map_ex2in.requires_grad = True


            if epoch == 0:
                instance_vec_ex_save = self.instance_vec_ex.weight.detach().cpu().numpy()
                instance_vec_in_save = (self.instance_vec_ex.weight.cuda(0) @ self.instance_map_ex2in.cuda(0)).detach().cpu().numpy()
                concept_vec_in_save = self.concept_vec_in.weight.detach().cpu().numpy()
                concept_vec_ex_save = self.concept_vec_ex.weight.detach().cpu().numpy()
                relation_vec_save = self.relation_vec.weight.detach().cpu().numpy()
                concept_r_save = self.concept_r.weight.detach().cpu().numpy()
                concept_axis_save = self.concept_axis.weight.detach().cpu().numpy()

                # with open("embeddings/transc/"+self.args.version+"_embeddings_epoch" + str(epoch) + ".pkl", 'wb') as file:
                #    pkl.dump({"entity_vec": entity_vec_save,
                #             "concept_vec": concept_vec_save,
                #             "relation_vec":relation_vec_save},file)
                #print("saved!")

                #write for cpp test
                with open("./vector/"+self.args.dataset +"/instance_vec_ex2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec in instance_vec_ex_save:
                        list_vec = list(vec)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        file.write(str_vec+"\n")
                
                with open("./vector/"+self.args.dataset +"/instance_vec_in2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec in instance_vec_in_save:
                        list_vec = list(vec)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        file.write(str_vec+"\n")                

                with open("./vector/"+ self.args.dataset + "/relation2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec in relation_vec_save:
                        list_vec = list(vec)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        file.write(str_vec+"\n")

                with open("./vector/" + self.args.dataset+"/concept_vec_ex2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec, r, r_axis in zip(concept_vec_ex_save, concept_r_save, concept_axis_save):
                        list_vec = list(vec)
                        list_r_axis = list(r_axis)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        str_r_axis = "\t".join([str(x) for x in list_r_axis])
                        str_r = str(r[0])
                        file.write(str_vec + "\n" + str_r_axis + "\t" + str_r + "\n")
                
                with open("./vector/"+self.args.dataset +"/concept_vec_in2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec in concept_vec_in_save:
                        list_vec = list(vec)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        file.write(str_vec+"\n")       


            for batch in range(nbatches):
                losses = []
                stime = time.time()
                pairs = [[], [], []]

                #normalize
                self.instance_vec_ex.weight.data = normalize_emb(self.instance_vec_ex.weight.data)
                self.instance_map_ex2in = normalize_emb(self.instance_map_ex2in)
                self.relation_vec.weight.data = normalize_emb(self.relation_vec.weight.data)
                self.concept_vec_ex.weight.data = normalize_emb(self.concept_vec_ex.weight.data)
                self.concept_vec_in.weight.data = normalize_emb(self.concept_vec_in.weight.data)
                self.concept_r.weight.data = normalize_radius(self.concept_r.weight.data)
                self.optimizer.zero_grad()
                for k in range(batchSize):
                    i = random.randint(0, self.D.trainSize - 1)
                    if i < len(self.D.fb_r):
                        cut = 1 - epoch % 1000 * self.args.hrt_cut / 1000
                        pairs[0].append(self.trainHLR(i, cut))
                    elif i < len(self.D.fb_r) + len(self.D.instanceOf):
                        cut = 1 - epoch % 1000 * self.args.ins_cut / 1000
                        pairs[1].append(self.trainInstanceOf(i, cut))
                    else:
                        cut = 1 - epoch % 1000 * self.args.sub_cut / 1000
                        pairs[2].append(self.trainSubClassOf(i, cut))

                # for k in range(batchSize):
                #     line = self.training_instance_file.readline()
                #     line = line.strip().split("\t")
                #     instance = [int(x) for x in line]
                #     # print(instance)
                #     if instance[0] == -1:
                #         pairs[0].append(instance[1:])
                #     if instance[0] == -2:
                #         pairs[1].append(instance[1:])
                #     if instance[0] == -3:
                #         pairs[2].append(instance[1:])
                # allreadyindex += batchSize

                tensor_pairs= []
                for i in range(3):
                    tensor_pairs.append(torch.stack([torch.tensor(x) for x in list(zip(*pairs[i]))]).cuda())
                loss1,dis_a,dis_b = self.doTrainHLR(tensor_pairs[0])
                loss2 = self.doTrainInstanceOf(tensor_pairs[1])
                loss3 = self.doTrainSubClassOf(tensor_pairs[2])
                losses = loss1 + loss2 + loss3
                losses.backward()

                dis_a_L.append(torch.sqrt(dis_a).sum()), dis_b_L.append(torch.sqrt(dis_b).sum()) # for logs
                dis_count += dis_a.size(0)

                self.optimizer.step()
                res += losses.detach().cpu().numpy() # 阻断反向传播并numpy形式数值。

            # print(sum(dis_a_L) / dis_count, sum(dis_b_L) / dis_count, dis_a.size())
            dis_a_L, dis_b_L = [], []
            dis_count = 0

            if epoch % 100 == 0:
                print("epoch:{} Res: {:.6f} Loss {:.6f},loss1: {:.6f},loss2: {:.6f},loss3 {:.6f}".format(epoch,res,losses,loss1,loss2,loss3))

            if (epoch + 1) % 500 == 0:
            # if (epoch + 1)  == 1: # test code
                instance_vec_ex_save = self.instance_vec_ex.weight.detach().cpu().numpy()
                instance_vec_in_save = (self.instance_vec_ex.weight.cuda(0) @ self.instance_map_ex2in.cuda(0)).detach().cpu().numpy()
                concept_vec_in_save = self.concept_vec_in.weight.detach().cpu().numpy()
                concept_vec_ex_save = self.concept_vec_ex.weight.detach().cpu().numpy()
                relation_vec_save = self.relation_vec.weight.detach().cpu().numpy()
                concept_r_save = self.concept_r.weight.detach().cpu().numpy()
                concept_axis_save = self.concept_axis.weight.detach().cpu().numpy()

                # with open("embeddings/transc/"+self.args.version+"_embeddings_epoch" + str(epoch) + ".pkl", 'wb') as file:
                #    pkl.dump({"entity_vec": entity_vec_save,
                #             "concept_vec": concept_vec_save,
                #             "relation_vec":relation_vec_save},file)
                #print("saved!")

                #write for cpp test
                with open("./vector/"+self.args.dataset +"/instance_vec_ex2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec in instance_vec_ex_save:
                        list_vec = list(vec)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        file.write(str_vec+"\n")
                
                with open("./vector/"+self.args.dataset +"/instance_vec_in2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec in instance_vec_in_save:
                        list_vec = list(vec)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        file.write(str_vec+"\n")                

                with open("./vector/"+ self.args.dataset + "/relation2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec in relation_vec_save:
                        list_vec = list(vec)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        file.write(str_vec+"\n")

                with open("./vector/" + self.args.dataset+"/concept_vec_ex2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec, r, r_axis in zip(concept_vec_ex_save, concept_r_save, concept_axis_save):
                        list_vec = list(vec)
                        list_r_axis = list(r_axis)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        str_r_axis = "\t".join([str(x) for x in list_r_axis])
                        str_r = str(r[0])
                        file.write(str_vec + "\n" + str_r_axis + "\t" + str_r + "\n")
                
                with open("./vector/"+self.args.dataset +"/concept_vec_in2vec" + str(epoch + 1) + ".vec", 'w') as file:
                    for vec in concept_vec_in_save:
                        list_vec = list(vec)
                        str_vec = "\t".join([str(x) for x in list_vec])
                        file.write(str_vec+"\n")      


    def trainHLR(self, i, cut):
        pr = 0.5
        cur_fbr, cur_fbh, cur_fbt = self.D.fb_r[i], self.D.fb_h[i], self.D.fb_t[i]
        if self.args.bern == 1:
            pr = float(self.D.right_num[cur_fbr]) / (self.D.right_num[cur_fbr] + self.D.left_num[cur_fbr])
        if random.uniform(0, 1) < pr:
            loop=True
            while loop:

                if len(self.D.instance_brother[cur_fbt]) > 0:
                    if random.uniform(0, 1) < cut:
                        j = random.randint(0, self.D.instance_num - 1)
                    else:
                        j = random.randint(0, len(self.D.instance_brother[cur_fbt]) - 1)
                        j = self.D.instance_brother[cur_fbt][j]
                else:
                    j = random.randint(0, self.D.instance_num - 1)
                loop = j in self.D.ok[(cur_fbh, cur_fbr)]
            return cur_fbh, cur_fbt, cur_fbr, cur_fbh, j, cur_fbr
        else:
            loop=True
            while loop:
                if len(self.D.instance_brother[cur_fbh]) > 0:
                    if random.uniform(0, 1) < cut:
                        j = random.randint(0, self.D.instance_num - 1)
                    else:
                        j = random.randint(0, len(self.D.instance_brother[cur_fbh]) - 1)
                        j = self.D.instance_brother[cur_fbh][j]
                else:
                    j = random.randint(0, self.D.instance_num - 1)
                loop = ((j,cur_fbr) in self.D.ok) and (cur_fbt in self.D.ok[(j, cur_fbr)])
            return cur_fbh, cur_fbt, cur_fbr, j, cur_fbt, cur_fbr

    def trainInstanceOf(self, i, cut):
        i = i - len(self.D.fb_h)
        cur_ins,cur_cpt = self.D.instanceOf[i]
        if random.randint(0, 1) == 0:
            loop=True
            while loop:
                if len(self.D.instance_brother[cur_ins]) > 0: #
                    if random.uniform(0, 1) < cut:
                        j = random.randint(0, self.D.instance_num - 1)
                    else:
                        j = random.randint(0, len(self.D.instance_brother[cur_ins]) - 1)
                        j = self.D.instance_brother[cur_ins][j]
                else:
                    j = random.randint(0, self.D.instance_num - 1)
                loop = (j, cur_cpt) in self.D.instanceOf_ok
            return cur_ins, cur_cpt, j, cur_cpt

        else:
            loop=True
            while loop:
                if len(self.D.concept_brother[cur_cpt]) > 0: #
                    if random.uniform(0, 1) < cut:
                        j = random.randint(0, self.D.concept_num - 1)
                    else:
                        j = random.randint(0, len(self.D.concept_brother[cur_cpt]) - 1)
                        j = self.D.concept_brother[cur_cpt][j]
                else:
                    j = random.randint(0, self.D.concept_num - 1)
                loop = (cur_ins, j) in self.D.instanceOf_ok
            return cur_ins, cur_cpt, cur_ins, j

    def trainSubClassOf(self, i, cut):
        i = i - len(self.D.fb_h) - len(self.D.instanceOf)

        cur_cpth,cur_cptt=self.D.subClassOf[i]
        if random.randint(0, 1) == 0:
            loop=True
            while loop:
                if len(self.D.concept_brother[cur_cpth]) > 0: #
                    if random.uniform(0, 1) < cut:
                        j = random.randint(0, self.D.concept_num - 1)
                    else:
                        j = random.randint(0, len(self.D.concept_brother[cur_cpth]) - 1)
                        j = self.D.concept_brother[cur_cpth][j]
                else:
                    j = random.randint(0, self.D.concept_num - 1)
                loop = (j, cur_cptt) in self.D.subClassOf_ok
            return cur_cpth, cur_cptt, j, cur_cptt
        else:
            loop=True
            while loop:
                if len(self.D.concept_brother[cur_cptt]) > 0: #
                    if random.uniform(0, 1) < cut:
                        j = random.randint(0, self.D.concept_num - 1)
                    else:
                        j = random.randint(0, len(self.D.concept_brother[cur_cptt]) - 1)
                        j = self.D.concept_brother[cur_cptt][j]
                else:
                    j = random.randint(0, self.D.concept_num - 1)
                loop = (cur_cpth, j) in self.D.subClassOf_ok
            return cur_cpth, cur_cptt, cur_cpth, j

    def doTrainHLR(self, ids):
        instance_embs_ex = self.instance_vec_ex(ids[[0, 1, 3, 4], :])
        relation_embs = self.relation_vec(ids[[2, 5], :])

        dis_a = norm(instance_embs_ex[0] + relation_embs[0] - instance_embs_ex[1], pnorm=self.args.pnorm)
        dis_b = norm(instance_embs_ex[2] + relation_embs[1] - instance_embs_ex[3], pnorm=self.args.pnorm)

        loss = F.relu(dis_a + self.args.margin_hrt - dis_b).sum()
        return loss,dis_a,dis_b

    def doTrainInstanceOf(self, ids):
        instance_embs_ex = self.instance_vec_ex(ids[[0, 2], :])
        concept_embs_ex = self.concept_vec_ex(ids[[1, 3], :])
        radius = torch.squeeze(self.concept_r(ids[[1, 3], :]), dim=-1)
        radius_axis = self.concept_axis(ids[[1, 3], :])
        instance_embs_in = self.instance_vec_ex(ids[[0, 2], :]).cuda(0) @ self.instance_map_ex2in.cuda(0)
        concept_embs_in = self.concept_vec_in(ids[[1, 3], :])

        dis_in = F.relu(norm(torch.div(instance_embs_ex - concept_embs_ex, radius_axis), pnorm=2) - radius ** 2)
        dis_ex = 1 - torch.cosine_similarity(concept_embs_in, instance_embs_in, -1)

        dis = dis_in * self.args.in_rate + dis_ex 
        loss = F.relu(dis[0] + self.args.margin_ins - dis[1]).sum()
        return loss

    def doTrainSubClassOf(self, ids):
        concept_embs_a_ex = self.concept_vec_ex(ids[[0, 2], :])
        concept_embs_b_ex = self.concept_vec_ex(ids[[1, 3], :])
        radius_a = torch.squeeze(self.concept_r(ids[[0, 2], :]), dim=-1)
        radius_b = torch.squeeze(self.concept_r(ids[[1, 3], :]), dim=-1)
        radius_a_axis = self.concept_axis(ids[[0, 2], :])
        radius_b_axis = self.concept_axis(ids[[1, 3], :])

        concept_embs_a_in = self.concept_vec_in(ids[[0, 2], :])
        concept_embs_b_in = self.concept_vec_in(ids[[1, 3], :])

        dis_ex = F.relu(norm(torch.div(concept_embs_a_ex, radius_a_axis) - torch.div(concept_embs_b_ex, radius_b_axis), pnorm=2)
                     + torch.square(radius_a) - torch.square(radius_b))
        dis_in = 1 - torch.cosine_similarity(concept_embs_a_in, concept_embs_b_in, -1) + torch.norm(concept_embs_a_in, dim=-1) - torch.norm(concept_embs_b_in, dim=-1)
        dis = dis_in * self.args.in_rate + dis_ex 

        loss = F.relu(dis[0] + self.args.margin_sub - dis[1]).sum()
        return loss

# def read_file(dataset,filename,split = 'Train'):
#     with open("data/" + dataset + "/" + split+"/"+filename + ".txt") as file:
#         L = file.readlines()
#         num = int(L[0].strip())
#         contents = [[x for x in line.strip().split()] for line in L[1:]]
#     return num, contents
#
# def read_triples(dataset,filename,split = 'Train'):
#     with open("data/" + dataset + "/" + split+"/"+filename + ".txt") as file:
#         L = file.readlines()
#         num = int(L[0].strip())
#         contents = [[int(x) for x in line.strip().split()] for line in L[1:]]
#     return num, contents
#
# def read_biples(dataset, filename,split = 'Train'):
#     with open("data/" + dataset + "/" + split+"/"+filename + ".txt") as file:
#         L = file.readlines()
#         contents = [[int(x) for x in line.strip().split()] for line in L[1:]]
#     return contents
#
#
# def parseargs():
#     parsers = argparse.ArgumentParser()
#     parsers.add_argument("--emb_dim", type=int,default=100)
#     parsers.add_argument("--margin_hrt", type=float, default=1.0)
#     parsers.add_argument("--margin_ins", type=float, default=0.4)
#     parsers.add_argument("--margin_sub", type=float, default=0.3)
#     parsers.add_argument("--hrt_cut", type=float, default=0.8)
#     parsers.add_argument("--ins_cut", type=float, default=0.8)
#     parsers.add_argument("--sub_cut", type=float, default=0.8)
#
#     parsers.add_argument("--nepoch", type=float, default=1000)
#     parsers.add_argument("--nbatches", type=float, default=100)
#
#     parsers.add_argument("--lr", type=float, default=0.001)
#     parsers.add_argument("--bern", type=int, default=1)
#     parsers.add_argument("--pnorm", type=int, default=2)
#     parsers.add_argument("--dataset", type=str, default="YAGO39K")
#     parsers.add_argument("--split", type=str, default="Train")
#     parsers.add_argument("--version", type=str, default='tmp')
#
#     args= parsers.parse_args()
#     return args
#
# def main():
#     args = parseargs()
#
#     if not os.path.exists("data/" + args.dataset + "/" + args.split + "/processed.pkl"):
#         dataset = Dataset(args=args)
#         dataset.setup()
#         dataset.save()
#     else:
#         dataset = load_processed(dataset_name=args.dataset, split=args.split)
#         print("dataset loaded")
#
#     train = Train(args = args,dataset= dataset).cuda()
#     train.doTrain()
#
#
# if __name__ == "__main__":
#     main()
#
#
#


