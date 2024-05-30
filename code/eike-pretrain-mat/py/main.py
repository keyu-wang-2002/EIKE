import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
from collections import Counter
import pickle as pkl
import os
import time
from classification_test_isA import test_isA
from classification_test_normal import test_triple
from link_prediction import test_begain
from transflex_pretrain_mat import Dataset, load_processed, Train
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def save_output(epoch, args, instance_out_dict, subclass_out_dict, m_instance_out_dict, m_subclass_out_dict, triple_out_dict,delta_ins, delta_sub,  m_delta_ins, m_delta_sub, link_result):
    with open('args.csv', 'a', encoding='utf-8') as f:
        f.write(str(epoch) + ',' +
                str(args.emb_dim) + ',' +
                str(args.nbatches) + ',' +
                str(args.pnorm) + ',' +
                str(args.margin_hrt) + ',' +
                str(args.margin_ins) + ',' +
                str(args.margin_sub) + ',' +
                str(args.lr) + ',' +
                str(delta_ins) + ',' +
                str(delta_sub) + ',' +
                str(m_delta_ins) + ',' +
                str(m_delta_sub) + ',' +
                str(instance_out_dict['accuracy']) + ',' +
                str(instance_out_dict['precision']) + ',' +
                str(instance_out_dict['recall']) + ',' +
                str(instance_out_dict['F1']) + ',' +
                str(subclass_out_dict['accuracy']) + ',' +
                str(subclass_out_dict['precision']) + ',' +
                str(subclass_out_dict['recall']) + ',' +
                str(subclass_out_dict['F1']) + ',' +
                str(triple_out_dict['accuracy']) + ',' +
                str(triple_out_dict['precision']) + ',' +
                str(triple_out_dict['recall']) + ',' +
                str(triple_out_dict['F1']) + ',' +
                str(m_instance_out_dict['accuracy']) + ',' +
                str(m_instance_out_dict['precision']) + ',' +
                str(m_instance_out_dict['recall']) + ',' +
                str(m_instance_out_dict['F1']) + ',' +
                str(m_subclass_out_dict['accuracy']) + ',' +
                str(m_subclass_out_dict['precision']) + ',' +
                str(m_subclass_out_dict['recall']) + ',' +
                str(m_subclass_out_dict['F1']) + ',' +
                str(link_result[0]) + ',' +
                str(link_result[1]) + ',' +
                str(link_result[2]) + ',' +
                str(link_result[3]) + ',' +
                str(link_result[4]) + ',' +
                str(link_result[5]) + ',' +
                str(link_result[6]) + ',' +
                str(link_result[7]) + ',' +
                str(link_result[8]) + ',' +
                str(args.bern) + ',' +
                '\n'
                )


def parseargs():
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--emb_dim", type=int,default=100)
    parsers.add_argument("--margin_hrt", type=float, default=1.0)
    parsers.add_argument("--margin_ins", type=float, default=0.3)
    parsers.add_argument("--margin_sub", type=float, default=0.1)
    parsers.add_argument("--in_rate", type=float, default=0.3)
    parsers.add_argument("--hrt_cut", type=float, default=0.8)
    parsers.add_argument("--ins_cut", type=float, default=0.8)
    parsers.add_argument("--sub_cut", type=float, default=0.8)

    parsers.add_argument("--nepoch", type=float, default=2000) #default=2000
    parsers.add_argument("--nbatches", type=float, default=100) #default=100

    parsers.add_argument("--lr", type=float, default=0.001)
    parsers.add_argument("--bern", type=int, default=1)
    parsers.add_argument("--pnorm", type=int, default=1)
    parsers.add_argument("--dataset", type=str, default="YAGO39K")
    parsers.add_argument("--split", type=str, default="Train")
    parsers.add_argument("--version", type=str, default='tmp')

    args= parsers.parse_args()
    return args


def main(margin_ins, margin_sub, ex_rate, bern, pnorm):
    code_start_time = time.asctime(time.localtime(time.time()))
    args = parseargs()
    args.margin_ins = margin_ins
    args.margin_sub = margin_sub
    args.ex_rate = ex_rate
    args.bern = bern
    args.pnorm = pnorm

    if not os.path.exists("data/" + args.dataset + "/" + args.split + "/processed.pkl"):
        dataset = Dataset(args=args)
        dataset.setup()
        dataset.save()
    else:
        dataset = load_processed(dataset_name=args.dataset, split=args.split)
        print("dataset loaded")

    train = Train(args=args, dataset=dataset).cuda()
    train.doTrain()
    code_train_end_time = time.asctime(time.localtime(time.time()))


    for epoch in [1000, 1500, 2000]:
        print("epoch: {0}".format(epoch))
        print('------test classification isA-----')
        instance_out_dict, subclass_out_dict, m_instance_out_dict, m_subclass_out_dict,delta_ins, delta_sub,  m_delta_ins, m_delta_sub = test_isA(args.dataset, epoch, ex_rate)
        print('------test classification normal-----')
        triple_out_dict = test_triple(args.dataset, args.pnorm,epoch)
        print('---------test link prediction ---------')
        link_result = test_begain(args.dataset, args.pnorm, args.emb_dim, epoch)
        save_output(epoch, args, instance_out_dict, subclass_out_dict, m_instance_out_dict, m_subclass_out_dict, triple_out_dict,delta_ins, delta_sub,  m_delta_ins, m_delta_sub, link_result)
    code_test_end_time = time.asctime(time.localtime(time.time()))
    print('start time:{}'.format(code_start_time))
    print('end time:{}'.format(code_train_end_time))
    print('test end time:{}'.format(code_test_end_time))

if __name__ == '__main__':
    #bern_list = [0, 1]
    #margin_list = [0.1, 0.3, 0.4]
    #pnorm_list = [1, 2]
    #parameters = [[0, 0.5, 0.1, 2], [1, 0.5, 0.1, 2], [0, 0.4, 0.3, 2]]
    # for bern in [1]:
    #     for margin_ins in margin_list[:-1]:
    #         for margin_sub in margin_list[:-1]:
    #             if margin_sub > margin_ins:
    #                 continue
    #             for pnorm in [1, 2]:
    #                 main(bern, margin_ins, margin_sub, pnorm)
    #for bern in [0, 1]:
    #    for margin_ins in margin_list[:-1]:
    #        for margin_sub in margin_list[:-1]:
    #            print("bern={0}, margin_ins={1}, margin_sub={2}, pnorm={3}".format(bern, margin_ins, margin_sub, 0))
    #            main(bern, margin_ins, margin_sub, pnorm=0)
    
    bern = 1
    pnorm = 1
    # margin_ins, margin_sub, ex_rate
    parameters = [
        [1.0, 0.4, 2],
        [1.0, 0.4, 1],
        [1.0, 0.4, 0.5],
        [1.0, 0.4, 0.1],

        [1.0, 0.3, 2],
        [1.0, 0.3, 1],
        [1.0, 0.3, 0.5]
    ]

    for margin_ins, margin_sub, ex_rate in parameters:
        print("margin_ins={0}, margin_sub={1}, ex_rate={2}, bern={3}, pnorm={4}".format(margin_ins, margin_sub, ex_rate, bern, pnorm))
        main(margin_ins=margin_ins, margin_sub=margin_sub , ex_rate=ex_rate, bern=bern, pnorm=pnorm)














