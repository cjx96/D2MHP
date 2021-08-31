import math
import copy
import random
import numpy as np
import torch


def cal_metric(rank_list):
    F1=[0] * len(rank_list)
    MRR = [0] * len(rank_list)
    recall3= [0] * len(rank_list)
    recall5 = [0] * len(rank_list)
    recall7 = [0] * len(rank_list)
    MRR3 = [0] * len(rank_list)
    MRR5 = [0] * len(rank_list)
    MRR7 = [0] * len(rank_list)
    NDCG3 = [0] * len(rank_list)
    NDCG5 = [0] * len(rank_list)
    NDCG7 = [0] * len(rank_list)
    for r_id, rank in enumerate(rank_list):
        MRR[r_id] = 1/(rank + 1)
        if rank<7:
            if rank<5:
                if rank<3:
                    if rank<1:
                        F1[r_id] = 1
                    recall3[r_id] = 1
                    MRR3[r_id] = 1/(rank+1)
                    NDCG3[r_id] = 1 / np.log2(rank + 2)
                recall5[r_id] = 1
                MRR5[r_id] = 1 / (rank + 1)
                NDCG5[r_id] = 1 / np.log2(rank + 2)
            recall7[r_id] = 1
            MRR7[r_id] = 1 / (rank + 1)
            NDCG7[r_id] = 1 / np.log2(rank + 2)

    F1 = sum(F1) / len(F1)
    MRR = sum(MRR) / len(MRR)
    recall3 = sum(recall3) / len(recall3)
    recall5 = sum(recall5) / len(recall5)
    recall7 = sum(recall7) / len(recall7)
    NDCG3 = sum(NDCG3) / len(NDCG3)
    NDCG5 = sum(NDCG5) / len(NDCG5)
    NDCG7 = sum(NDCG7) / len(NDCG7)
    MRR3 = sum(MRR3) / len(MRR3)
    MRR5 = sum(MRR5) / len(MRR5)
    MRR7 = sum(MRR7) / len(MRR7)
    print("F1: ", F1)
    print("MRR: ", MRR)




def adapt_time(seq, time_seq, target):
    ok = 0
    ans = []
    pre = 0
    for id, event_id in enumerate(seq):
        if event_id == 0:
            ans.append(0)
        else:
            if ok:
                ans.append(min((time_seq[id] - pre, 100)) + ans[-1])
                pre = time_seq[id]
            else:
                ans.append(0)
                pre = time_seq[id]
                ok = 1

    target = min((target - pre, 100)) / 10
    return np.array(ans, dtype=np.float32), target

def evaluate(trainer, dataset, args):
    [train, valid, test, entitynum, eventnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0

    if entitynum > 10000:
        entitys = random.sample(range(1, entitynum + 1), 10000)
    else:
        entitys = range(1, entitynum + 1)

    batches_seq = []
    batches_time = []
    batches_event = []
    batches_time_target = []
    batch_size = 64
    rmse = []
    rank_list = []
    for u_id, u in enumerate(entitys):

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        time_seq, target_time = adapt_time(seq, time_seq, test[u][0][1])
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        event_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, eventnum + 1)
            while t in rated: t = np.random.randint(1, eventnum + 1)
            event_idx.append(t)

        if batch_size :
            batch_size -= 1
            batches_seq.append(seq)
            batches_time.append(time_seq)
            batches_event.append(event_idx)
            batches_time_target.append(target_time)
        if batch_size == 0 or (u_id+1) == len(entitys):
            predictions, p_time = trainer.model.predict(*[torch.LongTensor(np.array(batches_seq)).cuda(), torch.FloatTensor(np.array(batches_time)).cuda(),torch.LongTensor(np.array(batches_event)).cuda()])


            for id, time in enumerate(p_time):
                time = time.item()
                rmse.append((time - batches_time_target[id]) * (time - batches_time_target[id]))
            for pred in predictions:
                rank = pred.argsort().argsort()[0].item()

                valid_entity += 1
                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
                if valid_entity % 100 == 0:
                    print('.', end='')
                rank_list.append(rank)
            batches_seq = []
            batches_time = []
            batches_event = []
            batches_time_target = []
            batch_size = 64

    print("test rmse:", sum(rmse) / len(rmse))
    cal_metric(rank_list)
    return NDCG / valid_entity, HT / valid_entity


def evaluate_valid(trainer, dataset, args):
    [train, valid, test, entitynum, eventnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_entity = 0.0
    HT = 0.0
    if entitynum > 10000:
        entitys = random.sample(range(1, entitynum + 1), 10000)
    else:
        entitys = range(1, entitynum + 1)

    batches_seq = []
    batches_time = []
    batches_time_target = []
    batches_event = []
    batch_size = 64
    rank_list = []
    rmse = []
    for u_id, u in enumerate(entitys):
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        time_seq, target_time = adapt_time(seq, time_seq, valid[u][0][1])
        batches_time_target.append(target_time)
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        event_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, eventnum + 1)
            while t in rated: t = np.random.randint(1, eventnum + 1)
            event_idx.append(t)

        if batch_size:
            batch_size -= 1
            batches_seq.append(seq)
            batches_time.append(time_seq)
            batches_event.append(event_idx)
            batches_time_target.append(target_time)
        if batch_size == 0 or (u_id+1) == len(entitys):
            predictions, p_time = trainer.model.predict(*[torch.LongTensor(np.array(batches_seq)).cuda(), torch.FloatTensor(np.array(batches_time)).cuda(),torch.LongTensor(np.array(batches_event)).cuda()])

            for id, time in enumerate(p_time):
                time = time.item()
                rmse.append((time - batches_time_target[id]) * (time - batches_time_target[id]))

            for pred in predictions:
                rank = pred.argsort().argsort()[0].item()
                rank_list.append(rank)
                valid_entity += 1
                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
                if valid_entity % 100 == 0:
                    print('.', end='')

            batches_seq = []
            batches_time = []
            batches_event = []
            batches_time_target = []
            batch_size = 64
    print("valid rmse:", sum(rmse) / len(rmse))
    # cal_metric(rank_list)
    return NDCG / valid_entity, HT / valid_entity