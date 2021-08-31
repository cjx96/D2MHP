import json
import random
import torch
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def data_partition(fname, opt):
    Entity = defaultdict(list)
    entity_train = {}
    entity_valid = {}
    entity_test = {}

    print('Preparing data...')
    f = open(fname, 'r')
    time_set = set()

    entity_count = defaultdict(int)
    event_count = defaultdict(int)
    cnt = 0
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        entity_count[u] += 1
        event_count[i] += 1
        cnt += 1
    f.close()
    f = open(fname, 'r')
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if entity_count[u] < 5 or event_count[i] < 5:
            continue
        time_set.add(timestamp)
        Entity[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set, opt)
    Entity, entitynum, eventnum, timenum = cleanAndsort(Entity, time_map)

    for entity in Entity:
        nfeedback = len(Entity[entity])
        if nfeedback < 3:
            entity_train[entity] = Entity[entity]
            entity_valid[entity] = []
            entity_test[entity] = []
        else:
            entity_train[entity] = Entity[entity][:-2]
            entity_valid[entity] = []
            entity_valid[entity].append(Entity[entity][-2])
            entity_test[entity] = []
            entity_test[entity].append(Entity[entity][-1])
    print('Preparing done...')
    return [entity_train, entity_valid, entity_test, entitynum, eventnum, timenum]


def timeSlice(time_set, opt):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = float(time - time_min) / 60 / 60
    return time_map


def cleanAndsort(Entity, time_map):
    entity_filted = dict()
    entity_set = set()
    event_set = set()
    for entity, events in Entity.items():
        entity_set.add(entity)
        entity_filted[entity] = events
        for event in events:
            event_set.add(event[0])
    entity_map = dict()
    event_map = dict()
    for u, entity in enumerate(entity_set):
        entity_map[entity] = u + 1
    for i, event in enumerate(event_set):
        event_map[event] = i + 1

    for entity, events in entity_filted.items():
        entity_filted[entity] = sorted(events, key=lambda x: x[1])

    entity_res = dict()
    for entity, events in entity_filted.items():
        entity_res[entity_map[entity]] = list(map(lambda x: [event_map[x[0]], time_map[x[1]]], events))

    time_max = set()
    time_max.add(0)

    return entity_res, len(entity_set), len(event_set), max(time_max)


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def adapt_time(seq, time_seq):
    ok = 0
    ans = []
    pre = 0
    for id, event_id in enumerate(seq):
        if event_id == 0:
            ans.append(0)
        else:
            if ok:
                ans.append(min((time_seq[id] - pre, 1000)) + ans[-1])
                pre = time_seq[id]
            else:
                ans.append(0)
                pre = time_seq[id]
                ok = 1
    return np.array(ans, dtype=np.float32)

def sample_function(entity_train, entitynum, eventnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        entity = np.random.randint(1, entitynum + 1)
        while len(entity_train[entity]) <= 1: entity = np.random.randint(1, entitynum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.float32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = entity_train[entity][-1][0]
        target_time = entity_train[entity][-1][1] - entity_train[entity][-2][1]
        idx = maxlen - 1

        ts = set(map(lambda x: x[0], entity_train[entity]))
        for i in reversed(entity_train[entity][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, eventnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_seq = adapt_time(seq,time_seq)
        return (entity, seq, time_seq, pos, neg, target_time)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, entity, entitynum, eventnum, batch_size=64, maxlen=10,n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            np.random.seed(i)
            self.processors.append(
                Process(target=sample_function, args=(entity,
                                                      entitynum,
                                                      eventnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def sample_function_point(entity_train, entitynum, eventnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        entity = np.random.randint(1, entitynum + 1)
        while len(entity_train[entity]) <= 1: entity = np.random.randint(1, entitynum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.float32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = entity_train[entity][-1][0]
        target_time = entity_train[entity][-1][1] - entity_train[entity][-2][1]
        idx = maxlen - 1

        ts = set(map(lambda x: x[0], entity_train[entity]))
        for i in reversed(entity_train[entity][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, eventnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_seq = adapt_time(seq, time_seq)
        return (entity, seq, time_seq, pos, neg, target_time)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler_point(object):
    def __init__(self, entity, entitynum, eventnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            np.random.seed(i)
            self.processors.append(
                Process(target=sample_function_point, args=(entity,
                                                      entitynum,
                                                      eventnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()