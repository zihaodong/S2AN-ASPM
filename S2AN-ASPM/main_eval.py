from __future__ import print_function, division

import os
import torch
import torch.multiprocessing as mp
import time
import numpy as np
import random

from tqdm import tqdm
from utils.net_util import ScalarMeanTracker
from runners import a3c_val, a3c_val_unseen,a3c_val_seen
from utils.misc_util import  write_json_data
os.environ["OMP_NUM_THREADS"] = "1"
def main_eval(args, create_shared_model, init_agent, load_model):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = args.load_model
    processes = []
    res_queue = mp.Queue()
    args.num_steps = 100
    target = a3c_val

    rank = 0
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                250,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()
    train_scalars_ba = ScalarMeanTracker()
    train_scalars_be = ScalarMeanTracker()
    train_scalars_k = ScalarMeanTracker()
    train_scalars_l = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if (args.scene_types[end_count] == 'bathroom'):
                train_scalars_ba.add_scalars(train_result)
            if (args.scene_types[end_count] == 'bedroom'):
                train_scalars_be.add_scalars(train_result)
            if (args.scene_types[end_count] == 'kitchen'):
                train_scalars_k.add_scalars(train_result)
            if (args.scene_types[end_count] == 'living_room'):
                train_scalars_l.add_scalars(train_result)
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)
        tracked_means = train_scalars.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()
        pbar.close()

    model = 'normal' + load_model.split('_')[-4]
    result_file = os.path.join('result', args.model, "normal", model + '.json')
    write_json_data(tracked_means, result_file)

def main_eval_seen(args, create_shared_model, init_agent, load_model):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = load_model
    processes = []
    res_queue = mp.Queue()
    args.num_steps = 50
    target = a3c_val_seen

    rank = 0
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                250,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()
    train_scalars_ba = ScalarMeanTracker()
    train_scalars_be = ScalarMeanTracker()
    train_scalars_k = ScalarMeanTracker()
    train_scalars_l = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if (args.scene_types[end_count] == 'bathroom'):
                train_scalars_ba.add_scalars(train_result)
            if (args.scene_types[end_count] == 'bedroom'):
                train_scalars_be.add_scalars(train_result)
            if (args.scene_types[end_count] == 'kitchen'):
                train_scalars_k.add_scalars(train_result)
            if (args.scene_types[end_count] == 'living_room'):
                train_scalars_l.add_scalars(train_result)
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)

        tracked_means = train_scalars.pop_and_reset()


    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()
        pbar.close()
    model = 'seen' + load_model.split('_')[-4]
    result_file = os.path.join('result', args.model, "seen", model +'.json')

    write_json_data(tracked_means, result_file)

    return 

def main_eval_unseen(args, create_shared_model, init_agent, load_model):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = load_model
    processes = []
    res_queue = mp.Queue()
    args.num_steps = 50
    target = a3c_val_unseen
    rank = 0

    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                250,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()
    train_scalars_ba = ScalarMeanTracker()
    train_scalars_be = ScalarMeanTracker()
    train_scalars_k = ScalarMeanTracker()
    train_scalars_l = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if (args.scene_types[end_count] == 'bathroom'):
                train_scalars_ba.add_scalars(train_result)
            if (args.scene_types[end_count] == 'bedroom'):
                train_scalars_be.add_scalars(train_result)
            if (args.scene_types[end_count] == 'kitchen'):
                train_scalars_k.add_scalars(train_result)
            if (args.scene_types[end_count] == 'living_room'):
                train_scalars_l.add_scalars(train_result)
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)
        tracked_means = train_scalars.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()
        pbar.close()
    model = 'unseen' + load_model.split('_')[-4]
    result_file = os.path.join('result', args.model, "unseen", model + '.json')
    write_json_data(tracked_means, result_file)




