from __future__ import print_function, division

import os
import random
import ctypes
import time
import numpy as np
import torch
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
from utils import flag_parser
from utils.class_finder import model_class, agent_class, optimizer_class
from utils.net_util import ScalarMeanTracker
from runners import a3c_train, a3c_train_seen

def main_train(args):

    args = flag_parser.parse_arguments()
    model_success = 0
    model_spl = 0

    if args.zsd:
        print("use zsd setting !")
        args.num_steps = 100
        target = a3c_train_seen
    else:
        args.num_steps = 100
        target = a3c_train

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)
    optimizer_type = optimizer_class(args.optimizer)

    start_time = time.time()
    local_start_time_str = time.strftime(
        "%Y-%m-%d_%H:%M:%S", time.localtime(start_time)
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.log_dir is not None:
        tb_log_dir = args.log_dir + "/" + args.title + "/" + args.title + "-" + local_start_time_str
        log_writer = SummaryWriter(log_dir=tb_log_dir)
    else:
        log_writer = SummaryWriter(comment=args.title)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn", force=True)
    shared_model = create_shared_model(args).to("cpu")

    train_total_ep = 0
    n_frames = 0
    if shared_model is not None:
        shared_model.share_memory()
        optimizer = optimizer_type(
            filter(lambda p: p.requires_grad, shared_model.parameters()), args
        )
        optimizer.share_memory()
        print(shared_model)
    else:
        assert (
                args.agent_type == "RandomNavigationAgent"
        ), "The model is None but agent is not random agent"
        optimizer = None

    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)

    train_res_queue = mp.Queue()

    if args.random_object:
        print("使用random shuffle")

    if args.order_object:
        print("使用order object")

    for rank in range(0, args.workers):
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                create_shared_model,
                shared_model,
                init_agent,
                optimizer,
                train_res_queue,
                end_flag,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print("Train agents created.")

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    args.save_model_dir = os.path.join(args.save_model_dir, args.title)
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    save_entire_model = 0
    try:
        time_start = time.time()
        reward_avg_list = []
        ep_length_avg_list = []
        success_list = []
        spl_list = []
        while train_total_ep < args.max_ep:
            train_result = train_res_queue.get()
            reward_avg_list.append(train_result["total_reward"])
            success_list.append(train_result["success"])
            spl_list.append(train_result["spl"])
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result["ep_length"]
            ep_length_avg_list.append(train_result["ep_length"])

            if (train_total_ep % train_thin) == 0:
                log_writer.add_scalar("n_frames", n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + "/train", tracked_means[k], train_total_ep
                    )

                    if (k == 'success') and tracked_means[k] > model_success and tracked_means[k] > 0:
                        save_success_dir = os.path.join(args.save_model_dir, "success")
                        if not os.path.exists(save_success_dir):
                            os.makedirs(save_success_dir)
                        state_to_save = shared_model.state_dict()
                        save_path = os.path.join(
                            save_success_dir,
                            "{0}_{1}_success_{2}_.dat".format(
                                args.model, train_total_ep, tracked_means[k]
                            ),
                        )
                        model_success = tracked_means[k]
                        torch.save(state_to_save, save_path)

                    if (k == 'spl') and tracked_means[k] > model_spl and tracked_means[k] > 0:
                        save_success_dir = os.path.join(args.save_model_dir, "spl")
                        if not os.path.exists(save_success_dir):
                            os.makedirs(save_success_dir)
                        state_to_save = shared_model.state_dict()
                        save_path = os.path.join(
                            save_success_dir,
                            "{0}_{1}_spl_{2}_.dat".format(
                                args.model, train_total_ep, tracked_means[k]
                            ),
                        )
                        model_spl = tracked_means[k]
                        torch.save(state_to_save, save_path)


            if (train_total_ep % args.ep_save_freq) == 0:
                print(n_frames)
                if not os.path.exists(args.save_model_dir):
                    os.makedirs(args.save_model_dir)

                save_am = os.path.join(args.save_model_dir,'am')
                if not os.path.exists(save_am):
                    os.makedirs(save_am)
                save_a3c = os.path.join(args.save_model_dir,'a3c')
                if not os.path.exists(save_a3c):
                    os.makedirs(save_a3c)

                state_to_save = shared_model.state_dict()
                save_path = os.path.join(
                    args.save_model_dir,
                    "{0}_{1}_{2}_{3}.dat".format(
                        args.model, train_total_ep, n_frames, local_start_time_str
                    ),
                )

                torch.save(state_to_save, save_path)
                save_entire_model += 1
                if (save_entire_model % 5) == 0:
                    state = {
                        'epoch': train_total_ep,
                        'state_dict': shared_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }

                    save_model_path = os.path.join(
                        args.save_model_dir,
                        "{0}_{1}_{2}.tar".format(
                            args.model, train_total_ep, local_start_time_str
                        ),
                    )
                    torch.save(state, save_model_path)
                    save_entire_model = 0

            if train_total_ep % 100 == 0:
                time_end = time.time()
                seconds = round(time_end - time_start)
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                reward_avg = sum(reward_avg_list) / len(reward_avg_list)
                success_avg = sum(success_list) / len(success_list)
                spl_avg = sum(spl_list) / len(spl_list)
                ep_length_avg = sum(ep_length_avg_list) / len(ep_length_avg_list)
                percent_complete = train_total_ep / args.max_ep * 100

                time_per_ep = seconds / train_total_ep
                remaining_ep = args.max_ep - train_total_ep
                remaining_time = remaining_ep * time_per_ep
                rem_h, rem_m = divmod(remaining_time // 60, 60)
                rem_s = remaining_time % 60

                print(
                    "epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] success:[{:.2f}] spl: [{:.2f}] reward:[{:.2f}] ep_length:[{:.1f} pc:[{:.6f}] reming time:[{:.0f}:{:.0f}:{:.0f}]"
                    .format(train_total_ep, args.max_ep, h, m, s,success_avg,spl_avg, reward_avg, ep_length_avg, percent_complete, rem_h,
                            rem_m,
                            rem_s))
                reward_avg_list = []
                ep_length_avg_list = []
                success_list = []
                spl_list = []
                save_path = os.path.join(args.save_model_dir, "{0}_{1}.txt".format(args.model, local_start_time_str))
                f = open(save_path, "a")
                if train_total_ep == 100:
                    f.write(str(args))
                    f.write("\n")
                    f.write("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]\n"
                            .format(train_total_ep, args.max_ep, h, m, s, reward_avg, ep_length_avg))
                else:
                    f.write("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]\n"
                            .format(train_total_ep, args.max_ep, h, m, s, reward_avg, ep_length_avg))
                f.close()
    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()