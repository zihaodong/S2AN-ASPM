from __future__ import division

import time
import copy
import torch
import random
import setproctitle

from datasets.data import get_data,get_seen_data,get_change_data
from datasets.glove import Glove
from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics,
)


def a3c_train(
    rank,
    args,
    create_shared_model,
    shared_model,
    initialize_agent,
    optimizer,
    res_queue,
    end_flag,
):

    glove = Glove(args.glove_file)
     #mark2
    if args.split == "None":
        scenes, possible_targets, targets, rooms = get_data(args.scene_types, args.train_scenes)
    else:
        scenes, possible_targets, targets, rooms = get_change_data(args.scene_types, args.train_scenes,args.split)

    random.seed(args.seed + rank)
    idx = [j for j in range(len(args.scene_types))]
    random.shuffle(idx)

    setproctitle.setproctitle("Training Agent: {}".format(rank))
    
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    

    torch.cuda.set_device(gpu_id)

    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)
    compute_grad = True



    j = 0

    while not end_flag.value:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
         #mark1
        new_episode(
            args, player, scenes[idx[j]], possible_targets, targets[idx[j]], rooms[idx[j]], glove=glove
        )
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()
        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, True)
            # Compute the loss.
            loss = compute_loss(args, player, gpu_id)
            if compute_grad:
                # Compute gradient.
                player.model.zero_grad()
                loss["total_loss"].backward()

                torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
                # Transfer gradient to shared model and step optimizer.
                transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
                optimizer.step()
                # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)

        spl, best_path_length = compute_spl(player, player_start_state)
        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)
        end_episode(
            player,
            res_queue,
            title=args.scene_types[idx[j]],
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            spl=spl,
            loss = loss["total_loss"].item(),
            **bucketed_spl,
        )
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()

def a3c_train_seen(
    rank,
    args,
    create_shared_model,
    shared_model,
    initialize_agent,
    optimizer,
    res_queue,
    end_flag,
):

    glove = Glove(args.glove_file)
    print("场景类型为：{}".format(args.scene_types))
    scenes, possible_targets, targets, rooms = get_seen_data(args.scene_types, args.train_scenes,args.split)

    random.seed(args.seed + rank)
    idx = [j for j in range(len(args.scene_types))]
    random.shuffle(idx)

    setproctitle.setproctitle("Training Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.cuda.set_device(gpu_id)
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)
    compute_grad = True

    j = 0
    while not end_flag.value:
        total_reward = 0
        player.eps_len = 0
         #mark1
        new_episode(
            args, player, scenes[idx[j]], possible_targets, targets[idx[j]], rooms[idx[j]], glove=glove
        )
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, True)
            # Compute the loss.
            loss = compute_loss(args, player, gpu_id)
            if compute_grad:
                # Compute gradient.
                player.model.zero_grad()
                loss["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
                # Transfer gradient to shared model and step optimizer.
                transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
                optimizer.step()
                # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)
        spl, best_path_length = compute_spl(player, player_start_state)
        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)

        end_episode(
            player,
            res_queue,
            title=args.scene_types[idx[j]],
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            spl=spl,
            loss = loss["total_loss"].item(),
            **bucketed_spl,
        )
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()