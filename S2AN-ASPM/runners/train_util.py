from __future__ import division
import torch

def run_episode(player, args, total_reward, training):
    num_steps = args.num_steps
    i = 0
    for _ in range(num_steps):
        player.action(training)
        i += 1
        total_reward = total_reward + player.reward
        if player.done:
            break
    return total_reward

def new_episode(
    args,
    player,
    scenes,
    possible_targets=None,
    targets=None,
    rooms=None,
    keep_obj=False,
    glove=None,
):
    if args.random_object:
        player.model.random_self_objects()
    player.episode.new_episode(args, scenes, possible_targets, targets, rooms, keep_obj, glove)
    player.reset_hidden()
    player.done = False

def a3c_loss(args, player, gpu_id):
    """ Borrowed from https://github.com/dgriff777/rl_a3c_pytorch. """
    R = torch.zeros(1, 1)
    if not player.done:
        _, output = player.eval_at_state()
        R = output.value.data

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            R = R.cuda()

    player.values.append(R)
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            gae = gae.cuda()

    for i in reversed(range(len(player.rewards))):
        R = args.gamma * R + player.rewards[i]
        advantage = R - player.values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = (
            player.rewards[i]
            + args.gamma * player.values[i + 1].data
            - player.values[i].data
        )

        gae = gae * args.gamma * args.tau + delta_t

        policy_loss = (
            policy_loss
            - player.log_probs[i] * gae
            - args.beta * player.entropies[i]
        )
    return policy_loss, value_loss

def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for param, shared_param in zip(
        player.model.parameters(), shared_model.parameters()
    ):
        if shared_param.requires_grad:
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()

def reset_player(player):
    player.clear_actions()
    player.repackage_hidden()

def compute_loss(args, player, gpu_id):
    policy_loss, value_loss = a3c_loss(args, player, gpu_id)
    total_loss = policy_loss + 0.5 * value_loss
    return dict(total_loss=total_loss, policy_loss=policy_loss, value_loss=value_loss)


def end_episode(
    player, res_queue, title=None, episode_num=0, include_obj_success=False, **kwargs
):

    results = {
        "done_count": player.episode.done_count,
        "ep_length": player.eps_len,
        "success": int(player.success),
    }
    #print(results)

    results.update(**kwargs)
    res_queue.put(results)


def get_bucketed_metrics(spl, best_path_length, success):
    out = {}
    for i in [1, 5]:
        if best_path_length >= i:
            out["GreaterThan/{}/success".format(i)] = success
            out["GreaterThan/{}/spl".format(i)] = spl
    return out

def get_bucketed_metrics_all(spl, best_path_length, success, dts,sspl):
    out = {}
    for i in [1, 5]:
        if best_path_length >= i:
            out["GreaterThan/{}/success".format(i)] = success
            out["GreaterThan/{}/spl".format(i)] = spl
            out["GreaterThan/{}/dts".format(i)] = dts
            out["GreaterThan/{}/sspl".format(i)] = sspl

    return out

def compute_spl(player, start_state):
    best = float("inf")
    for obj_id in player.episode.task_data:
        try:
            _, best_path_len, _ = player.environment.controller.shortest_path_to_target(
                start_state, obj_id, False
            )
            if best_path_len < best:
                best = best_path_len
        except:
            # This is due to a rare known bug.
            continue

    if not player.success:
        return 0, best

    if best < float("inf"):
        return best / float(player.eps_len), best

    # This is due to a rare known bug.
    return 0, best

def compute_sspl(player, start_state,ITR,FTR):

    best = float("inf")
    for obj_id in player.episode.task_data:
        try:
            _, best_path_len, _ = player.environment.controller.shortest_path_to_target(
                start_state, obj_id, False
            )
            if best_path_len < best:
                best = best_path_len
        except:
            # This is due to a rare known bug.
            continue
    enr = ITR - FTR
    if best < float("inf") and ITR != 0 and player.eps_len >= best:
            return (best / float(player.eps_len ) )* (max(ITR + enr,0)/ (2*ITR))
    return 0.5
