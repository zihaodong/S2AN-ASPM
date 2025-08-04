""" Base class for all Agents. """
from __future__ import division

import time
import torch
import torch.nn.functional as F
import math

from datasets.constants import DONE_ACTION_INT




class ThorAgent:
    """ Base class for all actor-critic agents. """

    def __init__(
        self, model, args, rank, episode=None, max_episode_length=1e3, gpu_id=-1
    ):
        self.gpu_id = gpu_id
        self._model = None
        self.model = model
        self._episode = episode
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.max_length = False
        self.hidden = None
        self.actions = []
        self.probs = []
        self.max_episode_length = max_episode_length
        self.success = False
        self.flag = 0
        self.verbose = args.verbose
        self.last_action_probs = None
        self.hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        self.vis = args.vis
        self.results_json = args.results_json
        self.x = 0
        self.y = 0
        self.ha = 0
        self.pa = 0

        torch.manual_seed(args.seed + rank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())

    def eval_at_state(self):
        """ Eval at state. """
        raise NotImplementedError()

    @property
    def episode(self):
        """ Return the current episode. """
        return self._episode

    @property
    def environment(self):
        """ Return the current environmnet. """
        return self.episode.environment

    @property
    def state(self):
        """ Return the state of the agent. """
        raise NotImplementedError()

    @state.setter
    def state(self, value):
        raise NotImplementedError()

    @property
    def model(self):
        """ Returns the model. """
        return self._model

    def print_info(self):
        """ Print the actions. """
        for action in self.actions:
            print(action)

    @model.setter
    def model(self, model_to_set):
        self._model = model_to_set
        if self.gpu_id >= 0 and self._model is not None:
            with torch.cuda.device(self.gpu_id):
                self._model = self.model.cuda()

    def _increment_episode_length(self):
        self.eps_len += 1
        if self.eps_len >= self.max_episode_length:
            if not self.done:
                self.max_length = True
                self.done = True
            else:
                self.max_length = False
        else:
            self.max_length = False

    def move_pose(self,action):
        if self.info:
            if action == 0:
                if self.ha == 0:
                    self.y += 0.25
                elif self.ha == 45:
                    self.x += 0.125 * math.sqrt(2)
                    self.y += 0.125 * math.sqrt(2)
                elif self.ha == 90:
                    self.x += 0.25
                elif self.ha == 135:
                    self.x += 0.125 * math.sqrt(2)
                    self.y -= 0.125 * math.sqrt(2)
                elif self.ha == 180:
                    self.y -= 0.25
                elif self.ha == -45:
                    self.x -= 0.125 * math.sqrt(2)
                    self.y += 0.125 * math.sqrt(2)
                elif self.ha == -90:
                    self.x -= 0.25
                elif self.ha == -135:
                    self.x -= 0.125 * math.sqrt(2)
                    self.y -= 0.125 * math.sqrt(2)
            elif action == 1:
                self.ha -= 45
                if self.ha == -180:
                    self.ha = 180
            elif action == 2:
                self.ha += 45
                if self.ha == 225:
                    self.ha = -135
            elif action == 3:
                self.pa += 45
            elif action == 4:
                self.pa -= 45

    def action(self,  training, demo=False):
        """ Train the agent. """
        if training:
            self.model.train()
        else:
            self.model.eval()

        model_input, out = self.eval_at_state()
        self.hidden = out.hidden
        self.flag = 1
        prob = F.softmax(out.logit, dim=1)
        action = prob.multinomial(1).data
        log_prob = F.log_softmax(out.logit, dim=1)
        self.last_action_probs = prob
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, action)
        sim = out.sim
        viewpoint = out.viewpoint
        self.reward, self.done, self.info = self.episode.step(action[0, 0],sim,viewpoint)
        self.move_pose(action[0,0])
        if self.verbose:
            print(self.episode.actions_list[action])
        self.probs.append(prob)
        self.entropies.append(entropy)
        self.values.append(out.value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.actions.append(action)
        self.episode.current_frame = self.state()


        self._increment_episode_length()
        if self.episode.strict_done and action.item() == DONE_ACTION_INT:
            self.success = self.info
            self.done = True
        elif self.done:
            self.success = not self.max_length
        return out.value, prob, action

    def reset_hidden(self, volatile=False):
        """ Reset the hidden state of the LSTM. """
        raise NotImplementedError()

    def repackage_hidden(self, volatile=False):
        """ Repackage the hidden state of the LSTM. """
        raise NotImplementedError()

    def clear_actions(self):
        """ Clear the information stored by the agent. """
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.probs = []
        self.reward = 0
        self.flag = 0
        return self

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        raise NotImplementedError()

    def exit(self):
        """ Called on exit. """
        pass

    def reset_episode(self):
        """ Reset the episode so that it is identical. """
        return self._episode.reset()
