import time
import torch

from utils.net_util import gpuify
from models.model_io import ModelInput
from .agent import ThorAgent

class SemanticAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, gpu_id):
        max_episode_length = args.max_episode_length
        hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        from utils.class_finder import episode_class

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)

        super(SemanticAgent, self).__init__(
            create_model(args), args, rank, episode, max_episode_length, gpu_id
        )
        self.hidden_state_sz = hidden_state_sz
        self.gpu_id = gpu_id
        self.args = args
    def eval_at_state(self):
        model_input = ModelInput()
        model_input.gpu_id = self.gpu_id
        if self.episode.current_objs is None:
            model_input.objbb = self.objstate()
        else:
            model_input.objbb = self.episode.current_objs
        model_input.hidden = self.hidden
        model_input.flag = self.flag
        model_input.target_class_embedding = self.episode.glove_embedding
        model_input.action_probs = self.last_action_probs
        model_input.target_name = self.episode.target_object
        model_input.pose = [self.x, self.y, self.ha, self.pa]
        return model_input, self.model.forward(model_input)
    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):
        self.flag = 0
        self.x = 0
        self.y = 0
        self.ha = 0
        self.pa = 0
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.hidden = (

                    torch.zeros(1, self.hidden_state_sz).cuda(),
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                )
        else:
            self.hidden = (

                torch.zeros(1, self.hidden_state_sz),
                torch.zeros(1, self.hidden_state_sz),
            )
        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )

    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass

    def objstate(self):
        return self.episode.objstate_for_agent()
