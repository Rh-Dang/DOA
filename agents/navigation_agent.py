import torch
import numpy as np
import h5py

from utils.model_util import gpuify, toFloatTensor
from models.model_io import ModelInput

from .agent import ThorAgent


class NavigationAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, scenes, targets, gpu_id):
        max_episode_length = args.max_episode_length   
        hidden_state_sz = args.hidden_state_sz   
        self.action_space = args.action_space   
        from utils.class_finder import episode_class    

        episode_constructor = episode_class(args.episode_type)   
        episode = episode_constructor(args, gpu_id, args.strict_done)   

        super(NavigationAgent, self).__init__(
            create_model(args), args, rank, scenes, targets, episode, max_episode_length, gpu_id
        )
        self.hidden_state_sz = hidden_state_sz
        self.keep_ori_obs = args.keep_ori_obs

        self.glove = {}
        if 'SP' in self.model_name:
            with h5py.File('/home/dhm/Code/vn/glove_map300d.hdf5', 'r') as rf:
                for i in rf:
                    self.glove[i] = rf[i][:]
        
        self.num_cate = args.num_category

    def eval_at_state(self, model_options):
        model_input = ModelInput()

        # model inputs
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame

        model_input.hidden = self.hidden

        model_input = self.process_detr_input(model_input)
        
        model_input.action_probs = self.last_action_probs

        model_input.scene = self.episode.environment.scene_name
        model_input.target_object = self.episode.target_object

        if 'Memory' in self.model_name:
            state_length = self.hidden_state_sz

            if len(self.episode.state_reps) == 0:
                model_input.states_rep = torch.zeros(1, state_length)
            else:
                model_input.states_rep = torch.stack(self.episode.state_reps)

            dim_obs = 512
            if len(self.episode.obs_reps) == 0:
                model_input.obs_reps = torch.zeros(1, dim_obs)
            else:
                model_input.obs_reps = torch.stack(self.episode.obs_reps)

            if len(self.episode.state_memory) == 0:
                model_input.states_memory = torch.zeros(1, state_length)
            else:
                model_input.states_memory = torch.stack(self.episode.state_memory)

            if len(self.episode.action_memory) == 0:
                model_input.action_memory = torch.zeros(1, 6)
            else:
                model_input.action_memory = torch.stack(self.episode.action_memory)

            model_input.states_rep = toFloatTensor(model_input.states_rep, self.gpu_id)
            model_input.states_memory = toFloatTensor(model_input.states_memory, self.gpu_id)
            model_input.action_memory = toFloatTensor(model_input.action_memory, self.gpu_id)
            model_input.obs_reps = toFloatTensor(model_input.obs_reps, self.gpu_id)

        return model_input, self.model.forward(model_input, model_options)

    def process_detr_input(self, model_input):
        # process detection features from DETR detector
        current_detection_feature = self.episode.current_detection_feature() 

        zero_detect_feats = np.zeros([22, 264]) 

        
        for cate_id in range(len(self.targets) + 1):   
            cate_index = current_detection_feature[:, 257] == cate_id  
            if cate_index.sum() > 0:                                 
                index = current_detection_feature[cate_index, 256].argmax(0)  
                zero_detect_feats[cate_id - 1, :] = current_detection_feature[cate_index, :][index]  
                
        current_detection_feature = zero_detect_feats
        detection_inputs = {
            'features': current_detection_feature[:, :256],
            'scores': current_detection_feature[:, 256],
            'labels': current_detection_feature[:, 257], 
            'bboxes': current_detection_feature[:, 260:],
            'target': self.targets.index(self.episode.target_object),
        }

        # generate target indicator array based on detection results labels
        target_embedding_array = np.zeros((detection_inputs['features'].shape[0], 1))
        target_embedding_array[
            detection_inputs['labels'][:] == (self.targets.index(self.episode.target_object) + 1)] = 1
        detection_inputs['indicator'] = target_embedding_array 

        detection_inputs = self.dict_toFloatTensor(detection_inputs)

        model_input.target_class_embedding = detection_inputs 

        return model_input

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):
        with torch.cuda.device(self.gpu_id):
            self.hidden = (
                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
            )

        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )

        #self.model.reset()

    def repackage_hidden(self):
        if self.hidden != None:                            
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass

    def dict_toFloatTensor(self, dict_input):

        for key in dict_input:
            dict_input[key] = toFloatTensor(dict_input[key], self.gpu_id)

        return dict_input
