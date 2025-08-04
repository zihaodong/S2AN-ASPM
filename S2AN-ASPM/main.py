from __future__ import print_function, division

import os
from utils import flag_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval,main_eval_seen,main_eval_unseen
from main_train import main_train

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():

    args = flag_parser.parse_arguments()
    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    if args.eval:
        if args.zsd:
            print("Evaluate Unseen Classes !")
            main_eval_unseen(args, create_shared_model, init_agent, args.load_model)
            print("#######################################################")
            print("Evaluate Seen Classes !")
            main_eval_seen(args, create_shared_model, init_agent, args.load_model)

            return
        else:
            main_eval(args, create_shared_model, init_agent, args.load_model)
            return

    main_train(args)


if __name__ == "__main__":
    main()
