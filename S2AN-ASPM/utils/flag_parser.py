import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="SAVN.")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.00,
        metavar="T",
        help="parameter for GAE (default: 1.00)",
    )
    parser.add_argument(
        "--beta", type=float, default=1e-2, help="entropy regularization term"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        metavar="W",
        help="how many training processes to use (default: 4)",
    )

    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=30,
        metavar="M",
        help="maximum length of an episode (default: 30)",
    )
    parser.add_argument(
        "--load_model", type=str, default="", help="Path to load a saved model."
    )

    parser.add_argument(
        "--ep_save_freq",
        type=int,
        # default=1e5,
        default=10000,
        help="save model after this # of training episodes (default: 1e+5)",
    )
    parser.add_argument(
        "--optimizer",
        default="SharedAdam",
        metavar="OPT",
        help="shared optimizer choice of SharedAdam or SharedRMSprop",
    )
    parser.add_argument(
        "--save-model-dir",
        default="./trained_models",
        metavar="SMD",
        help="folder to save trained navigation",
    )
    parser.add_argument(
        "--log-dir", default="runs/", metavar="LG", help="folder to save logs"
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        default=[0],
        nargs="+",
        help="GPUs to use [-1 CPU only] (default: -1)",
    )
    parser.add_argument(
        "--amsgrad", default=True, metavar="AM", help="Adam optimizer amsgrad parameter"
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.25,
        metavar="GS",
        help="The grid size used to discretize AI2-THOR maps.",
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="If true, output will contain more information.",
    )
    parser.add_argument(
        "--max_ep", type=float, 
        default=5000000,
        help="maximum # of episodes"
    )

    parser.add_argument("--model",
                        type=str,

                        default="ZSA_mo_split",
                        help="Model to use.")

    parser.add_argument(
        "--train_thin", type=int, default=50, help="How often to print"
    )
    parser.add_argument(
        "--local_executable_path",
        type=str,
        default=None,
        help="a path to the local thor build.",
    )

    parser.add_argument(
        "--train_scenes", type=str, default="[1-20]", help="scenes for training."
    )
    parser.add_argument(
        "--val_scenes",
        type=str,
        default="[21-30]",
        help="old validation scenes before formal split.",
    )

    parser.add_argument(
        "--possible_targets",
        type=str,
        default="FULL_OBJECT_CLASS_LIST",
        help="all possible objects.",
    )

    parser.add_argument(
        "--glove_dim",
        type=int,
        default=300,
        help="which dimension of the glove vector to use",
    )
    parser.add_argument(
        "--action_space", type=int, default=6, help="space of possible actions."
    )

    parser.add_argument(
        "--hidden_state_sz", type=int, default=512, help="size of hidden state of LSTM."
    )

    parser.add_argument("--compute_spl", action="store_true", help="compute the spl.")

    parser.add_argument("--eval", action="store_true", help="run the test code")

    parser.add_argument(
        "--offline_data_dir",
        type=str,
        default="./data/thor_v1_offline_data", #thor_offline_data
        help="where dataset is stored.",
    )
    parser.add_argument(
        "--glove_dir",
        type=str,
        default="./data/thor_glove/glove_thorv1_300.hdf5",
        help="where the glove files are stored.",
    )
    parser.add_argument(
        "--images_file_name",
        type=str,
        default="resnet18_featuremap.hdf5",
        help="Where the controller looks for images. Can be switched out to real images or Resnet features.",
    )
    parser.set_defaults(strict_done=True)

    parser.add_argument(
        "--results_json", type=str, default="metrics.json", help="Write the results."
    )

    parser.add_argument(
        "--agent_type",
        type=str,
        default="NavigationAgent",
        help="Which type of agent. Choices are NavigationAgent or RandomAgent.",
    )

    parser.add_argument(
        "--episode_type",
        type=str,
        default="BasicEpisode",
        help="Which type of agent. Choices are NavigationAgent or RandomAgent.",
    )

    parser.add_argument(
        "--fov", type=float, default=100.0, help="The field of view to use."
    )

    parser.add_argument(
        "--scene_types",
        nargs="+",
        default=["kitchen", "living_room", "bedroom", "bathroom"],
        # default=["kitchen"],
    )

    parser.add_argument('--data_dir', type=str, default='/home/wanhao/project/ZSA_END/data/gcn',
                        help="location of kg data directory")

    parser.add_argument("--test_or_val", default="test", help="test or val")
    parser.add_argument("--partial_reward",type=bool, default=False, help="using partial reward for parent objects")
    parser.add_argument("--vis",type=bool,default=False, help="whether to store action log for visualization")
    parser.add_argument("--room_results", default=False, help="whether to save results room-wise during evaluation")
    parser.add_argument("--zsd",type=bool, default=False, help="whether to use zsd setting")
    parser.add_argument("--split", default="None", help="class split")
    parser.add_argument("--seen_scene_types", nargs="+", default=["kitchen", "living_room", "bedroom", "bathroom"], help="用来测seen target")
    parser.add_argument("--seen_split", default="18/4", help="class split")
    parser.add_argument("--random_object", default=False, help="random object")
    parser.add_argument("--order_object", default=False, help="order object")
    parser.add_argument("--num", default=1)
    args = parser.parse_args()
    args.glove_file = args.glove_dir
    args.title = args.model
    return args
