import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # hardware config
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ids to use (separated by comma e.g. 0,1,2,3)")
    parser.add_argument("--num_workers", type=int, default=0, help="number of dataloader workers")
    # training config
    parser.add_argument("--method", type=str, default="jdot", choices=["jdot", "jumbot", "jpmbot"], help="model name")
    parser.add_argument("--use_bomb", action="store_true", help="whether to use BomB version")
    parser.add_argument("--source_ds", type=str, default="svhn", help="The source dataset")
    parser.add_argument("--target_ds", type=str, default="mnist", help="The target dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--k", type=int, default=1, help="number of minibatches")
    parser.add_argument("--mbsize", type=int, default=500, help="minibatch size")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epoch at k=1")
    parser.add_argument("--test_interval", type=int, default=1, help="interval of two continuous test phase")
    parser.add_argument("--nclass", type=int, default=10, help="number of classes")
    parser.add_argument("--epsilon", type=float, default=0, help="OT regularization coefficient")
    parser.add_argument(
        "--batch_epsilon", type=float, default=0.0, help="OT regularization coefficient between minibatches"
    )
    parser.add_argument("--tau", type=float, default=1, help="marginal penalization coeffidient")
    parser.add_argument("--mass", type=float, default=1,
                        help="transported mass s (partial OT).  On Digits the m-KPOT runs "
                             "override this per task -- 0.85 / 0.90 / 0.80 for "
                             "SVHN->MNIST / USPS->MNIST / MNIST->USPS -- see "
                             "sh/train_KPG_mPOT.sh.  Every other benchmark uses 0.65.")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=1980, help="random seed")
    parser.add_argument("--eta1", type=float, default=0.1, help="weight of embedding loss")
    parser.add_argument("--eta2", type=float, default=0.1, help="weight of transportation loss")

    # ---- KPG-RL parameters ------------------------------------------------
    parser.add_argument("--use_kpg", action="store_true", help="enable KPG-RL keypoint-guided OT")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="KPG-RL-KP blending: cost = alpha * C_norm + (1-alpha) * G. "
             "(1 = pure standard cost, 0 = pure guiding cost)",
    )

    # ---- Sec. IV formulation ------------------------------
    parser.add_argument(
        "--kp_strategy", type=str, default="random",
        choices=["centroid", "random", "farthest"],
        help="Keypoint-selection strategy (Sec. V-A-3).  "
             "'centroid' = medoid of the true target class (ORACLE, uses GT "
             "target labels), 'random' = random pseudo-class sample (practical), 'farthest' "
             "= pseudo-class sample farthest from its centroid (adversarial).",
    )
    parser.add_argument(
        "--kp_per_class", type=int, default=1,
        help="keypoint pairs per class; k = kp_per_class x n_class and must satisfy k < m",
    )
    parser.add_argument(
        "--rho", type=float, default=0.1,
        help="dimensionless relation-profile temperature of Eq. (8); the softmax scale is "
             "rho * max(c).",
    )
    parser.add_argument(
        "--kp_metric", type=str, default="euclidean",
        choices=["euclidean", "sqeuclidean"],
        help="ground metric for the relation profiles of Eq. (8)",
    )
    parser.add_argument(
        "--kp_probe", type=int, default=4096,
        help="samples per domain encoded once to choose the fixed keypoints",
    )
    parser.add_argument(
        "--kuot_eps", type=float, default=0.01,
        help="entropic regularisation of Eq. (12) used by the derived scaling solver when "
             "--epsilon is 0 (unbalanced regime only; balanced/partial keep --epsilon)",
    )
    parser.add_argument(
        "--kuot_iters", type=int, default=1000,
        help="scaling iterations N in Alg. 1",
    )

    args = parser.parse_args()

    return args
