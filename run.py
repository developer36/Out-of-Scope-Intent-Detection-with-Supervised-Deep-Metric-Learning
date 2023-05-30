import sys
import os
import logging
import argparse
import datetime
from config.config import ParamManager
from dataloader.base import DataManager
from model.manager import DRTManager
from model.base import TextRepresentation
from utils.functions import save_results

# from manager import ADBManager


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logger_name', type=str, default='Detection', help="Logger name for open intent detection.")

    parser.add_argument('--log_dir', type=str, default='logs', help="Logger directory.")

    parser.add_argument("--dataset", default='banking', type=str, help="The name of the dataset to train selected")

    parser.add_argument("--epsilon", default=1.0, type=float)

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")

    parser.add_argument("--labeled_ratio", default=1.0, type=float,
                        help="The ratio of labeled samples in the training set")

    parser.add_argument("--train", action="store_true", help="Whether to train the model")

    parser.add_argument("--pretrain", action="store_true", help="Whether to pre-train the model")

    parser.add_argument("--save_model", action="store_true", help="save trained-model for open intent detection")

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--data_dir", default=sys.path[0] + './data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--output_dir", default='./saved_models', type=str,
                        help="The output directory where all train data will be written.")

    parser.add_argument("--pretrain_model_dir", default='./', type=str,
                        help="The pretrain model directory.")

    parser.add_argument("--model_dir", default='models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--load_pretrained_method", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--result_dir", type=str, default='results', help="The path to save results")

    parser.add_argument("--results_file_name", type=str, default='results.csv',
                        help="The file name of all the results.")

    parser.add_argument("--save_results", action="store_true", help="save final results for open intent detection")

    parser.add_argument("--loss_fct", default="CrossEntropyLoss", help="The loss function for training.")

    parser.add_argument("--dataset_neg", default="SQUAD", help="")
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=50, type=int)
    parser.add_argument("--margin", default=1.0, type=float, help="TripletLoss hyper parameter")

    args = parser.parse_args()

    return args


def set_logger(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"{args.dataset}_{args.known_cls_ratio}_{args.labeled_ratio}_{time}.log"

    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.log_dir, file_name))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def run(args, data, text_encoder):
    model = DRTManager(args, data, text_encoder, logger_name=args.logger_name)
    if args.train:
        print('training begin...')
        model.train(args, data)

    print('testing begin...')
    outputs = model.test(args, data)

    if args.save_results:
        print('Results saved in %s', str(os.path.join(args.result_dir, args.results_file_name)))
        save_results(args, outputs)


if __name__ == '__main__':
    sys.path.append('.')
    args = parse_arguments()
    param = ParamManager(args)
    args = param.args
    # print(args)
    print('='*60)
    print('data loading')
    data = DataManager(args, logger_name=args.logger_name)
    print('text representation model loading')
    text_encoder = TextRepresentation(args, data, logger_name=args.logger_name)

    run(args, data, text_encoder)
