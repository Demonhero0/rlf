import sys
import numpy
import torch
import random
import argparse
import logging
import time

from .fuzzers import Environment
from .fuzzers.random import PolicyRandom, ObsRandom
from .fuzzers.F_random import PolicyFRandom, ObsFRandom
from .fuzzers.reinforcement import PolicyReinforcement, ObsReinforcement
from .execution import Execution
from .common import set_logging

import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution', dest='execution', type=str, default='./execution.so') # the Dynamic library
    parser.add_argument('--proj', dest='proj', type=str, default=None) # the backend of fuzz 
    parser.add_argument('--contract', dest='contract', type=str, default=None) # the contract to test
    parser.add_argument('--limit', dest='limit', type=int, default=100)
    # parser.add_argument('--limit_time', dest='limit_time', type=int, default=60)
    parser.add_argument('--fuzzer', dest='fuzzer', choices=['random', 'frandom', 'reinforcement'], default='random') # the mode of fuzzer symbolic=training

    parser.add_argument('--model', dest='model', type=str, default='model_imitation') # the model used to fuzz

    parser.add_argument('--seed', dest='seed', type=int, default=1)
    parser.add_argument('--log_to_file', dest='log_to_file', type=str, default=None)
    parser.add_argument('-v', dest='v', type=int, default=1, metavar='LOG_LEVEL',
                        help='Log levels: 0 - NOTSET, 1 - INFO, 2 - DEBUG, 3 - ERROR')

    parser.add_argument('--train_dir', dest='train_dir', type=str, default=None) # the train data used to fuzz
    parser.add_argument('--dataset_dump_path', dest='dataset_dump_path', type=str, default=None) # the export path of train data

    parser.add_argument('--output_path', dest='output_path', type=str, default=None)
    parser.add_argument('--address', dest='address', type=str, default=None)
    parser.add_argument('--mode', dest='mode', type=str, default='test')
    parser.add_argument('--max_episode', dest='max_episode', type=int, default=50)
    parser.add_argument('--reward', dest='reward', choices=['cov','cov+bugs','bugs'], default=None)
    parser.add_argument('--rl_model', dest='rl_model', type=str, default='model_dqn')
    parser.add_argument('--bug_rate', dest='bug_rate', type=float, default=0.5)
    parser.add_argument('--detect_bugs', dest='detect_bugs', choices=['all','ls'], default=None)
    parser.add_argument('--limit_time', dest='limit_time', type=int, default=1800)
    args = parser.parse_args()
    return args


def init(args):
    random.seed(args.seed)
    set_logging(args.v, args.log_to_file)
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    sys.setrecursionlimit(8000)

class Args(object):

    def __init__(self, arg_dict):
        self.execution = arg_dict.get('execution','./execution.so')
        self.proj = arg_dict.get('proj',None)
        self.contract = arg_dict.get('contract',None)
        self.limit = arg_dict.get('limit',100)
        self.fuzzer = arg_dict.get('fuzzer','random')
        self.model = arg_dict.get('model',None)
        self.seed = arg_dict.get('seed',1)
        self.log_to_file = arg_dict.get('log_to_file',None)
        self.v = arg_dict.get('v',1)
        self.train_dir = arg_dict.get('train_dir',None)
        self.dataset_dump_path = arg_dict.get('dataset_dump_path',None)
        self.address = arg_dict.get('address',None)


def main():
    start_time = time.time()
    args = get_args()
    init(args)
    print(args)

    LOG = logging.getLogger(__name__) # initialize the LOG
    LOG.info('fuzzing start')

    if args.proj is not None:
        execution = Execution(args.execution)
        backend_loggers = execution.set_backend(args.proj)
        contract_manager = execution.get_contracts()
        if args.contract is not None:
            contract_manager.set_fuzz_contracts([args.contract])
        account_manager = execution.get_accounts()

    if args.fuzzer == 'random':
        print('fuzzer random')
        policy = PolicyRandom(execution, contract_manager, account_manager)
        obs = ObsRandom(contract_manager, account_manager, args.dataset_dump_path)
    elif args.fuzzer == 'frandom':
        print('fuzzer frandom')
        policy = PolicyFRandom(execution, contract_manager, account_manager)
        obs = ObsFRandom(contract_manager, account_manager, args.dataset_dump_path)
    elif args.fuzzer == 'reinforcement':
        policy = PolicyReinforcement(execution, contract_manager, account_manager, args)
        if args.mode == 'train':
            print('train mode')
        policy.load_model()
        obs = ObsReinforcement(contract_manager, account_manager, args.dataset_dump_path)

    environment = Environment(args.limit, args.seed, args.max_episode, start_time)
    if args.fuzzer == 'reinforcement':
        result, count_dict = environment.fuzz_loop_RL(policy, obs, start_time, args)
    else:
        result = environment.fuzz_loop(policy, obs)

    end_time = time.time()

    result['fuzzer'] = args.fuzzer
    result['start_time'] = start_time
    result['time'] = end_time - start_time
    result['end_time'] = end_time
    result['final'] = obs.stat.export_result()
    if args.output_path:
        with open(f'{args.output_path}/{args.address}.json','w') as f:
            json.dump(result,f)
    print('fuzz_output')

if __name__ == '__main__':
    main()