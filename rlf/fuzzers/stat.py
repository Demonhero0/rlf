import json

from collections import OrderedDict
from .checkers import *

from ..ethereum import SELFDESTRUCT


# the state of the fuzzer
class Stat:
    """
    the state of the contract, including coverage of insn and block, bugs
    """

    def __init__(self, contract_manager, account_manager):
        self.contract_manager = contract_manager
        self.account_manager = account_manager

        self.tx_count = 0
        self.tx_count_dict = dict() # the num of calling of the contract
        for name in contract_manager.contract_dict:
            self.tx_count_dict[name] = 0 

        self.all_pcs_dict = dict() # the dict contract_name -> set(insn.pc)
        for name in contract_manager.contract_dict:
            contract = contract_manager[name]
            self.all_pcs_dict[name] = set(map(lambda insn: insn.pc, contract.insns))

        self.covered_pcs_dict = dict() # the dict contract_name -> covered_insn.pc
        self.temp_covered_pcs_dict = dict()
        for name in contract_manager.contract_dict:
            self.covered_pcs_dict[name] = set()
            self.temp_covered_pcs_dict[name] = set()

        self.covered_blocks_dict = dict()
        self.temp_covered_blocks_dict = dict()
        for name in contract_manager.contract_dict:
            self.covered_blocks_dict[name] = set()
            self.temp_covered_blocks_dict[name] = set()

        self.bug_res = dict()
        for name in contract_manager.contract_dict:
            self.bug_res[name] = dict()

        self.update_bug = dict()
        self.checkers = [
            BlockStateDep(),
            DangerousDelegatecall(contract_manager, account_manager),
            Leaking(),
            Locking(contract_manager, account_manager),
            Suicidal(contract_manager, account_manager),
            UnhandledException(),
            Reentrancy(contract_manager, account_manager)
        ]

    def reset_coverage(self):
        for name in self.contract_manager.contract_dict:
            self.temp_covered_pcs_dict[name] = set()

        for name in self.contract_manager.contract_dict:
            self.temp_covered_blocks_dict[name] = set()

    def update(self, logger):
        """
        update the coverage of insn and block
        check the bug in checkers
        """
        # print(dir(logger))
        # print(logger.trace_log_sotrage)
        destruct = False
        contract = self.contract_manager[logger.tx.contract]
        self.tx_count += 1
        self.tx_count_dict[contract.name] += 1

        call_stack = []
        call_stack.append(contract.addresses[0]) # contain the related address

        # record the pcs and block in last execution
        executed_pcs = set()
        executed_blocks = set()
        for i, log in enumerate(logger.logs):
            if i > 0:
                prev_log = logger.logs[i-1]
                if log.depth > prev_log.depth:
                    call_stack.append(prev_log.stack[-2])
                elif log.depth < prev_log.depth:
                    call_stack.pop()

            if call_stack[-1] not in self.contract_manager.address_to_contract:
                continue
            
            # print(log)
            if log.op == SELFDESTRUCT:
                destruct = True

            cur_contract = self.contract_manager.address_to_contract[call_stack[-1]]
            if log.pc in cur_contract.insn_pc_to_idx:
                executed_pcs.add(log.pc)
                self.covered_pcs_dict[cur_contract.name].add(log.pc)
                self.temp_covered_pcs_dict[cur_contract.name].add(log.pc) # temp
                insn_idx = cur_contract.insn_pc_to_idx[log.pc]
                if insn_idx in cur_contract.cfg.insn_idx_to_block:
                    block = cur_contract.cfg.insn_idx_to_block[insn_idx]
                    block.covered = True
                    executed_blocks.add(block.start_idx)
                    self.covered_blocks_dict[cur_contract.name].add(block.start_idx)
                    self.temp_covered_blocks_dict[cur_contract.name].add(block.start_idx) # temp

        if logger.tx.method not in self.contract_manager[logger.tx.contract].abi.methods_by_name:
            return 0, {}, {}

        for checker in self.checkers:
            if checker.check(logger):
                if checker.__class__.__name__ not in self.bug_res[logger.tx.contract]:
                    self.bug_res[logger.tx.contract][checker.__class__.__name__] = set()
                if checker.__class__.__name__ not in self.update_bug:
                    self.update_bug[checker.__class__.__name__] = set()
                self.bug_res[logger.tx.contract][checker.__class__.__name__].add(logger.tx.method)
                self.update_bug[checker.__class__.__name__].add(logger.tx.method)

        return destruct, executed_pcs, executed_blocks

    def clear_tx_count(self):
        self.tx_count = 1
        for name in self.tx_count_dict:
            self.tx_count_dict[name] = 1


    def to_json(self):
        contract_jsons = OrderedDict()

        contract_jsons['tx_count'] = self.tx_count
        contract_jsons['num_contracts'] = len(self.contract_manager.fuzz_contract_names)

        coverage_insns = 0
        coverage_blocks = 0
        # only deal with the fuzz_contract
        for name in self.contract_manager.fuzz_contract_names:
            all_insns = len(self.all_pcs_dict[name])
            covered_insns = len(self.covered_pcs_dict[name])
            coverage_insns += covered_insns / all_insns # the proportion of the coverage_insns

            all_blocks = len(self.contract_manager[name].cfg.blocks)
            covered_blocks = len(self.covered_blocks_dict[name])
            coverage_blocks += covered_blocks / all_blocks
        contract_jsons['insn_coverage'] = coverage_insns / len(self.contract_manager.fuzz_contract_names) # the average of the insn_coverage
        contract_jsons['block_coverage'] = coverage_blocks / len(self.contract_manager.fuzz_contract_names) # the average of the blocK_coverage

        for name in self.contract_manager.fuzz_contract_names:
            contract_json = OrderedDict()
            contract_json['tx_count'] = self.tx_count_dict[name]

            all_insns = len(self.all_pcs_dict[name])
            covered_insns = len(self.covered_pcs_dict[name])
            coverage_insns = covered_insns / all_insns
            contract_json['covered_insns'] = covered_insns
            contract_json['insn_coverage'] = coverage_insns
            contract_json['temp_insn_coverage'] = len(self.temp_covered_pcs_dict[name]) / all_insns
            contract_json['all_insns'] = all_insns

            all_blocks = len(self.contract_manager[name].cfg.blocks)
            covered_blocks = len(self.covered_blocks_dict[name])
            coverage_blocks = covered_blocks / all_blocks
            contract_json['covered_blocks'] = covered_blocks
            contract_json['block_coverage'] = coverage_blocks
            contract_json['temp_block_coverage'] = len(self.temp_covered_blocks_dict[name]) / all_blocks
            contract_json['all_blocks'] = all_blocks

            contract_json['bugs'] = OrderedDict()
            if len(self.bug_res[name]) > 0:
                for bug in sorted(self.bug_res[name]):
                    contract_json['bugs'][bug] = list(sorted(self.bug_res[name][bug]))

            contract_jsons[name] = contract_json

        return contract_jsons

    def export_result(self):
        res = self.to_json()
        for name in self.contract_manager.fuzz_contract_names:
            covered_insn_pc = list(self.covered_pcs_dict[name])
            covered_block = list(self.covered_blocks_dict[name])
            all_insns = list(self.all_pcs_dict[name])
            all_blocks = list(self.contract_manager[name].cfg.blocks)
            res[name]['covered_insn_pc'] = covered_insn_pc
            res[name]['covered_block'] = covered_block
            res[name]['all_insns'] = all_insns
            res[name]['all_blocks'] = all_blocks
        return res


    def __str__(self):
        return json.dumps(self.to_json())

    def get_insn_coverage_with_input(self, contract_name, executed_pcs):
        all_insns = len(self.all_pcs_dict[contract_name])
        covered_insns = len(executed_pcs)
        coverage_insns = covered_insns / all_insns 
        return coverage_insns

    def get_block_coverage_with_input(self, contract_name, executed_blocks):
        all_blocks = len(self.contract_manager[contract_name].cfg.blocks)
        covered_blocks = len(executed_blocks)
        coverage_blocks = covered_blocks / all_blocks
        return coverage_blocks

    def get_insn_coverage(self, contract_name):
        all_insns = len(self.all_pcs_dict[contract_name])
        covered_insns = len(self.covered_pcs_dict[contract_name])
        coverage_insns = covered_insns / all_insns 
        return coverage_insns


    def get_block_coverage(self, contract_name):
        all_blocks = len(self.contract_manager[contract_name].cfg.blocks)
        covered_blocks = len(self.covered_blocks_dict[contract_name])
        coverage_blocks = covered_blocks / all_blocks
        return coverage_blocks

    def get_coverage(self, contract_name):
        all_insns = len(self.all_pcs_dict[contract_name])
        covered_insns = len(self.temp_covered_pcs_dict[contract_name])
        coverage_insns = covered_insns / all_insns 

        all_blocks = len(self.contract_manager[contract_name].cfg.blocks)
        covered_blocks = len(self.temp_covered_blocks_dict[contract_name])
        coverage_blocks = covered_blocks / all_blocks
        return coverage_insns, coverage_blocks