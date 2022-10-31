import json

from collections import OrderedDict
from pickle import TRUE
from re import M
from ..ethereum import Method
from ..ethereum.evm.opcode import INVALID, RETURN, REVERT, CALL
from ..ethereum import SolType


class RecordManager:

    def __init__(self, obs, contract_manager, account_manager):
        self.obs = obs
        self.contract_manager = contract_manager
        self.account_manager = account_manager
        self.addresses = []
        for contract in contract_manager.contract_dict.values():
            self.addresses += contract.addresses

        for account in account_manager.accounts:
            self.addresses.append(account.address)

        self.tx_count = 0

        self.method_records = dict()
        for contract in self.contract_manager.contract_dict.values():
            self.method_records[contract.name] = dict()
            for method in contract.abi.methods:
                self.method_records[contract.name][method.name] = MethodRecord(
                    method.name,
                    method.num_addrs_in_args(),
                    method.len_args(),
                    contract.is_payable(method.name),
                    method.name == Method.FALLBACK,
                    method.insns,
                    method.blocks,
                    0 if len(contract.insns) == 0 else (len(method.insns) / len(contract.insns)),
                    0 if len(contract.cfg.blocks) == 0 else (len(method.blocks) / len(contract.cfg.blocks)),
                )


    def get_method_features(self, contract):
        """
        get the vec features of methods
        """
        res = OrderedDict()
        for method_name, method_record in self.method_records[contract].items():
            res[method_name] = method_record.to_vec(self.tx_count)
        return res

    def get_method_coverage(self, contract):
        res = dict()
        for method_name, method_record in self.method_records[contract].items():
            res[method_name] = dict()
            res[method_name]['insn_cov'] = len(method_record.covered_insns) / method_record.len_insns if method_record.len_insns > 0 else 0
            res[method_name]['block_cov'] = len(method_record.covered_blocks) / method_record.len_blocks if method_record.len_blocks > 0 else 0
        return res

    def to_json(self, chosen_contract, chosen_method):
        """
        return the json of the chosen_method in chosen_contract
        """
        method_record = self.method_records[chosen_contract][chosen_method]

        res_json = OrderedDict()
        res_json['methods'] = OrderedDict()
        for method_name, method_record in self.method_records[chosen_contract].items():
            res_json['methods'][method_name] = method_record.to_vec(self.tx_count)

        return res_json


    def update(self, logger, insn_coverage_change, block_coverage_change):
        self.tx_count += 1
        tx = logger.tx

        method_record = self.method_records[tx.contract][tx.method]
        method_record.tx_count += 1
        method_record.insn_coverage += insn_coverage_change
        method_record.block_coverage += block_coverage_change
        method_record.last_revert = False
        method_record.last_invalid = False
        method_record.last_return = False

        call_stack = []
        call_stack.append(self.contract_manager[logger.tx.contract].addresses[0])
        for i, log in enumerate(logger.logs):
            # ??? why
            if i > 0:
                prev_log = logger.logs[i-1]
                if log.depth > prev_log.depth:
                    call_stack.append(prev_log.stack[-2])
                elif log.depth < prev_log.depth:
                    call_stack.pop()

            if call_stack[-1] not in self.contract_manager.address_to_contract:
                continue

            cur_contract = self.contract_manager.address_to_contract[call_stack[-1]]

            if cur_contract.name == tx.contract and log.pc in cur_contract.insn_pc_to_idx:
                insn_idx = cur_contract.insn_pc_to_idx[log.pc]
                if insn_idx in method_record.all_insns:
                    method_record.covered_insns.add(insn_idx)
                if insn_idx in cur_contract.cfg.insn_idx_to_block:
                    block = cur_contract.cfg.insn_idx_to_block[insn_idx]
                    if block.start_idx in method_record.all_blocks:
                        method_record.covered_blocks.add(block.start_idx)

                if i == len(logger.logs) - 1:
                    if log.op == REVERT:
                        method_record.last_revert = True
                        method_record.count_revert += 1
                    elif log.op == INVALID:
                        method_record.last_invalid = True
                        method_record.count_invalid += 1
                    elif log.op == RETURN:
                        method_record.last_return = True
                        method_record.count_return += 1


class MethodRecord:
    """
    the record of method
    """
    def __init__(self, name, num_address_in_args, len_args, payable, fallback, all_insns, all_blocks, frac_insns, frac_blocks):
        self.name = name

        self.num_address_in_args = num_address_in_args
        self.len_args = len_args
        self.payable = payable
        self.fallback = fallback
        self.all_insns = all_insns
        self.all_blocks = all_blocks
        self.len_insns = len(self.all_insns)
        self.len_blocks = len(self.all_blocks)
        self.frac_insns = frac_insns 
        self.frac_blocks = frac_blocks

        self.covered_insns = set()
        self.covered_blocks = set()
        self.tx_count = 0

        self.last_revert = False # true if t_last ends with revert
        self.last_invalid = False # true if t_last ends with invalid
        self.last_return = False # true if t_last ends with return
        self.count_revert = 0 # the num of transactions end with revert
        self.count_invalid = 0 # the num of transactions end with invalid
        self.count_return = 0 # the num of transactions end with return

        self.insn_coverage = 0
        self.block_coverage = 0


    def to_vec(self, tx_count):
        feature = []

        feature.append(self.tx_count / tx_count if tx_count else 0) # Fraction of transactions in txs that calls self method
        feature.append(len(self.covered_insns) / self.len_insns if self.len_insns > 0 else 0) # Instruction coverage of self method
        feature.append(self.num_address_in_args) # Number of arguments of type address
        feature.append(self.len_args) # Number of arguments of self method
        feature.append(float(self.payable)) 
        feature.append(float(self.fallback))
        feature.append(len(self.covered_blocks) / self.len_blocks if self.len_blocks > 0 else 0) # Block coverage of self method
        feature.append(int(self.last_revert))  # 1 if t_last ends with revert
        feature.append(int(self.last_invalid)) # 1 if t_last ends with invalid
        feature.append(int(self.last_return)) # 1 if t_last ends with return
        feature.append((self.count_revert / self.tx_count) if self.tx_count else 0) # Fraction of transactions in txs that end with revert
        feature.append((self.count_invalid / self.tx_count) if self.tx_count else 0) # Fraction of transactions in txs that end with invalid
        feature.append((self.count_return / self.tx_count) if self.tx_count else 0) # Fraction of transactions in txs that end with return
        feature.append(self.insn_coverage) # the num of insn coverage of contract
        feature.append(self.block_coverage) # the num of block coverage of contract
        feature.append(self.frac_insns) # Fraction of self method insn in contract
        feature.append(self.frac_blocks) # Fraction of self method block in contract

        return feature