import abc
import json
import numpy as np
from .stat import Stat
from .record import RecordManager
from collections import OrderedDict
from ..ethereum import select_interesting_ops

class ObsBase:

    def __init__(self, contract_manager, account_manager, dataset_dump_path):
        self.contract_manager = contract_manager
        self.account_manager =  account_manager
        self.dataset_dump_path = dataset_dump_path

        self.all_trace_bow = self.get_all_trace_bow(None)
        self.trace_bow = self.get_trace_bow(None) # the frequency of 50 most representative opcodes in the logger(this tx)
        self.stat = Stat(contract_manager, account_manager)
        self.record_manager = RecordManager(self, contract_manager, account_manager)

    def init(self):
        if self.dataset_dump_path is None:
            return

        j = OrderedDict()
        j['type'] = 'init'
        j['contracts'] = OrderedDict()
        for name in self.contract_manager.fuzz_contract_names:
            j['contracts'][name] = self.contract_manager[name].to_json()
            with open(self.dataset_dump_path, 'w') as w:
                w.write(json.dumps(j))
                w.write('\n')


    def update(self, logger, is_init_txs):
        tx = logger.tx

        if self.dataset_dump_path and not is_init_txs:
            j = OrderedDict()
            j['type'] = 'tx'
            j['tx'] = tx.to_json()
            j['trace_bow'] = self.trace_bow
            j['features'] = self.record_manager.to_json(tx.contract, tx.method)

        old_insn_coverage = self.stat.get_insn_coverage(tx.contract)
        old_block_coverage = self.stat.get_block_coverage(tx.contract)
        # update stat
        destruct, executed_pcs, executed_blocks = self.stat.update(logger)
        new_insn_coverage = self.stat.get_insn_coverage(tx.contract)
        new_block_coverage = self.stat.get_block_coverage(tx.contract)
        # reward
        insn_coverage_change = new_insn_coverage - old_insn_coverage
        block_coverage_change = new_block_coverage - old_block_coverage

        if self.dataset_dump_path and not is_init_txs:
            j['insn_coverage_change'] = insn_coverage_change
            j['block_coverage_change'] = block_coverage_change
            with open(self.dataset_dump_path, 'a') as w:
                w.write(json.dumps(j))
                w.write('\n')

        self.trace_bow = self.get_trace_bow(logger)
        self.all_trace_bow = self.get_all_trace_bow(logger)
        # update method record
        if tx.method in self.contract_manager[tx.contract].abi.methods_by_name:
            self.record_manager.update(logger, insn_coverage_change, block_coverage_change)

        return destruct, self.stat.get_insn_coverage_with_input(tx.contract, executed_pcs), self.stat.get_block_coverage_with_input(tx.contract, executed_blocks)

    def get_trace_bow(self, logger) -> list:
        """
        count the interesting_ops
        """
        bow = [0 for _ in range(256)]

        if logger is not None:
            for log in logger.logs:
                bow[log.op] += 1

        return select_interesting_ops(bow)

    def get_all_trace_bow(self, logger) -> list:
        """
        count the interesting_ops
        """
        bow = [0 for _ in range(256)]

        if logger is not None:
            for log in logger.logs:
                bow[log.op] += 1

        return bow