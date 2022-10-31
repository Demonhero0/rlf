import random
import numpy as np

from ..policy_base import PolicyBase
from ...ethereum import SolType
from ...execution import Tx

# import seed
from ..seed.int_values import INT_VALUES_FREQUENT, INT_VALUES_UNFREQUENT, INT_VALUES_UNFREQUENT
from ..seed.amounts import AMOUNTS
from ..seed.addr_map import ADDR_MAP

INT_EXPLORE_RATE = -1

class PolicyFRandom(PolicyBase):

    def __init__(self, execution, contract_manager, account_manager):
        super().__init__(execution, contract_manager, account_manager)

        self.addr_map = ADDR_MAP
        self.int_values_frequent = INT_VALUES_FREQUENT
        self.int_values_unfrequent = INT_VALUES_UNFREQUENT
        self.amounts = AMOUNTS
        self.slice_size = 2


    def select_tx_for_method(self, contract, method, obs):
        self.slice_size = random.randint(1, 5)
        address = contract.addresses[0]
        sender = self._select_sender()
        arguments = self._select_arguments(contract, method, sender, obs)
        amount = self._select_amount(contract, method, sender, obs)
        timestamp = self._select_timestamp(obs)

        tx = Tx(self, contract.name, address, method.name, bytes(), arguments, amount, sender, timestamp, True)
        return tx

    def select_tx(self, obs):
        r = random.random()
        if r >= 0.2:
            self.slice_size = random.randint(1, 5)
        else:
            self.slice_size = None
        contract = self._select_contract()
        address = contract.addresses[0]
        method = self._select_method(contract)
        sender = self._select_sender()
        arguments = self._select_arguments(contract, method, sender, obs)
        amount = self._select_amount(contract, method, sender, obs)
        timestamp = self._select_timestamp(obs)

        tx = Tx(self, contract.name, address, method.name, bytes(), arguments, amount, sender, timestamp, True)
        return tx


    def _select_contract(self):
        contract_name = random.choice(self.contract_manager.fuzz_contract_names)
        return self.contract_manager[contract_name]

    def _select_sender(self):
        return random.choice(range(0, len(self.account_manager.accounts)))

    def _select_method(self, contract):
        return random.choice(contract.abi.methods)

    def _select_amount(self, contract, method, sender, obs, pred_amount=None):
        if sender in self.account_manager.attacker_indices:
            return 0

        if self.contract_manager.is_payable(contract.name, method.name):
            if pred_amount is None:
                amount = random.randint(0, self.account_manager[sender].amount)
            else:
                amount = self.amounts[pred_amount]
            return amount
        else:
            return 0


    def _select_arguments(self, contract, method, sender, obs):

        arguments = []
        for arg in method.inputs:
            t = arg.evm_type.t
            if t == SolType.IntTy or t == SolType.UintTy:
                chosen_int = None
                if t == SolType.IntTy:
                    arguments.append(self._select_int(contract, method, arg.evm_type.size, obs, chosen_int))
                elif t == SolType.UintTy:
                    arguments.append(self._select_uint(contract, method, arg.evm_type.size, obs, chosen_int))
            elif t == SolType.BoolTy:
                arguments.append(self._select_bool())
            elif t == SolType.StringTy:
                arguments.append(self._select_string(obs))
            elif t == SolType.SliceTy:
                arg = self._select_slice(contract, method, sender, arg.evm_type.elem, obs)
                arguments.append(arg)
            elif t == SolType.ArrayTy:
                arg = self._select_array(contract, method, sender, arg.evm_type.size, arg.evm_type.elem, obs)
                arguments.append(arg)
            elif t == SolType.AddressTy:
                # TODO select address
                chosen_addr = np.random.choice(len(self.addresses))
                arguments.append(self._select_address(sender, chosen_addr))
            elif t == SolType.FixedBytesTy:
                arguments.append(self._select_fixed_bytes(arg.evm_type.size, obs))
            elif t == SolType.BytesTy:
                arguments.append(self._select_bytes(obs))
            else:
                assert False, 'type {} not supported'.format(t)
        return arguments

    # def _select_int(self, contract, method, size, obs, chosen_int=None):
    #     if chosen_int is not None and chosen_int != len(self.int_values):
    #         value = self.int_values[chosen_int]
    #         value &= ((1 << size) - 1)
    #         if value & (1 << (size - 1)):
    #             value -= (1 << size)
    #         return value

    #     p = 1 << (size - 1)
    #     return random.randint(-p, p-1)

    # def _select_uint(self, contract, method, size, obs, chosen_int=None):
    #     if chosen_int is not None and chosen_int != len(self.int_values):
    #         value = self.int_values[chosen_int]
    #         value &= ((1 << size) - 1)
    #         return value

    #     p = 1 << size
    #     return random.randint(0, p-1)

    def _select_int(self, contract, method, size, obs, chosen_int=None):
        # if chosen_int is not None and chosen_int != len(self.int_values):
        #     value = self.int_values[chosen_int]
        #     value &= ((1 << size) - 1)
        #     if value & (1 << (size - 1)):
        #         value -= (1 << size)
        #     return value
        s = random.random()
        if s < 0.9:
            value = random.choice(self.int_values_frequent)
        elif s < 0.98:
            value = random.choice(self.int_values_unfrequent)
        else:
            p = 1 << (size - 1)
            return random.randint(-p, p-1)

        value &= ((1 << size) - 1)
        if value & (1 << (size - 1)):
            value -= (1 << size)
        return value

    def _select_uint(self, contract, method, size, obs, chosen_int=None):
        # if chosen_int is not None and chosen_int != len(self.int_values):
        #     value = self.int_values[chosen_int]
        #     value &= ((1 << size) - 1)
        #     return value

        # p = 1 << size
        # return random.randint(0, p-1)
        s = random.random()
        if s < 0.9:
            value = random.choice(self.int_values_frequent)
        elif s < 0.98:
            value = random.choice(self.int_values_unfrequent)
        else:
            p = 1 << size
            return random.randint(0, p-1)
        return value

    def _select_address(self, sender, idx=None):
        if sender in self.account_manager.attacker_indices:
            if idx is None:
                return random.choice(self.addresses)
            else:
                return self.addresses[idx]
        else:
            if idx is None or self.addresses[idx] in self.account_manager.attacker_addresses:
                l = [addr for addr in self.addresses if addr not in self.account_manager.attacker_addresses]
                return random.choice(l)
            else:
                return self.addresses[idx]

    def _select_bool(self):
        return random.choice([True, False])

    def _select_string(self, obs):
        bs = []
        size = random.randint(0, 40)
        for _ in range(size):
            bs.append(random.randint(1, 127))
        return bytearray(bs).decode('ascii')

    def _select_slice(self, contract, method, sender, typ, obs):
        if self.slice_size is None:
            size = random.randint(1, 15)
        else:
            size = self.slice_size
        return self._select_array(contract, method, sender, size, typ, obs)

    def _select_array(self, contract, method, sender, size, typ, obs):
        t = typ.t
        arr = []

        for _ in range(size):
            if t in (SolType.IntTy, SolType.UintTy):
                # s = random.random()
                # if s >= INT_EXPLORE_RATE:
                #     # TODO select int
                #     chosen_int = np.random.choice(len(self.int_values)+1)
                # else:
                #     chosen_int = None
                chosen_int = None
                if t == SolType.IntTy:
                    arr.append(self._select_int(contract, method, typ.size, obs, chosen_int))
                elif t == SolType.UintTy:
                    arr.append(self._select_uint(contract, method, typ.size, obs, chosen_int))
            elif t == SolType.BoolTy:
                arr.append(self._select_bool())
            elif t == SolType.StringTy:
                arr.append(self._select_string(obs))
            elif t == SolType.SliceTy:
                arg = self._select_slice(contract, method, sender, typ.elem, obs)
                arr.append(arg)
            elif t == SolType.ArrayTy:
                arg = self._select_array(contract, method, sender, typ.size, typ.elem, obs)
                arr.append(arg)
            elif t == SolType.AddressTy:
                # TODO select address
                chosen_addr = np.random.choice(len(self.addresses))
                arr.append(self._select_address(sender, chosen_addr))
            elif t == SolType.FixedBytesTy:
                arr.append(self._select_fixed_bytes(typ.size, obs))
            elif t == SolType.BytesTy:
                arr.append(self._select_bytes(obs))
            else:
                assert False, 'type {} not supported'.format(t)

        return arr

    def _select_fixed_bytes(self, size, obs):
        bs = []
        for _ in range(size):
            bs.append(random.randint(0, 255))
        return bs

    def _select_bytes(self, obs):
        size = random.randint(1, 15)
        return self._select_fixed_bytes(size, obs)