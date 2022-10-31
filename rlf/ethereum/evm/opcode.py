STOP = 0x0
ADD = 0x1
MUL = 0x2
SUB = 0x3
DIV = 0x4
SDIV = 0x5
MOD = 0x6
SMOD = 0x7
ADDMOD = 0x8
MULMOD = 0x9
EXP = 0xa
SIGNEXTEND = 0xb

LT = 0x10
GT = 0x11
SLT = 0x12
SGT = 0x13
EQ = 0x14
ISZERO = 0x15
AND = 0x16
OR = 0x17
XOR = 0x18
NOT = 0x19
BYTE = 0x1a
SHL = 0x1b
SHR = 0x1c
SAR = 0x1d

SHA3 = 0x20

ADDRESS = 0x30
BALANCE = 0x31
ORIGIN = 0x32
CALLER = 0x33
CALLVALUE = 0x34
CALLDATALOAD = 0x35
CALLDATASIZE = 0x36
CALLDATACOPY = 0x37
CODESIZE = 0x38
CODECOPY = 0x39
GASPRICE = 0x3a
EXTCODESIZE = 0x3b
EXTCODECOPY = 0x3c
RETURNDATASIZE = 0x3d
RETURNDATACOPY = 0x3e

BLOCKHASH = 0x40
COINBASE = 0x41 # the beneficiary address of block
TIMESTAMP = 0x42
NUMBER = 0x43
DIFFICULTY = 0x44
GASLIMIT = 0x45

POP = 0x50
MLOAD = 0x51
MSTORE = 0x52
MSTORE8 = 0x53
SLOAD = 0x54
SSTORE = 0x55
JUMP = 0x56
JUMPI = 0x57
PC = 0x58
MSIZE = 0x59
GAS = 0x5a
JUMPDEST = 0x5b

PUSH1 = 0x60
PUSH2 = 0x61
PUSH3 = 0x62
PUSH4 = 0x63
PUSH5 = 0x64
PUSH6 = 0x65
PUSH7 = 0x66
PUSH8 = 0x67
PUSH9 = 0x68
PUSH10 = 0x69
PUSH11 = 0x6a
PUSH12 = 0x6b
PUSH13 = 0x6c
PUSH14 = 0x6d
PUSH15 = 0x6e
PUSH16 = 0x6f
PUSH17 = 0x70
PUSH18 = 0x71
PUSH19 = 0x72
PUSH20 = 0x73
PUSH21 = 0x74
PUSH22 = 0x75
PUSH23 = 0x76
PUSH24 = 0x77
PUSH25 = 0x78
PUSH26 = 0x79
PUSH27 = 0x7a
PUSH28 = 0x7b
PUSH29 = 0x7c
PUSH30 = 0x7d
PUSH31 = 0x7e
PUSH32 = 0x7f

DUP1 = 0x80
DUP2 = 0x81
DUP3 = 0x82
DUP4 = 0x83
DUP5 = 0x84
DUP6 = 0x85
DUP7 = 0x86
DUP8 = 0x87
DUP9 = 0x88
DUP10 = 0x89
DUP11 = 0x8a
DUP12 = 0x8b
DUP13 = 0x8c
DUP14 = 0x8d
DUP15 = 0x8e
DUP16 = 0x8f

SWAP1 = 0x90
SWAP2 = 0x91
SWAP3 = 0x92
SWAP4 = 0x93
SWAP5 = 0x94
SWAP6 = 0x95
SWAP7 = 0x96
SWAP8 = 0x97
SWAP9 = 0x98
SWAP10 = 0x99
SWAP11 = 0x9a
SWAP12 = 0x9b
SWAP13 = 0x9c
SWAP14 = 0x9d
SWAP15 = 0x9e
SWAP16 = 0x9f

LOG0 = 0xa0
LOG1 = 0xa1
LOG2 = 0xa2
LOG3 = 0xa3
LOG4 = 0xa4

PUSH = 0xb0
DUP = 0xb1
SWAP = 0xb2

CREATE = 0xf0
CALL = 0xf1
CALLCODE = 0xf2
RETURN = 0xf3
DELEGATECALL = 0xf4
STATICCALL = 0xfa
REVERT = 0xfd
INVALID = 0xfe
SELFDESTRUCT = 0xff


def is_push(op):
    return 0x60 <= op <= 0x7f


def is_dup(op):
    return 0x80 <= op <= 0x8f


def is_swap(op):
    return 0x90 <= op <= 0x9f


STACK_CHANGES = {
    STOP: 0,
    ADD: -1,
    MUL: -1,
    SUB: -1,
    DIV: -1,
    SDIV: -1,
    MOD: -1,
    SMOD: -1,
    ADDMOD: -2,
    MULMOD: -2,
    EXP: -1,
    SIGNEXTEND: -1,

    LT: -1,
    GT: -1,
    SLT: -1,
    SGT: -1,
    EQ: -1,
    ISZERO: 0,
    AND: -1,
    OR: -1,
    XOR: -1,
    NOT: 0,
    BYTE: -1,
    SHL: -1,
    SHR: -1,
    SAR: -1,

    SHA3: -1,

    ADDRESS: 1,
    BALANCE: 0,
    ORIGIN: 1,
    CALLER: 1,
    CALLVALUE: 1,
    CALLDATALOAD: 0,
    CALLDATASIZE: 1,
    CALLDATACOPY: -3,
    CODESIZE: 1,
    CODECOPY: -3,
    GASPRICE: 1,
    EXTCODESIZE: 0,
    EXTCODECOPY: -4,
    RETURNDATASIZE: 1,
    RETURNDATACOPY: -3,

    BLOCKHASH: 0,
    COINBASE: 1,
    TIMESTAMP: 1,
    NUMBER: 1,
    DIFFICULTY: 1,
    GASLIMIT: 1,

    POP: -1,
    MLOAD: 0,
    MSTORE: -2,
    MSTORE8: -2,
    SLOAD: 0,
    SSTORE: -2,
    JUMP: -1,
    JUMPI: -2,
    PC: 1,
    MSIZE: 1,
    GAS: 1,
    JUMPDEST: 0,

    PUSH1: 1,
    PUSH2: 1,
    PUSH3: 1,
    PUSH4: 1,
    PUSH5: 1,
    PUSH6: 1,
    PUSH7: 1,
    PUSH8: 1,
    PUSH9: 1,
    PUSH10: 1,
    PUSH11: 1,
    PUSH12: 1,
    PUSH13: 1,
    PUSH14: 1,
    PUSH15: 1,
    PUSH16: 1,
    PUSH17: 1,
    PUSH18: 1,
    PUSH19: 1,
    PUSH20: 1,
    PUSH21: 1,
    PUSH22: 1,
    PUSH23: 1,
    PUSH24: 1,
    PUSH25: 1,
    PUSH26: 1,
    PUSH27: 1,
    PUSH28: 1,
    PUSH29: 1,
    PUSH30: 1,
    PUSH31: 1,
    PUSH32: 1,

    DUP1: 1,
    DUP2: 1,
    DUP3: 1,
    DUP4: 1,
    DUP5: 1,
    DUP6: 1,
    DUP7: 1,
    DUP8: 1,
    DUP9: 1,
    DUP10: 1,
    DUP11: 1,
    DUP12: 1,
    DUP13: 1,
    DUP14: 1,
    DUP15: 1,
    DUP16: 1,

    SWAP1: 0,
    SWAP2: 0,
    SWAP3: 0,
    SWAP4: 0,
    SWAP5: 0,
    SWAP6: 0,
    SWAP7: 0,
    SWAP8: 0,
    SWAP9: 0,
    SWAP10: 0,
    SWAP11: 0,
    SWAP12: 0,
    SWAP13: 0,
    SWAP14: 0,
    SWAP15: 0,
    SWAP16: 0,

    LOG0: -2,
    LOG1: -3,
    LOG2: -4,
    LOG3: -5,
    LOG4: -6,

    CREATE: -2,
    CALL: -6,
    CALLCODE: -6,
    RETURN: -2,
    DELEGATECALL: -5,
    STATICCALL: -5,
    REVERT: -2,
    INVALID: 0,
    SELFDESTRUCT: -1
}


OP_NAME = {
    STOP: 'STOP',
    ADD: 'ADD',
    MUL: 'MUL',
    SUB: 'SUB',
    DIV: 'DIV',
    SDIV: 'SDIV',
    MOD: 'MOD',
    SMOD: 'SMOD',
    ADDMOD: 'ADDMOD',
    MULMOD: 'MULMOD',
    EXP: 'EXP',
    SIGNEXTEND: 'SIGNEXTEND',

    LT: 'LT',
    GT: 'GT',
    SLT: 'SLT',
    SGT: 'SGT',
    EQ: 'EQ',
    ISZERO: 'ISZERO',
    AND: 'AND',
    OR: 'OR',
    XOR: 'XOR',
    NOT: 'NOT',
    BYTE: 'BYTE',
    SHL: 'SHL',
    SHR: 'SHR',
    SAR: 'SAR',

    SHA3: 'SHA3',

    ADDRESS: 'ADDRESS',
    BALANCE: 'BALANCE',
    ORIGIN: 'ORIGIN',
    CALLER: 'CALLER',
    CALLVALUE: 'CALLVALUE',
    CALLDATALOAD: 'CALLDATALOAD',
    CALLDATASIZE: 'CALLDATASIZE',
    CALLDATACOPY: 'CALLDATACOPY',
    CODESIZE: 'CODESIZE',
    CODECOPY: 'CODECOPY',
    GASPRICE: 'GASPRICE',
    EXTCODESIZE: 'EXTCODESIZE',
    EXTCODECOPY: 'EXTCODECOPY',
    RETURNDATASIZE: 'RETURNDATASIZE',
    RETURNDATACOPY: 'RETURNDATACOPY',

    BLOCKHASH: 'BLOCKHASH',
    COINBASE: 'COINBASE',
    TIMESTAMP: 'TIMESTAMP',
    NUMBER: 'NUMBER',
    DIFFICULTY: 'DIFFICULTY',
    GASLIMIT: 'GASLIMIT',

    POP: 'POP',
    MLOAD: 'MLOAD',
    MSTORE: 'MSTORE',
    MSTORE8: 'MSTORE8',
    SLOAD: 'SLOAD',
    SSTORE: 'SSTORE',
    JUMP: 'JUMP',
    JUMPI: 'JUMPI',
    PC: 'PC',
    MSIZE: 'MSIZE',
    GAS: 'GAS',
    JUMPDEST: 'JUMPDEST',

    PUSH1: 'PUSH1',
    PUSH2: 'PUSH2',
    PUSH3: 'PUSH3',
    PUSH4: 'PUSH4',
    PUSH5: 'PUSH5',
    PUSH6: 'PUSH6',
    PUSH7: 'PUSH7',
    PUSH8: 'PUSH8',
    PUSH9: 'PUSH9',
    PUSH10: 'PUSH10',
    PUSH11: 'PUSH11',
    PUSH12: 'PUSH12',
    PUSH13: 'PUSH13',
    PUSH14: 'PUSH14',
    PUSH15: 'PUSH15',
    PUSH16: 'PUSH16',
    PUSH17: 'PUSH17',
    PUSH18: 'PUSH18',
    PUSH19: 'PUSH19',
    PUSH20: 'PUSH20',
    PUSH21: 'PUSH21',
    PUSH22: 'PUSH22',
    PUSH23: 'PUSH23',
    PUSH24: 'PUSH24',
    PUSH25: 'PUSH25',
    PUSH26: 'PUSH26',
    PUSH27: 'PUSH27',
    PUSH28: 'PUSH28',
    PUSH29: 'PUSH29',
    PUSH30: 'PUSH30',
    PUSH31: 'PUSH31',
    PUSH32: 'PUSH32',

    DUP1: 'DUP1',
    DUP2: 'DUP2',
    DUP3: 'DUP3',
    DUP4: 'DUP4',
    DUP5: 'DUP5',
    DUP6: 'DUP6',
    DUP7: 'DUP7',
    DUP8: 'DUP8',
    DUP9: 'DUP9',
    DUP10: 'DUP10',
    DUP11: 'DUP11',
    DUP12: 'DUP12',
    DUP13: 'DUP13',
    DUP14: 'DUP14',
    DUP15: 'DUP15',
    DUP16: 'DUP16',

    SWAP1: 'SWAP1',
    SWAP2: 'SWAP2',
    SWAP3: 'SWAP3',
    SWAP4: 'SWAP4',
    SWAP5: 'SWAP5',
    SWAP6: 'SWAP6',
    SWAP7: 'SWAP7',
    SWAP8: 'SWAP8',
    SWAP9: 'SWAP9',
    SWAP10: 'SWP10',
    SWAP11: 'SWP11',
    SWAP12: 'SWP12',
    SWAP13: 'SWP13',
    SWAP14: 'SWP14',
    SWAP15: 'SWP15',
    SWAP16: 'SWP16',

    LOG0: 'LOG0',
    LOG1: 'LOG1',
    LOG2: 'LOG2',
    LOG3: 'LOG3',
    LOG4: 'LOG4',

    CREATE: 'CREATE',
    CALL: 'CALL',
    CALLCODE: 'CALLCODE',
    RETURN: 'RETURN',
    DELEGATECALL: 'DELEGATECALL',
    STATICCALL: 'STATICCALL',
    REVERT: 'REVERT',
    INVALID: 'INVALID',
    SELFDESTRUCT: 'SELFDESTRUCT',
}

# may be the essential opcode?
INTERESTING_OPS = [
    LT,
    GT,
    SLT,
    SGT,
    ISZERO,
    AND,
    OR,
    XOR,
    NOT,

    SHA3,

	ADDRESS,
	BALANCE,
	ORIGIN,
	CALLER,
	CALLVALUE,
	CALLDATALOAD,
	CALLDATASIZE,
	CALLDATACOPY,
	CODESIZE,
	CODECOPY,
	GASPRICE,
	EXTCODESIZE,
	EXTCODECOPY,
	RETURNDATASIZE,
	RETURNDATACOPY,
	BLOCKHASH,
	COINBASE,
	TIMESTAMP,
	NUMBER,
	DIFFICULTY,
	GASLIMIT,

	SLOAD,
	SSTORE,

	PC,
	MSIZE,
	GAS,

	LOG0,
	LOG1,
	LOG2,
	LOG3,
	LOG4,

	CREATE,
	CALL,
	CALLCODE,
	RETURN,
	DELEGATECALL,
	STATICCALL,
	REVERT,
    INVALID,
	SELFDESTRUCT,
]

MEM_WRITES = {
    MSTORE,
    MSTORE8,
    CALLDATACOPY,
    CODECOPY,
    RETURNDATACOPY,
    EXTCODECOPY,
}

MEM_READS = {
    SHA3,
    LOG0,
    LOG1,
    LOG2,
    LOG3,
    LOG4,
    MLOAD,
    CREATE,
    RETURN,
    REVERT,
}

MEM_READ_WRITES = {
    CALL,
    CALLCODE,
    DELEGATECALL,
    STATICCALL,
}

# select the 50 most representative opcodes
def select_interesting_ops(bow: list):
    res = []
    for op in INTERESTING_OPS:
        res.append(bow[op])
    return res

import numpy as np
remove_list = [PC, LOG0, LOG1, LOG2, LOG3, LOG4]
def select_opcode_for_RL(bow: list) -> np.array: 
    op_count = dict()
    for i, op in enumerate(INTERESTING_OPS):
        if op in remove_list:
            continue
        op_count[OP_NAME[op]] = bow[i]
    temp_bow = np.array(list(op_count.values()))
    sum_bow = temp_bow.sum()
    if sum_bow > 0:
        temp_bow = temp_bow/sum_bow
    return temp_bow

def select_opcode_for_cloud(bow) -> np.array: 
    op_count = dict()
    for i, op in enumerate(INTERESTING_OPS):
        if op in remove_list:
            continue
        op_count[OP_NAME[op]] = bow[i]
    temp_bow = np.array(list(op_count.values()))
    sum_bow = temp_bow.sum()
    if sum_bow > 0:
        temp_bow = temp_bow/sum_bow
        for key in op_count:
            op_count[key] = op_count[key]/sum_bow
    return op_count

def count_function_feature_for_kmeans(method) -> np.array:
    row_bow = method['row_bow']
    op_count = np.array(row_bow)
    temp_bow = np.zeros(14)
    # Caculation
    temp_bow[0] = op_count[ADD:EXP+1].sum() * 0.5
    # Logic
    temp_bow[1] = op_count[LT:NOT+1].sum() * 0.5
    # about MSG
    temp_bow[2] = op_count[ADDRESS:RETURNDATACOPY+1].sum()
    # about block
    temp_bow[3] = op_count[BLOCKHASH:GASLIMIT+1].sum()
    # about Memory
    temp_bow[4] = op_count[MLOAD:MSTORE8+1].sum() * 0.5
    # about storage
    temp_bow[5] = op_count[SLOAD:SSTORE+1].sum()
    # JUMP
    temp_bow[6] = op_count[JUMP:JUMPI+1].sum()
    # about LOG
    # temp_bow[7] = op_count[LOG0:LOG4+1].sum()
    # single opcode
    single_opcode = [SHA3, CALL, DELEGATECALL, STATICCALL, CREATE, SELFDESTRUCT]
    for i, opcode in enumerate(single_opcode):
        temp_bow[7+i] = op_count[opcode] * 10

    temp_bow[-1] = 10 if method['payable'] else 0

    sum_bow = temp_bow.sum()
    if sum_bow > 0:
        temp_bow = temp_bow/sum_bow
    return temp_bow

def select_trace_opcode(row_bow: list) -> np.array:
    op_count = np.array(row_bow)
    # single opcode
    single_opcode = [SLOAD, SSTORE, SHA3, CALL, DELEGATECALL, SELFDESTRUCT, JUMP, JUMPI]
    temp_bow = np.zeros(len(single_opcode))
    for i, opcode in enumerate(single_opcode):
        temp_bow[i] = op_count[opcode]

    return temp_bow