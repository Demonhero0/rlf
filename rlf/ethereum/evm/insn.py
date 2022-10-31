from .opcode import OP_NAME


class Instruction:

    def __init__(self, *args, **kwargs):
        self.contract = kwargs['contract']
        self.pc = kwargs['pc'] # program count, the count in bytecode
        self.arg = kwargs['arg'] # only PUSH has args
        self.op = kwargs['op'] # the hex of opcode
        self.op_name = OP_NAME[self.op] # the name of opcode
        self.idx = None # the line of insn, one line is that (opcode,arg)
        self.states = set()


    def is_push(self):
        return 0x60 <= self.op <= 0x7f


    def __str__(self):
        s = '{} {} {}'.format(self.idx, format(self.pc, '02x'), self.op_name)
        if self.is_push():
            s += ' {}'.format(format(self.arg, '02x'))

        return s


    def add_state(self, new_state):
        self.states.add(new_state)
