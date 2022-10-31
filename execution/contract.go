package execution

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/asm"
	"github.com/ethereum/go-ethereum/core/vm"
)

var bzzr0 = fmt.Sprintf("%x", append([]byte{0xa1, 0x65}, []byte("bzzr0")...))

type Contract struct {
	Name      string           `json:"name"`
	Addresses []common.Address `json:"addresses"`
	ABI       *abi.ABI         `json:"abi"`
	Payable   map[string]bool  `json:"payable"`
	Insns     []*Instruction   `json:"insns"`
}

func (contract *Contract) ContainAddress(address common.Address) bool {
	for _, contractAddress := range contract.Addresses {
		if contractAddress == address {
			return true
		}
	}
	return false
}

type ContractManager struct {
	DeployedContracts map[string]*Contract `json:"contracts"`
	ProjPath          string               `json:"proj_path"`
}

func (manager *ContractManager) GetContractByName(name string) *Contract {
	return manager.DeployedContracts[name]
}

// such as PUSH 0x30, Op=PUSH, Arg=0x30
type Instruction struct {
	PC  uint64    `json:"pc"`
	Arg *big.Int  `json:"arg"`
	Op  vm.OpCode `json:"op"`
}

func NewContract(
	name string,
	address common.Address,
	ABI *abi.ABI,
	payable map[string]bool,
	bytecode string,
) *Contract {
	idx := strings.Index(bytecode, bzzr0)
	bytecode = bytecode[:idx-1]
	script, _ := hex.DecodeString(replacePlaceHolders(bytecode))
	it := asm.NewInstructionIterator(script) // like the class in C++, it has var: pc,arg,op, it can change by using it.Next(), than transfer them to insn
	// change the code to instractuion list, read bytecode one by one
	var insns []*Instruction
	for it.Next() {
		arg := new(big.Int)
		insn := &Instruction{
			PC:  it.PC(),
			Arg: arg.SetBytes(it.Arg()),
			Op:  it.Op(),
		}
		insns = append(insns, insn)
	}

	contract := &Contract{
		Name:    name,
		ABI:     ABI,
		Payable: payable,
		Insns:   insns,
	}
	contract.Addresses = append(contract.Addresses, address) // why a contract has more than one address?

	return contract
}

func GetSwarmHash(bytecode string) (string, bool) {
	idx := strings.Index(bytecode, bzzr0)
	if idx == -1 {
		return "", false
	}
	swarmHash := bytecode[idx+len(bzzr0) : idx+len(bzzr0)+64]
	return swarmHash, true
}

func replacePlaceHolders(bytecode string) string {
	// deal with the libraries address
	var buffer bytes.Buffer
	for i := 0; i < len(bytecode); i++ {
		if bytecode[i] == '_' {
			i += 39
			buffer.WriteString("0000000000000000000000000000000000000000")
			continue
		}
		buffer.WriteString(string(bytecode[i]))
	}
	return buffer.String()
}
