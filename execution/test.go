package main

import (
	"fmt"
	// "math/big"
	// "crypto/ecdsa"
	// "github.com/ethereum/go-ethereum/crypto"
	// "github.com/ethereum/go-ethereum/core/vm"
	// "github.com/ethereum/go-ethereum/core/asm"
)
  
func main() {
	var aa  = make(map[int]string)
	aa[1] = "test"
	if a,ok := aa[2]; ok{
		fmt.Printf(a,ok)
	} else {
		fmt.Println(ok)
	}
}
