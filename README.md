# RLF_Artifacts
## Setup

### Manually

We provide the procedures for local setup (tested on [Ubuntu 18.04](http://releases.ubuntu.com/18.04/)).

Install [golang](https://golang.org/), for example:
```
$ wget https://dl.google.com/go/go1.10.4.linux-amd64.tar.gz
$ tar -xvf go1.10.4.linux-amd64.tar.gz
$ sudo mv go /usr/lib/go-1.10
$ echo 'export GOPATH=$HOME/go' >> ~/.bashrc
$ echo 'export GOROOT=/usr/lib/go-1.10' >> ~/.bashrc
$ echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.bashrc
$ echo 'export PATH=$PATH:$GOROOT/bin' >> ~/.bashrc
$ source ~/.bashrc
```

Clone this repo:
```
$ mkdir -p $GOPATH/src
$ cd $GOPATH/src
$ git clone https://github.com/Demonhero0/rlf.git
```

Clone [go-ethereum](https://geth.ethereum.org/) and apply our patch:
```
$ mkdir -p $GOPATH/src/github.com/ethereum
$ cd $GOPATH/src/github.com/ethereum
$ git clone https://github.com/ethereum/go-ethereum.git
$ cd go-ethereum
$ git checkout 86be91b3e2dff5df28ee53c59df1ecfe9f97e007
$ git apply $GOPATH/src/RLF_Artifacts/script/patch.geth
```

Install python dependencies:
```
$ pip3 install -r requirements.txt
```

Install execution backend:
```
$ go build -o execution.so -buildmode=c-shared export/execution.go
```

The following steps are necessary only when you want to use ILF to fuzz new contracts other than our example. Install [nodejs](https://nodejs.org/en/), [Truffle](https://www.trufflesuite.com/truffle), [web3.js](https://web3js.readthedocs.io/en/v1.2.4/) and [Ganache-CLI](https://github.com/trufflesuite/ganache-cli):
```
$ curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
$ sudo apt-get install nodejs
$ mkdir ~/.npm-global
$ npm config set prefix '~/.npm-global'
$ echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
$ source ~/.bashrc
$ npm install -g truffle web3 ganache-cli
```

Install [solc](https://github.com/ethereum/solidity) 0.4.25:
```
$ wget https://github.com/ethereum/solidity/releases/download/v0.4.25/solc-static-linux
$ chmod +x solc-static-linux
$ sudo mv solc-static-linux /usr/bin/solc
```

## Usage

### Fuzzing example

```
python3 -m rlf --proj example/crowdsale/ --contract Crowdsale --fuzzer reinforcement --limit 2000 --reward cov+bugs --mode test
```