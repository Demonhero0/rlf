module.exports = {
  networks: {
    development: {
      host: "127.0.0.1",
      port: 8545,
      network_id: "*",
      gas: 1000000000
    }
  },
  compilers: {
     solc: {
       version: "0.4.25",
       optimizer: {
         enabled: true,
         runs: 200
       }
     }
  }
};