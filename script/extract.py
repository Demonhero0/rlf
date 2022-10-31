import os
import json
import shutil
import signal
import argparse
import subprocess
import json

accounts = [
    "1c6dbb1fe61bbb7c256f0ffcbd34087e211173dbc8454220b8b166ed6ada5c00",
	"b1cff43bf95333788b080b6cd5c5e2fcbe321ccd4132ed80cb3e72478c69e9a7",
	"aa3eeb453426d9c9292f89be5fa7e6caa0330d312255f84c0caa6764ae1adf00",
	"34a5a824b045c9ce797589d334394c11ee28d9cd8757f1a9b0ccf0fd0008c641",
	"a7a163dcb33958498cf5736282f53e39bd6cb7a58f5d4a948445dc86faa34f90",
]
amount = '100000000000000000000000000000'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj', dest='proj', type=str, required=True)
    parser.add_argument('--port', dest='port', type=int, default=8545)
    parser.add_argument('--fuzz_contract', dest='fuzz_contract', type=str, default=None)
    parser.add_argument('--params', dest='params', type=str, default=None)
    args = parser.parse_args()
    return args


args = get_args()


def modify_truffle_js():
    global args

    s = ('module.exports = {\n'
         '  networks: {\n'
         '    development: {\n'
         '      host: "127.0.0.1",\n'
         '      port: ' + str(args.port) + ',\n'
         '      network_id: "*",\n'
         '      gas: 1000000000\n'
         '    }\n'
         '  },\n'
         '  compilers: {\n'
         '     solc: {\n'
         '       version: "0.4.25",\n'
         '       optimizer: {\n'
         '         enabled: true,\n'
         '         runs: 200\n'
         '       }\n'
         '     }\n'
         '  }\n'
         '};')

    with open(os.path.join(args.proj, 'truffle-config.js'), 'w') as f:
        f.write(s)


def run_ganache():
    account_cmd = []
    for account in accounts:
        account_cmd.append('--account=0x{},{}'.format(account, amount))

    cmd = ['ganache-cli', '-p', str(args.port), '--gasLimit', '0xfffffffffffff'] + account_cmd
    pid = subprocess.Popen(cmd).pid
    return pid

remove_list = ['Migrations.json']

def extract_transactions():
    global args
    os.chdir(args.proj)
    build_path = os.path.join(args.proj, 'build')
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    subprocess.call('truffle compile', shell=True)

    # deploy and extract transaction
    # find the main contract
    # print(os.getcwd())
    # code_length_dict = dict()
    # source_path = "build/contracts"
    # for filename in os.listdir(source_path):
    #     if filename in remove_list:
    #         continue
    #     with open(f"{source_path}/{filename}") as f:
    #         j = json.load(f)
    #         code_length_dict[j['contractName']] = len(j['bytecode'])
    # fuzz_contract = sorted(code_length_dict.items(), key = lambda kv:(kv[1], kv[0]))[-1][0]
    # print(fuzz_contract)
    # generate_the_migrations(fuzz_contract)
    if not args.params:
        generate_the_migrations(fuzz_contract=args.fuzz_contract)

    subprocess.call('truffle deploy', shell=True)
    extract_js_path = os.path.join(os.environ['GOPATH'], 'src', 'ilf', 'script', 'extract.js')
    subprocess.call('truffle exec {}'.format(extract_js_path), shell=True)

def generate_the_migrations(fuzz_contract):
    s = ('var contract = artifacts.require("'+fuzz_contract+'"); \n'
         'module.exports = function(deployer) {\n'
         '   deployer.deploy(contract);\n'
         '};\n'
        )
    # print(os.getcwd())
    with open(os.path.join('migrations/2_deploy_contracts.js'), 'w') as f:
        f.write(s)


def main():
    global args
    # print('modify_truffle_js')
    modify_truffle_js()
    # print('run_ganache')
    pid = run_ganache()
    # print('extract_transactions')
    extract_transactions()
    os.kill(pid, signal.SIGTERM)


if __name__ == '__main__':
    main()

    # with open('result.json','r') as f:
    #     result = json.load(f)
    
    # res = dict()
    # for address,content in result.items():
    #     if 'v0.4.25' in content['version_name']:
    #         res[address] = content
    # print(len(res))
    # with open('result_v0425.json','w') as f:
    #     json.dump(res,f)