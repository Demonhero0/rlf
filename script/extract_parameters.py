#!/usr/bin/env python
# coding: utf-8

# In[20]:


abi = {"inputs":[{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactETHForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"payable","type":"function"}


# In[31]:


def decodeParameter(abi, paraBytecode):
    inputs = abi['inputs']
    length = len(inputs)
    arg = {}
    typeList = []
    nameList = []
    bytecodeList = []
    for i in range(int(len(paraBytecode) / 64)):
        bytecodeList.append(paraBytecode[i*64 : i*64+64])
    for i in range(length):
        typeList.append(inputs[i]['type'])
        nameList.append(inputs[i]['name'])
        if('[]' not in typeList[i]):
            if("address" in typeList[i]):
                arg[nameList[i]] = "0x" + str(bytecodeList[i])[-40:]
            else:
                arg[nameList[i]] = "0x" + str(bytecodeList[i]).lstrip("0")
        else:
            loc = int(str(bytecodeList[i]).lstrip("0"), base=16)
            number = int(str(paraBytecode[loc*2 : loc*2+64]).lstrip("0"), base=16)
            arg[nameList[i]] = []
            for j in range(number):
                if("address" in typeList[i]):
                    arg[nameList[i]].append("0x" + str(bytecodeList[int(loc/32)+j+1])[-40:])
                else:
                    arg[nameList[i]].append("0x" + str(bytecodeList[int(loc/32)+j+1]).lstrip("0"))
    return arg


# In[21]:


testPara = "0000000000000000000000000000000000000000000000004113275ecda81fdb0000000000000000000000000000000000000000000000000000000000000080000000000000000000000000a255b73dbd9938008d3ede7fd0a152ecc905758800000000000000000000000000000000000000000000000000000000618fe9d10000000000000000000000000000000000000000000000000000000000000002000000000000000000000000bb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c0000000000000000000000000e09fabb73bd3ade0a17ecc321fd13a19e81ce82"


# In[32]:

import os
import csv
import json
# contract_list = [x.split('.')[0] for x in os.listdir('v0425_source')]
contract_list = os.listdir('../built/v0425')
contract_set = set(contract_list)

csvFile = open("contract_information_20211027.csv","r")
reader = csv.reader(csvFile)
contract_info_dict = {}
key_list = []
for item in reader:
    if reader.line_num == 1:
        key_list = item
        key_list = key_list[1:]
        # print(len(key_list),key_list)
    else:
        value_list = item
        if value_list[0] in contract_set:
            contract_info_dict[value_list[0]] = dict()
            for index,value in enumerate(value_list[1:]):
                contract_info_dict[value_list[0]][key_list[index]] = value

print(len(contract_info_dict))
# with open('contract_info_dict.json','w') as f:
#     json.dump(contract_info_dict,f)

# In[ ]:




# %%
import json
path = '../built/v0425'
built = 0
constructor = 0
constructor_arg = 0
error = 0
for index, address in enumerate(os.listdir(path)[:]):
    if 'build' in os.listdir(f'{path}/{address}'):
        built += 1
        contract_name = contract_info_dict[address]['ContractName']
        with open(f'{path}/{address}/build/contracts/{contract_name}.json','r', encoding='utf-8') as f:
            abi_json = json.load(f)
        for abi in abi_json['abi']:
            if abi['type'] == 'constructor':
                constructor += 1
                contract_info_dict[address]['constructor_abi'] = abi
                if abi['inputs']:
                    constructor_arg += 1
                    try:
                        params = decodeParameter(contract_info_dict[address]['constructor_abi'], contract_info_dict[address]['ConstructorArguments'])
                        contract_info_dict[address]['params'] = params
                    except:
                        pass
                        error += 1
                        # print(index, address, contract_info_dict[address]['ConstructorArguments'])

# %%
print(built, constructor, constructor_arg)
print(error)
print(len(contract_info_dict))
# %%
import json
with open('contract_info_dict_v0425.json','r') as f:
    contract_info_dict = json.load(f)
k = 0
for contract in contract_info_dict:
    if 'params' in contract_info_dict[contract]:
        print(contract_info_dict[contract]['params'])
        print(contract_info_dict[contract]['constructor_abi'])
        k += 1
        if k > 5:
            break

print(k)
# %%
with open('contract_info_dict_v0425.json','w') as f:
    json.dump(contract_info_dict, f)

# %%
