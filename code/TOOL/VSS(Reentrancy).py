#Slice Criteria Reentrancy ：
#fallback () 
#call.value () 
#a function involving call.value 
#variable: correspond to user balance 

var_list = [
    'balances[msg.sender]', 'participated[msg.sender]', 'playerPendingWithdrawals[msg.sender]',
    'nonces[msgSender]', 'balances[beneficiary]', 'transactions[transactionId]', 'tokens[token][msg.sender]',
    'totalDeposited[token]', 'tokens[0][msg.sender]', 'accountBalances[msg.sender]', 'accountBalances[_to]',
    'creditedPoints[msg.sender]', 'balances[from]', 'withdrawalCount[from]', 'balances[recipient]',
    'investors[_to]', 'Bal[msg.sender]', 'Accounts[msg.sender]', 'Holders[_addr]', 'balances[_pd]',
    'ExtractDepositTime[msg.sender]', 'Bids[msg.sender]', 'participated[msg.sender]', 'deposited[_participant]',
    'Transactions[TransHash]', 'm_txs[_h]', 'balances[investor]', 'this.balance', 'proposals[_proposalID]',
    'accountBalances[accountAddress]', 'Chargers[id]', 'latestSeriesForUser[msg.sender]',
    'balanceOf[_addressToRefund]', 'tokenManage[token_]', 'milestones[_idMilestone]', 'payments[msg.sender]',
    'rewardsForA[recipient]', 'userBalance[msg.sender]', 'credit[msg.sender]', 'credit[to]', 'round_[_rd]',
    'userPendingWithdrawals[msg.sender]', '[msg.sender]', '[from]', '[to]', '[_to]', 'call.value',"msg.sender",'[call.value]'
]

# 文件路径
#file_path = './preprocessing/test.sol'
file_path = './preprocessing/hello.sol'
# 1.读取文件
with open(file_path, 'r') as file:
    contract_code = file.read()

# 2.先提取call.value所包含的变量 A...
extracted_variables_from_code = extract_variables_from_call_value_in_code(contract_code)
#print(extracted_variables_from_code)

# 3.再提取与A...直接相关的变量B... ：数据流关系
process_extracted_variables(contract_code, extracted_variables_from_code)

# 4.添加所有关键变量A B C ...
var_list.extend(extracted_variables_from_code)
# Print the updated variable list
#print(var_list)

# 5. 提取代码片段：控制流关系
extracted_snippets = code_slicing(contract_code, var_list)
unique_sliced_code = list(dict.fromkeys(extracted_snippets))
# 打印提取的代码片段
for snippet in unique_sliced_code:
    print(snippet)