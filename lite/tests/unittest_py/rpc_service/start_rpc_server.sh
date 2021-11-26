# get process-id of current rpc server
RPC_PID=$(lsof -i:18812 | grep Python | awk -F ' ' '{print $2}')
# restart rpc server
if [[ $RPC_PID != "" ]]; then
  kill -9 ${RPC_PID}
fi
python3.9 server.py
