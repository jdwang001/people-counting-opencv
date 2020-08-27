import requests
import json
import time


headers = {'content-type': 'application/json', 'Connection': 'close'}
WPTIMEOUT = 3
SERVERADD = 'https://mancy.wifipix.com/api/data/cameraLog  '

def postjsoninfo(jsondata):
    print("postmachinfo is ", json.dumps(jsondata))
    trycount = WPTIMEOUT
    while trycount > 0:
        try:
            response = requests.post(SERVERADD, json.dumps(jsondata), headers=headers, timeout=WPTIMEOUT)
            response.close()

            # Consider any status other than 2xx an error
            if not response.status_code // 100 == 2:
                print("PostMachInfo post is error %s, %d", response, trycount)
            else:
                print("PostMachInfo post ok")
                break
        except requests.exceptions.RequestException as e:
            print("PostMachInfo is error ".format(e))

        trycount = trycount - 1
        time.sleep(3)
    print("exit postmachinfo")
