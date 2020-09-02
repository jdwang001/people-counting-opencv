import requests
import json
import time
from wp.log import Log
from wp import common

headers = {'content-type': 'application/json', 'Connection': 'close'}
WPTIMEOUT = 3
SERVERADD = 'https://mancy.wifipix.com/api/data/cameraLog'

log = Log(__name__,common.LOGFN).getlog()

def postjsoninfo(jsondata):
    log.info("postmachinfo is %s", json.dumps(jsondata))
    trycount = WPTIMEOUT
    while trycount > 0:
        try:
            response = requests.post(SERVERADD, json.dumps(jsondata), headers=headers, timeout=WPTIMEOUT)
            response.close()

            # Consider any status other than 2xx an error
            if not response.status_code // 100 == 2:
                log.info("PostMachInfo post is error %s, %d", response, trycount)
            else:
                log.info("PostMachInfo post ok")
                break
        except requests.exceptions.RequestException as e:
            log.error("PostMachInfo is error %s ",format(e))

        trycount = trycount - 1
        time.sleep(3)
    log.info("exit postmachinfo")
