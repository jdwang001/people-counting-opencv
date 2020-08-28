#ipcam=(192.168.1.161 192.168.1.103 192.168.1.140 192.168.1.104 192.168.1.106 192.168.1.84 192.168.1.54 192.168.1.65 192.168.1.71 192.168.1.42 192.168.1.23 192.168.1.6 192.168.2.63 192.168.2.97 192.168.2.83 192.168.2.94 192.168.2.103 192.168.2.131 192.168.2.48 192.168.2.3)
ipcam=(192.168.1.104 192.168.1.136 192.168.1.140 192.168.1.142 192.168.1.161 192.168.1.194 192.168.1.54 192.168.1.6 192.168.1.84 192.168.1.93 192.168.2.103 192.168.2.131 192.168.2.48 192.168.2.83 192.168.2.96)

for ip in ${ipcam[@]}
do
	#rtspadd="rtsp://admin:admin@"$ip":554/cam/realmonitor?channel=1&subtype=1"
	rtspadd="rtsp://admin:admin@"$ip
	echo $ip $rtspadd
	#/home/wp/.virtualenvs/cv/bin/python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel -l 0,150 -l 500,150 -s 5 --input $rtspadd  2>&1 &
	/home/wp/.virtualenvs/cv/bin/python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel -l 0,170 -l 500,170 --input $rtspadd 2>&1 &
done
