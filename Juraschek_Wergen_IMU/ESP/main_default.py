##############################################################################################
#                                  PACKAGES AND LIBRARIES                                    #
##############################################################################################
import machine
from machine import Timer, I2C
import wifi
from imu import MPU6050
from vector3d import Vector3d
from time import sleep, time, ticks_ms
import ubinascii
from umqttsimple import MQTTClient
import usocket as socket
       
##############################################################################################
#                                       SETTINGS                                             #
##############################################################################################
fs = 100
gateway_IP = "ADD HERE YOUR IP!"

i2c = machine.I2C(scl=machine.Pin(5), sda=machine.Pin(4), freq=100000)
mpu6050 = MPU6050(i2c)
        
##############################################################################################
#                                         WIFI                                               #
##############################################################################################
wifi.connect_wifi()
sleep(1) 

##############################################################################################
#                           TCP - DATA COLLECTION AND SENDING                                #
##############################################################################################
BUFFER_SIZE = 1024
s = socket.socket()
try:
    print("Waiting for gateway..")
    addrinfos = socket.getaddrinfo(gateway_IP, 8000)
    s.connect(addrinfos[0][4])
    print("Connected to gateway via TCP-IP!")
    sleep(2)
    
except OSError as e:
    print("Connection via TCP-IP failed. Check your IP in line 18!")
    print("Reset machine in 10 seconds..")
    sleep(10)
    machine.reset()
    
##############################################################################################
#                       TIMER - SETTING,INIT AND READ IMU FUNCTION                           #
##############################################################################################
accel = mpu6050.accel
counter = 0
time_reduce = 0
def read_imu(tim):
    global accel, counter, time_reduce

    if counter ==0:
        time_reduce = ticks_ms()
        
    string=accel.xyz + str((ticks_ms()-time_reduce)/1000) + '\n'
    
    counter+=1
    
    s.send(string.encode())

print("Start sending data..")
tim = Timer(-1)
tim.init(period=int((1/fs)*1000), mode=Timer.PERIODIC, callback=read_imu)

t_start = time()
  
##############################################################################################
#                                      MAIN PROGRAMM                                         #
##############################################################################################
while True:
    
    t_stop = time()
    
    laufzeit = (t_stop-t_start)    
    
    if laufzeit == 100:
        tim.deinit()
        print("PROZESSAUFNAHME BEENDET:")
        print('Laufzeit: ', laufzeit,'Sekunden')
        print('Frequenz: ', fs, 'Hz')
        print('Gesendet Soll: ', laufzeit*fs)

        sleep(5)
        s.close()

        break
