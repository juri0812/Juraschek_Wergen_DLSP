##############################################################################################
#                                  PACKAGES AND LIBRARIES                                    #
##############################################################################################
import machine
from machine import Pin, Timer, I2C
import wifi
from imu import MPU6050
from vector3d import Vector3d
from time import sleep, time, ticks_ms
import ubinascii
from umqttsimple import MQTTClient
import usocket as socket
       
##############################################################################################
#                                 SETTINGS AND FUNCTIONS                                     #
##############################################################################################

gateway_IP = 'ADD YOUR IP HERE!'

# LED:
gruen = Pin(14, Pin.OUT)
gruen.value(0)

blau = Pin(13, Pin.OUT)
blau.value(0)

rot = Pin(12, Pin.OUT)
rot.value(0)

# MPU6050:
i2c = machine.I2C(scl=machine.Pin(25), sda=machine.Pin(26), freq=100000)
mpu6050 = MPU6050(i2c)

# QUIT-BUTTON:
btn = Pin(27, Pin.IN, Pin.PULL_UP)

def blink_led(led, count, delay):
    for _ in range(0,count):
        led.value(1)
        sleep(delay)
        led.value(0)
        sleep(delay)
        
##############################################################################################
#                                         WIFI                                               #
##############################################################################################
# Connect Wifi:
blau.value(1)
wifi.connect_wifi()
sleep(1) 
blau.value(0)

##############################################################################################
#                                  MQTT RECIEVE FREQUENCY                                    #
##############################################################################################
mqtt_server = gateway_IP
client_id = ubinascii.hexlify(machine.unique_id())

topic_sub = '/frequency'
topic_pub01 = '/msg_recieved'
topic_pub02 = '/esp_connected'

fs = 0

def sub_cb(topic, msg):
    global fs

    msg = str(msg, 'UTF-8')   
    fs = int(msg)
    print("Frequency",fs,"Hz received!")
    
def connect_and_subscribe():
    global client_id, mqtt_server, topic_sub
    client = MQTTClient(client_id, mqtt_server, keepalive=60)
    client.set_callback(sub_cb)
    client.connect()
    client.subscribe(topic_sub)
    print('Connected to %s MQTT broker, subscribed to %s topic' % (mqtt_server, topic_sub))
    return client

def restart_and_reconnect():
    print('Failed to connect to MQTT broker/TCP IP. Reset machine..')
    rot.value(1)
    sleep(4)
    machine.reset()

try:
    client = connect_and_subscribe()
    client.wait_msg()
    gruen.value(1)
    rot.value(1)
    
except OSError as e:
    print("Check your IP in line 18!")
    restart_and_reconnect()

sleep(1)
client.publish(topic_pub02, 'recieved_freq')
client.disconnect()
print('Disconnected from MQTT broker.')
gruen.value(0)
rot.value(0)
sleep(1)

##############################################################################################
#                           TCP - DATA COLLECTION AND SENDING                                #
##############################################################################################
BUFFER_SIZE = 1024
s = socket.socket()        
try:
    blau.value(1)
    gruen.value(1)
    rot.value(1)
    print("Connect to socket...")
    addrinfos = socket.getaddrinfo(gateway_IP, 8000)
    s.connect(addrinfos[0][4])
    sleep(2)
    print("Connection sucessfull!")
    blau.value(0)
    rot.value(0)
    print("Sending data...")
    
except OSError as e:
  restart_and_reconnect()
    
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

tim = Timer(-1)
tim.init(period=int((1/fs)*1000), mode=Timer.PERIODIC, callback=read_imu)

t_start = time()
  
##############################################################################################
#                                      MAIN PROGRAMM                                         #
##############################################################################################
while True:
    
    t_stop = time()
    
    laufzeit = (t_stop-t_start)    
    #if laufzeit == 15:
    if btn.value()==0:
        tim.deinit()

        print('Laufzeit: ', laufzeit,'Sekunden')
        print('Frequenz: ', fs, 'Hz')
        print('Gesendet Soll: ', laufzeit*fs)
        
        blink_led(gruen, 2, 1)
        blau.value(1)
        sleep(5)
        s.close()
        blau.value(0)
        
        break
