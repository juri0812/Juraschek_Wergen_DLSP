import network
import time
import machine
 
def connect_wifi():

    print("Connect to WiFi...")
    ssid = 'ADD YOUR SSID'
    pwd = 'ADD YOUR PASSWORD'

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if not wlan.isconnected():
        wlan.connect(ssid, pwd)
        while not wlan.isconnected():
            time.sleep_ms(100)
            pass

    print('Connection successful!')
