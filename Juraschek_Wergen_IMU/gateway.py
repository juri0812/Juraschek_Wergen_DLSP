import sys
import os
import threading

import Qt.gui
import Qt.icons.icons
from Qt.gui import *
import PySide6.QtCharts
from PySide6 import QtCharts

import usocket as socket
import paho.mqtt.client as mqtt
from pymongo import MongoClient, errors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time import localtime, sleep

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):  # Initialisieren der Klasse
        print("Initiate programm..")
        super().__init__()
        self.setupUi(self)

        gateway_IP = 'ADD HERE YOUR IP!'
        mongoDB_Cloud = 'ADD HERE YOUR MONGODB CLOUD ADDRESS!'  # Without address the program uses localhost

        self.no_cloud = False
        self.no_mqtt = False

        ################################################################################################################
        # MQTT SETTINGS
        ################################################################################################################
        self.client = mqtt.Client()
        self.mqtt_server = gateway_IP
        self.mqtt_port = 1883
        self.mqtt_con_count = 0

        ################################################################################################################
        # TCP SETTINGS
        ################################################################################################################
        self.TCP_IP = gateway_IP
        self.TCP_PORT = 8000
        self.BUFFER_SIZE = 15000
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        ################################################################################################################
        # MONGODB SETTINGS
        ################################################################################################################
        try:
            print("Establishing connection to MongoDB Atlas cloud, please wait..")
            self.mDB_client01 = MongoClient(mongoDB_Cloud)
            self.mDB_client01.server_info()
            self.first_info_lbl.setText("Ready for connection, simulation or plotting!")
        except errors.ServerSelectionTimeoutError:
            print("Connection to MongoDB failed. Check the adress in line 32 or ask developer for access!")
            print("Using localhost for MongoDB connection..")
            self.mDB_client01 = MongoClient("mongodb://localhost:27017")
            self.no_cloud = True

        ################################################################################################################
        # MACHINE LEARNING SETTINGS
        ################################################################################################################
        self.classifier = MLPClassifier(alpha=0.03, max_iter=5000, verbose=False, early_stopping=False)
        self.sc = StandardScaler()
        self.ClassifierColumns = 50

        ################################################################################################################
        # CHART SETTINGS
        ################################################################################################################
        self.acce_series_ax = QtCharts.QLineSeries()
        self.acce_series_ay = QtCharts.QLineSeries()
        self.acce_series_az = QtCharts.QLineSeries()

        self.acce_chart = QtCharts.QChart()

        self.label_font = QFont('Arial', 12)
        self.title_font = QFont('Arial', 15)
        self.label_font.setBold(False)
        self.title_font.setBold(True)

        self.acce_series_ax.setName(" a_x")
        self.acce_series_ay.setName(" a_y")
        self.acce_series_az.setName(" a_z")

        self.acce_chart.addSeries(self.acce_series_ax)
        self.acce_chart.addSeries(self.acce_series_ay)
        self.acce_chart.addSeries(self.acce_series_az)

        self.acce_chart.legend().setAlignment(Qt.AlignRight)
        self.acce_chart.createDefaultAxes()
        self.acce_chart.setTheme(PySide6.QtCharts.QChart.ChartThemeDark)

        self.acce_chart.setTitleFont(self.title_font)
        self.acce_chart.setTitle('MPU6050 acceleration data')

        self.acce_chart.axisX().setTitleFont(self.label_font)
        self.acce_chart.axisX().setTitleText('Time')

        self.acce_chart.axisY().setTitleFont(self.label_font)
        self.acce_chart.axisY().setTitleText('Acceleration')

        self.acce_chartView.setLineWidth(0.5)
        self.acce_chartView.setChart(self.acce_chart)
        self.axis_to_plot_cb.clear()

        for axes in ['all', 'ax', 'ay', 'az', 'a_abs']:
            self.axis_to_plot_cb.addItem(axes)

        ################################################################################################################
        # SIMULATION SETTINGS
        ################################################################################################################
        self.stop_simulation_flag = False

        ################################################################################################################
        # UPDATE HISTORY COMBOBOX
        ################################################################################################################
        self.saved_data_cb.clear()

        list_of_collections = []

        # Collections from LocalStorage:
        for root, dirs, files in os.walk(os.getcwd() + '\LocalStorage\MongoDB_collections'):
            for name in files:
                fullname = os.path.join(name.replace('.csv', ''))
                list_of_collections.append(fullname)

        # Cloud/LocalHost collections:
        for db in self.mDB_client01.list_database_names():
            if db == 'IMU':
                for co in self.mDB_client01[db].list_collection_names():
                    collection_name = co[:].replace(' | ', '_')
                    if collection_name not in list_of_collections:
                        list_of_collections.append(collection_name)

        for name in list_of_collections:
            self.saved_data_cb.addItem(name)

        self.saved_data_cb.model().sort(0, Qt.AscendingOrder)
        self.saved_data_cb.setCurrentIndex(0)
        self.axis_to_plot_cb.setCurrentIndex(0)

        ################################################################################################################
        # THREADS
        ################################################################################################################
        self.wt_washing_analysis = threading.Thread(target=self.send_freq)
        self.wt_simulation = threading.Thread(target=self.simulate)
        self.wt_prepare_machine_learning = threading.Thread(target=self.prepare_machine_learning)

        ################################################################################################################
        # BUTTON SETTINGS
        ################################################################################################################
        self.connect_to_esp_cmd.clicked.connect(self.start_washing_analysis)
        self.start_simulation_cmd.clicked.connect(self.start_simulation)
        self.show_history_cmd.clicked.connect(self.show_history_plot)

        ################################################################################################################
        # LOAD BACKGROUND DATA
        ################################################################################################################
        print("Loading training data and fitting classifier..")
        self.wt_prepare_machine_learning.start()
        self.wt_prepare_machine_learning.join()

        self.connection_fr.setStyleSheet(u"background-color: rgb(0, 45, 66);")
        self.frequency_spinButton.setEnabled(True)
        self.connect_to_esp_cmd.setEnabled(True)
        self.connect_to_esp_cmd.setStyleSheet(u"background-color: rgb(0, 85, 127); color: rgb(255, 255, 255);")
        self.start_simulation_cmd.setEnabled(True)
        self.start_simulation_cmd.setStyleSheet(u"background-color: rgb(100, 104, 126);color: rgb(255, 255, 255)")
        print("Start programm..")

    ####################################################################################################################
    # FUNCTIONS TO START WORKER THREADS VIA BUTTONS
    ####################################################################################################################
    def start_washing_analysis(self):
        self.disable_menu(False)
        self.wt_washing_analysis.start()

    def start_simulation(self):
        self.disable_menu(True)
        self.axis_to_plot_cb.setCurrentIndex(0)

        if self.start_simulation_cmd.text() == "simulate":
            self.start_simulation_cmd.setText("stop")
            self.wt_simulation.start()

        else:
            self.stop_simulation_flag = True
            self.info_fr.setStyleSheet(u"background-color: rgb(105, 225, 140);")
            self.start_simulation_cmd.setStyleSheet(u"background-color: rgb(56, 56, 56);")
            self.first_info_lbl.setText("Well done!")
            self.second_info_lbl.setText("Please restart for new investigation..")
            self.start_simulation_cmd.setText("simulate")
            self.menue_fr.setEnabled(False)

    def disable_menu(self, sim):
        self.connect_to_esp_cmd.setEnabled(False)
        self.connect_to_esp_cmd.setStyleSheet(u"background-color: rgb(56, 56, 56);")

        self.frequency_spinButton.setEnabled(False)

        self.saved_data_cb.setEnabled(False)
        self.saved_data_cb.setStyleSheet(u"background-color: rgb(56, 56, 56);")

        self.axis_to_plot_cb.setEnabled(False)
        self.axis_to_plot_cb.setStyleSheet(u"background-color: rgb(56, 56, 56);")

        self.show_history_cmd.setEnabled(False)
        self.show_history_cmd.setStyleSheet(u"background-color: rgb(56, 56, 56);")

        if not sim:
            self.start_simulation_cmd.setEnabled(False)
            self.start_simulation_cmd.setStyleSheet(u"background-color: rgb(56, 56, 56);")

    ####################################################################################################################
    # TARGET FUNCTION - MACHINE LEARNING
    ####################################################################################################################
    def prepare_machine_learning(self):
        df = pd.read_csv('LocalStorage/2022.02.12_MLP_training/CL_100_2022-02-12_13-25-48.csv')
        split_factor = 0.25
        x = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_factor, random_state=1, stratify=y)
        self.sc.fit(x_train)
        x_train_std = self.sc.transform(x_train)
        self.classifier.fit(x_train_std, y_train)

    def predict(self, df_both_sets):
        a_x = df_both_sets.iloc[0:self.ClassifierColumns*2, 0].values
        a_y = df_both_sets.iloc[0:self.ClassifierColumns*2, 1].values
        a_z = df_both_sets.iloc[0:self.ClassifierColumns*2, 2].values

        a_abs = np.sqrt(a_x*a_x + a_y*a_y + a_z*a_z)
        a_abs_freq = self.calcFFT(a_abs, len(a_abs))

        a_abs_freq = np.delete(a_abs_freq, 0)
        a_abs_freq = a_abs_freq.reshape(1, -1)

        a_abs_freq_std = self.sc.transform(a_abs_freq)

        y_pred = self.classifier.predict(a_abs_freq_std)
        proba = self.classifier.predict_proba(a_abs_freq_std)

        if float(max(proba[0])) > 0.90:
            pred = y_pred[0]
            proba = 'Probability of prediction: ' + str(round(max(proba[0]), 4))
            if pred == 0:
                self.show_state([True, False, False, False, False], proba)
            elif pred == 1:
                self.show_state([False, True, False, False, False], proba)
            elif pred == 2:
                self.show_state([False, False, True, False, False], proba)
            elif pred == 3:
                self.show_state([False, False, False, True, False], proba)
            elif pred == 4:
                self.show_state([False, False, False, False, True], proba)

    def show_state(self, state_lis, proba):
        self.standstill_light.setEnabled(state_lis[0])
        self.standstill_lbl.setEnabled(state_lis[0])
        self.pump_in_light.setEnabled(state_lis[1])
        self.pump_in_lbl.setEnabled(state_lis[1])
        self.washing_light.setEnabled(state_lis[2])
        self.washing_lbl.setEnabled(state_lis[2])
        self.pump_out_light.setEnabled(state_lis[3])
        self.pump_out_lbl.setEnabled(state_lis[3])
        self.spinning_light.setEnabled(state_lis[4])
        self.spinning_lbl.setEnabled(state_lis[4])
        self.second_info_lbl.setText(proba)

    def calcFFT(self, accel, nr_samples):
        accel_without_mean = accel - np.mean(accel)  # Subtract mean Value to reduce the DC Offset in the FFT
        freq = np.fft.rfft(accel_without_mean, nr_samples, norm='ortho')
        freq = np.abs(freq)
        freq = freq / nr_samples  # Normalize the Amplitude by the known sample number

        return freq

    ####################################################################################################################
    # TARGET FUNCTION - WASHING ANALYSIS
    ####################################################################################################################
    def send_freq(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        try:
            print("Establishing MQTT connection to ESP..")
            self.client.connect(self.mqtt_server, self.mqtt_port)
            self.client.loop_forever()
        except ConnectionRefusedError:
            print("MQTT connection to ESP failed. Please check Server ID in line 31 or use 'main_default.py' on ESP")
            self.no_mqtt = True
            self.first_info_lbl.setText("No MQTT communication to ESP.")
            self.second_info_lbl.setText("Use main_default.py (100 Hz)!")
            self.info_fr.setStyleSheet(u"background-color: rgb(240, 85, 95);")
            self.establish_tcp_connection()
        except socket.gaierror:
            print("IP is missing.. add your IP in line 31 and restart the program!")
            self.connect_to_esp_cmd.setStyleSheet(u"background-color: rgb(56, 56, 56);")
            self.start_simulation_cmd.setStyleSheet(u"background-color: rgb(56, 56, 56);")
            self.show_history_cmd.setStyleSheet(u"background-color: rgb(56, 56, 56);")
            self.connect_to_esp_cmd.setEnabled(False)
            self.start_simulation_cmd.setEnabled(False)
            self.connect_to_esp_cmd.setEnabled(False)
            self.frequency_spinButton.setEnabled(False)
            self.first_info_lbl.setText("IP is missing..")
            self.second_info_lbl.setText("Add your IP in line 31 and restart..")
            self.info_fr.setStyleSheet(u"background-color: rgb(240, 85, 95);")

    ####################################################################################################################
    # MQTT - SEND FREQUENCY TO ESP
    ####################################################################################################################
    def on_connect(self, mqttc, obj, flags, rc):
        print("Connected with result code " + str(rc))
        self.client.subscribe('/esp_connected')
        self.client.subscribe('/msg_recieved')
        print("Publish frequency..")
        self.client.publish('/frequency', str(self.frequency_spinButton.value()))
        self.freq_send_ico.setEnabled(True)

    def on_message(self, mqttc, userdata, msg):
        print("ESP received frequency!")
        self.client.disconnect()
        sleep(1)
        self.machine_ico.setEnabled(True)
        self.establish_tcp_connection()

    ####################################################################################################################
    # TCP - RECEIVE ACCELERATION DATA FROM ESP
    ####################################################################################################################
    def establish_tcp_connection(self):
        print("Connected via TCP!")
        self.s.bind((self.TCP_IP, self.TCP_PORT))
        self.s.listen(1)
        print("Listening...")
        conn, addr = self.s.accept()
        print("Connection to gateway succesfull!")
        print("Collecting data..")
        self.collect_data(conn, addr)

    ####################################################################################################################
    # COLLECT ACCELERATION DATA
    ####################################################################################################################
    def collect_data(self, conn, addr):
        if self.no_mqtt:
            self.freq_send_ico.setEnabled(False)
        else:
            self.freq_send_ico.setEnabled(True)

        if self.no_cloud:
            self.cloud_lbl.setText("Local")
        else:
            self.cloud_lbl.setText("Cloud")

        self.machine_ico.setEnabled((True))
        self.tcp_connection_ico.setEnabled(True)
        self.gateway_ico.setEnabled(True)
        self.plus_ico.setEnabled(True)
        self.gui_ico.setEnabled(True)
        self.data_send_ico.setEnabled(True)
        self.cloud_ico.setEnabled(True)
        self.first_info_lbl.setText("TCP-Addr.: " + str(addr))
        self.second_info_lbl.setText('Connection successful!')

        total_sample_length = 0
        df = pd.DataFrame()
        data_string = ''
        counter = 1
        last_time = 0.0

        db01 = self.mDB_client01['IMU']
        save_name = str(self.frequency_spinButton.value()) + '_' + \
                    '{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(localtime()[0], localtime()[1],
                                                                       localtime()[2], localtime()[3],
                                                                       localtime()[4], localtime()[5])

        while 1:
            data = conn.recv(self.BUFFER_SIZE)
            data_string += data.decode()

            if len(data_string) != 0:
                if data_string.count('\n')/2 >= 100 or not data:
                    list_of_data = data_string.split('\n')
                    data_string = ''
                    list_of_data_clean = []
                    x_time = []

                    for n in range(0, len(list_of_data)):
                        data_split = list(filter(None,list_of_data[n].split(',')))

                        if len(data_split) == 3 and n != len(list_of_data)-1:
                            #print('Checked_Value_length 1:',data_split[0],len(data_split[0].replace('-', '')))
                            #print('Checked_Value_length 2:',data_split[1],len(data_split[1].replace('-', '')))
                            #print('Checked_Value_length 3:',data_split[2],len(data_split[2].replace('-', '')))

                            if len(data_split[0].replace('-', '')) == 7 and len(data_split[1].replace('-', '')) == 7 and len(data_split[2].replace('-', '')) == 7:
                                #print('Check_time_length:',len(list_of_data[n+1]))

                                if len(list_of_data[n+1].split('.')) == 2:
                                    if float(list_of_data[n+1]) > last_time:
                                        list_of_data_clean.append(np.float_(data_split))
                                        x_time.append(float(list_of_data[n+1]))
                                        last_time = float(list_of_data[n+1])
                                    #else:
                                     #   print('Timestamps:', last_time, float(list_of_data[n + 1]))

                    #print('X_Values:',list_of_data_clean)
                    #print('X_time:',len(x_time))

                    sample_length = len(list_of_data_clean)
                    total_sample_length += sample_length

                    df_new = pd.DataFrame(list_of_data_clean, columns=['a_x', 'a_y', 'a_z'])
                    df_new['x_time'] = x_time

                    df = pd.concat([df, df_new])
                    df.reset_index(drop=True, inplace=True)

                    if counter == 2 and len(df) >= self.ClassifierColumns*2:
                   
                        self.predict(df)
                        counter = 0

                    self.paint(df)
                    self.save(df_new, db01, save_name)

                    df = df_new
                    counter += 1

            if not data:
                print("Close TCP connection...")
                conn.close()
                print("TCP connection closed!")
                print('Received datasets:', total_sample_length)

                self.cloud_ico.setEnabled(False)
                self.data_send_ico.setEnabled(False)
                self.plus_ico.setEnabled(False)
                self.gateway_ico.setEnabled(False)
                self.tcp_connection_ico.setEnabled(False)
                self.machine_ico.setEnabled(False)
                self.freq_send_ico.setEnabled(False)
                self.info_fr.setStyleSheet(u"background-color: rgb(105, 225, 140);")
                self.first_info_lbl.setText('Well done!')
                self.second_info_lbl.setText('Restart for new analysis!')

                break

    ####################################################################################################################
    # VISUALIZE ACCELERATION DATA AND MACHINE STATE
    ####################################################################################################################
    def paint(self, df):
        self.acce_series_ax.clear()
        self.acce_series_ay.clear()
        self.acce_series_az.clear()

        sleep(.002)
        for i in range(0, len(df)):
            self.acce_series_ax.append(df['x_time'][i], df['a_x'][i])
            self.acce_series_ay.append(df['x_time'][i], df['a_y'][i])
            self.acce_series_az.append(df['x_time'][i], df['a_z'][i])

        a_max_of_all = max([df['a_x'].max(), df['a_y'].max(), df['a_z'].max()])
        a_min_of_all = min([df['a_x'].min(), df['a_y'].min(), df['a_z'].min()])

        self.acce_chart.axisY().setRange(a_min_of_all - 0.1, a_max_of_all + 0.1)
        self.acce_chart.axisX().setRange(df['x_time'].min(), df['x_time'].max())

    ####################################################################################################################
    # MONGODB - SAVE COLLECTED DATA IN CLOUD
    ####################################################################################################################
    def save(self, df_new, db01, save_name):
        imu_list = []

        for row in df_new.to_dict(orient='split')['data']:
            imu_list.append({'a_x': row[0], 'a_y': row[1], 'a_z': row[2], 'x_time': row[3]})

        result = db01[save_name].insert_many(imu_list)

    ####################################################################################################################
    # MONGODB - SIMULATION OF HISTORIC WASHING PROCESSES
    ####################################################################################################################
    def simulate(self):
        collection = self.saved_data_cb.currentText()
        try:
            print("Try to load data from local storage for simulation, please wait..")
            xyz = pd.read_csv('LocalStorage/MongoDB_collections/' + collection + '.csv')
            print("Loading local data for simulation successful!")
        except:
            print("Load local data for simulation failed...")
            print("Try to loading data from cloud storage for simulation, please wait..")
            xyz = pd.DataFrame(self.mDB_client01['IMU'][collection].find().sort('_id'))
            print("Loading cloud data for simulation successful!")

        self.first_info_lbl.setText("Run simulation...")
        self.machine_ico.setEnabled(True)

        start = 0
        set_length = 100
        counter = 1

        for delay in range(0, len(xyz), 50):
            if self.stop_simulation_flag:
                self.machine_ico.setEnabled(True)
                break

            df_both_sets = xyz.iloc[start+delay:set_length+delay, 1:5]
            df_both_sets.reset_index(drop=True, inplace=True)

            if counter == 2 and len(df_both_sets) > self.ClassifierColumns:
                self.predict(df_both_sets)
                counter = 0

            self.paint(df_both_sets)

            counter += 1
            sleep(1)

    ####################################################################################################################
    # MONGODB - PLOT ACCELERATION AND FREQUENCY OF HISTORIC WASHING PROCESSES
    ####################################################################################################################
    def show_history_plot(self):
        if self.saved_data_cb.currentIndex() > -1:
            collection = self.saved_data_cb.currentText()
            freq = int(collection.split('_')[0])
            try:
                print("Try to load data from local storage for plot, please wait..")
                xyz = pd.read_csv('LocalStorage/MongoDB_collections/' + collection + '.csv')
                print("Loading local data for plot successful!")
            except:
                print("Load local data failed...")
                print("Try to loading data from cloud storage for plot, please wait..")
                xyz = pd.DataFrame(self.mDB_client01['IMU'][collection].find().sort('_id'))
                print("Loading cloud data for plot successful!")

            ax_acce = xyz.iloc[:, 1]
            ay_acce = xyz.iloc[:, 2]
            az_acce = xyz.iloc[:, 3]
            x_time = xyz.iloc[:, 4]
            a_abs = np.sqrt(ax_acce * ax_acce + ay_acce * ay_acce + az_acce * az_acce)

            ax_freq = self.calcFFT(np.array(ax_acce), len(ax_acce))
            ay_freq = self.calcFFT(np.array(ay_acce), len(ay_acce))
            az_freq = self.calcFFT(np.array(az_acce), len(az_acce))
            a_abs_freq = self.calcFFT(np.array(a_abs), len(a_abs))
            x_freq = np.linspace(0.0, (freq / 2), int(len(xyz) / 2) + 1)

            fttfig, (ax1, ax2) = plt.subplots(2, figsize=(5, 5), num=self.saved_data_cb.currentText() + ' | ' + self.axis_to_plot_cb.currentText())

            if self.axis_to_plot_cb.currentText() == 'all':
                ax1.plot(x_time, ax_acce, '.-', label="a_x", linewidth=0.5, ms=1)
                ax1.plot(x_time, ay_acce, '.-', label="a_y", linewidth=0.5, ms=1)
                ax1.plot(x_time, az_acce, '.-', label="a_z", linewidth=0.5, ms=1)
                ax2.plot(x_freq, ax_freq, '.-', label="a_x_freq", linewidth=0.5, ms=1)
                ax2.plot(x_freq, ay_freq, '.-', label="a_y_freq", linewidth=0.5, ms=1)
                ax2.plot(x_freq, az_freq, '.-', label="a_z_freq", linewidth=0.5, ms=1)

            elif self.axis_to_plot_cb.currentText() == 'ax':
                ax1.plot(x_time, ax_acce, '.-', label="a_x", linewidth=0.5, ms=1)
                ax2.plot(x_freq, ax_freq, '.-', label="ax_freq", linewidth=0.5, ms=1)

            elif self.axis_to_plot_cb.currentText() == 'ay':
                ax1.plot(x_time, ay_acce, '.-', label="a_y", linewidth=0.5, ms=1)
                ax2.plot(x_freq, ay_freq, '.-', label="ay_freq", linewidth=0.5, ms=1)

            elif self.axis_to_plot_cb.currentText() == 'az':
                ax1.plot(x_time, az_acce, '.-', label="a_z", linewidth=0.5, ms=1)
                ax2.plot(x_freq, az_freq, '.-', label="az_freq", linewidth=0.5, ms=1)

            elif self.axis_to_plot_cb.currentText() == 'a_abs':
                ax1.plot(x_time, a_abs, '.-', label="a", linewidth=0.5, ms=1)
                ax2.plot(x_freq, a_abs_freq, '.-', label="a_freq", linewidth=0.5, ms=1)

            ax1.set_title("IMU history from MongoDB")
            ax1.set(xlabel="Time")
            ax1.set(ylabel="Acceleration")
            ax1.legend()
            ax1.grid(True)

            ax2.set(xlabel="Frequency")
            ax2.set(ylabel="Amplitude")
            ax2.legend()
            ax2.grid(True)

            plt.show()


app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec())
