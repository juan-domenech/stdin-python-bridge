"""
stdin Python Bridge

Attempt to channel waveforms coming from stdin into Twinlan UHD bridge
SCPI protocol to be read from ngscopeclient https://www.ngscopeclient.org/
"""

import signal
import sys
import logging
import argparse
import socketserver
import threading
import sys, os
import numpy as np
from scipy.signal import hilbert


logger = logging.getLogger(__name__)
FORMAT = '%(levelname)s:\t%(threadName)s %(funcName)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def get_args():
    """get cli arguments"""
    parser = argparse.ArgumentParser(description="stdin_python_bridge.py")
    parser.add_argument(
        "--log-level", default="debug", action="store_true", required=False, help="Log level selector"
    )

    return parser.parse_args()


class SCPI_Handler (socketserver.BaseRequestHandler) :
    def handle(self):
        while True :
            self.data = self.request.recv(1024).strip()
            received = self.data.decode()
            if received :
                logger.debug("received %s", received)
                response = responses( received )
            if response :
                logger.info("sendall: %s", response.strip())
                self.request.sendall( response.encode() )
                response = ''


class WAVE_Handler (socketserver.BaseRequestHandler) :
    def handle(self):

        # Works! stdin
        # while True:
        #     data = sys.stdin.buffer.read(100)          
        #     # print("DEBUG: ",data)
        #     print("INFO len:",len(data))

        #     for item in data :
        #         # send = "X:"+item
        #         # print("DEBUG ", hex(item))
        #         to_send = item.to_bytes()
        #         print("DEBUG ", to_send)
        #         # sys.stdout.write(data)
        #         self.request.send( to_send )

        #     # sys.stdin = os.fdopen(sys.stdin.fileno(), 'rb')
        #     # print("DEBUG: ",sys.stdin.read(1024))

        # https://stackoverflow.com/questions/48043004/how-do-i-generate-a-sine-wave-using-python
        fs = 44100 # Sample frequency
        f  =  1000 # Fundamental frequency
        t  =   0.1 # Duration of signal (in seconds)
        samples = np.arange(t * fs) / fs
        signal = np.sin(2 * np.pi * f * samples)
        # Amplify?
        # signal *= 32767
        signal *= 1000
        signal = np.int16(signal)
        signal_iq = hilbert(signal)
        logger.debug("len(signal_iq)): %i", len(signal_iq))
        signal_complex64 = signal_iq.astype(np.complex64)
        logger.debug("len(signal_complex64): %i", len(signal_complex64))
            
        while True :
            if PLAY == True :
                data = to_uint64(DEPTH).tobytes()
                logger.debug("send: DEPTH:\t%s", hex_print(data))
                self.request.send(data)

                sample_hz = 10000
                data = to_uint64(sample_hz).tobytes()
                logger.debug("send: sample_hz:\t%s", hex_print(data))
                self.request.send(data)

                counter = 0
                for item in signal_complex64 :
                    data = item.tobytes()
                    logger.debug("send: %s", hex_print(data))
                    self.request.send(data)
                    counter = counter +1
                    if counter == DEPTH :
                        logger.debug("BREAK: %i", counter)
                        break


def clean_array (array_to_clean) :
    logger.debug("coming in: %s", array_to_clean)
    # Revemove nulls
    step1 = []
    for item in array_to_clean :
        if item :
            step1.append(item)
    # Remove duplicates
    array_clean = list(dict.fromkeys(step1))
    logger.debug("goin out: %s", array_clean)
    return array_clean


def to_uint64 (integer) :
    uint64 = np.uint64(integer)
    return uint64


def responses (data) :
    global PLAY
    response = ''
    logger.info("received: %s", data)
    logger.debug("items in data: %i", len(data.split('\n')))

    commands = clean_array( data.split('\n') )
    for command in commands :
        logger.info("command: %s", command)
        match command:
            case "ping":
                response = "pong\n"
            case "*IDN?":
                # SendReply(GetMake() + "," + GetModel() + "," + GetSerial() + "," + GetFirmwareVersion());
                response = "Ettus Research,B200,ABCD12345678,1.0\n"
                logger.info("command: '*IDN?' answering: %s", response.strip())
            case ":RX:OFFS -0.000000":
                logger.debug("command: %s matched", command)
            case ":RX:RANGE 2.000000":
                logger.debug("command: %s matched", command)
            case "REFCLK internal":
                logger.debug("command: %s matched", command)
            case "RATES?":
                # auto block = rates.substr(istart, i-istart);
                # auto fs = stoll(block);
                # auto hz = FS_PER_SECOND / fs;
                # ret.push_back(hz);
                # it is asking for 1,000,000,000,000 (1 TSample)
                # response = "100000,500000,1000000,10000000,15000000\n"
                # response = "1000,5000,10000,100000,150000\n"
                response = "1000,5000,10000\n"
                logger.info("command: 'RATES?' answering: %s", response.strip())
            case "DEPTHS?":
                # response = "10000,20000,30000,40000,50000,100000\n"
                response = str(DEPTH)+"\n"
                logger.info("command: 'DEPTHS?' answering: %s", response).strip()
            case "DEPTH 100000":
                logger.debug("command: %s matched", command)
            case "START":
                logger.info("command: START Playing!")
                PLAY = True
            case "STOP":
                logger.info("command: STOP Stoping!")
                PLAY = False
            case "RXGAIN 35":
                logger.debug("command: %s matched", command)
            case "RXBW 10000000":
                # Span?
                logger.debug("command: %s matched", command)
            case "RXFREQ 1000000000":
                # Center Frequency
                logger.debug("command: %s matched", command)

            # Missing
            case _:
                logger.error("command: %s not matched!", command)
 
    return response


def signal_handler (sig, frame) :
    # print('Exiting...')
    logger.info("Exiting...")
    server_scpi.server_close()
    server_wave.server_close()
    sys.exit()


def hex_print (bytes_to_convert) :
    result = ''
    bytes_to_hex = bytes_to_convert.hex()
    count = 0
    for symbol in bytes_to_hex :
        result = result + symbol
        count = count +1
        if (count % 2) == 0 :
            result = result + ' '
    result = result + '| '
    for symbol in bytes_to_convert :
        if symbol > 32 and symbol < 127 :
            result = result + chr(symbol)
        else :
            result = result + '.'
    # result = result + ' |' + str(val(bytes_to_convert))
    result = result + ' |'
    return result 



if __name__ == "__main__":

    DEPTH = 100000
    # PLAY = True
    PLAY = False

    args = get_args()

    logger.info("Starting")

    # Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)

    socketserver.TCPServer.allow_reuse_address = True
    server_scpi = socketserver.TCPServer(("", 5025), SCPI_Handler)
    server_wave = socketserver.TCPServer(("", 5026), WAVE_Handler)

    thread1 = threading.Thread(name='SCPI', target=server_scpi.serve_forever)
    thread2 = threading.Thread(name='WAVE', target=server_wave.serve_forever)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
