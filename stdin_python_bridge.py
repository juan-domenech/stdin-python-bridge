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


def signal_handler (sig, frame) :
    """ CTRL+C handler """
    logger.info("Exiting...")
    server_scpi.server_close()
    server_wave.server_close()
    sys.exit()


def get_args():
    """ Get CLI arguments """
    parser = argparse.ArgumentParser(description="stdin_python_bridge.py")
    parser.add_argument(
        "--log-level", default="debug", action="store_true", required=False, help="Log level selector"
    )
    parser.add_argument(
        "--sampling", default=44100, required=False, help="Sampling frequency"
    )
    parser.add_argument(
        "--stdin", action="store_true", required=False, help="Expect stdin"
    )
    parser.add_argument(
        "--tone", action="store_true", required=False, help="Generate test tone"
    )
    parser.add_argument(
        "--tonefreq", default=2600, required=False, help="Test tone frequency in Hz"
    )
    parser.add_argument(
        "--toneduration", default=1, required=False, help="Test tone duration in seconds"
    )
    return parser.parse_args()


class SCPI_Handler (socketserver.BaseRequestHandler) :
    """ SCPI TCP port server """
    def handle(self):
        while True :
            self.data = self.request.recv(1024).strip()
            received = self.data.decode()
            if received :
                logger.debug("received %s", repr(received))
                response = scpi_responses( received )
            if response :
                logger.info("sendall: %s", repr(response))
                self.request.sendall( response.encode() )
                response = ''


def scpi_responses (data) :
    global PLAY
    response = ''
    logger.debug("received: %s", repr(data))
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
                logger.info("command: '*IDN?' answering: %s", repr(response))
            case ":RX:OFFS -0.000000":
                logger.debug("command: %s matched", command)
            case ":RX:RANGE 2.000000":
                logger.debug("command: %s matched", command)
            case "REFCLK internal":
                logger.debug("command: %s matched", command)
            case "RATES?":
                """
                ngscopeclient divides Femtosecond in a second (FS_PER_SECOND) by our rates
                https://github.com/ngscopeclient/scopehal/blob/master/scopehal/UHDBridgeSDR.cpp#L306
                1000000000000000 / Desired Samples/s = Rate
                i.e:
                Rate   100000000 =   10 MS/s
                Rate  1000000000 =    1 MS/s
                Rate 22675736961 = 44100 S/s
                """
                response = "22675736961,1000000000,100000000,\n" # Additional coma required
                logger.info("command: 'RATES?' answering: %s", repr(response))
            case "DEPTHS?":
                # response = "10000,20000,30000,40000,50000,100000\n"
                response = str(DEPTH)+",\n"
                # response = "1000,,\n" # Additional coma required
                logger.info("command: 'DEPTHS?' answering: %s", repr(response))
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
            case "RXBW 10000000": # Span
                logger.debug("command: %s matched", command)
            case "RXFREQ 1000000000": # Center frequency
                logger.debug("command: %s matched", command)

            # Missing
            case _:
                logger.error("command: %s not matched!", command)
 
    return response


def clean_array (array_to_clean) :
    """ Remove nulls and duplicates from array """
    logger.debug("coming in: %s", array_to_clean)
    # Remove nulls
    step1 = []
    for item in array_to_clean :
        if item :
            step1.append(item)
    # Remove duplicates
    array_clean = list(dict.fromkeys(step1))
    logger.debug("going out: %s", array_clean)
    return array_clean


class WAVE_Handler (socketserver.BaseRequestHandler) :
    """ Waveform TCP port server """
    def handle(self):

        logger.info("Connection received")

        if args.stdin :
            logger.info("stdin")
            # Works! stdin
            while True:
                data = sys.stdin.buffer.read(8) # 64 complex = 8 bytes

                # TODO handle partial data
                if len(data) == 8 :
                    # logger.debug("stdin data: %s", data)
                    # https://stackoverflow.com/questions/28995937/convert-python-byte-string-to-numpy-int
                    sample_complex64 = np.frombuffer(data, dtype=np.complex64)
                    logger.debug("stdin send: %s", hex_print(sample_complex64.tobytes(), sample_complex64))
                    self.request.send( sample_complex64.tobytes())

                # sys.stdin = os.fdopen(sys.stdin.fileno(), 'rb')
                # print("DEBUG: ",sys.stdin.read(1024))


        elif args.tone :
            logger.info("Rendering tone")
            # https://stackoverflow.com/questions/48043004/how-do-i-generate-a-sine-wave-using-python
            fs = TONE_SAMPLE
            f  = TONE_FREQ
            t  = TONE_DURATION
            samples = np.arange(t * fs) / fs
            signal = np.sin(2 * np.pi * f * samples)
            # Amplify
            signal *= TONE_AMP
            # logger.debug("signal: %s", signal)
            signal_int16 = np.int16(signal)
            # logger.debug("signal_int16: %s", signal_int16)
            # From real to complex
            # https://panoradio-sdr.de/how-to-convert-between-real-and-complex-iq-signals/
            signal_iq = hilbert(signal_int16)
            # logger.debug("signal_iq: %s", signal_iq)
            logger.debug("len(signal_iq)) samples: %i", len(signal_iq))
            # From complex to complex 64 bits (32 bits real + 32 bits imaginary)
            signal_complex64 = signal_iq.astype(np.complex64)
            signal_length = len(signal_complex64)
            # logger.debug("signal_complex64: %s", signal_complex64)
            logger.debug("len(signal_complex64) samples: %i", signal_length)

            signal_position = 0

            # Stay into the handler to keep the stablished connection always open
            while True :
                # Has command PLAY been issued?
                if PLAY == True :
                    logger.info("Sending tone")
                    # data_to_send = to_uint64(DEPTH).tobytes()
                    # logger.debug("send: DEPTH:\t%s", hex_print(data_to_send))
                    # self.request.send(data_to_send)

                    # sample_hz = 10000
                    # data_to_send = to_uint64(sample_hz).tobytes()
                    # logger.debug("send: sample_hz:\t%s", hex_print(data_to_send))
                    # self.request.send(data_to_send)

                    # Mark + number of digits to follow + block length
                    # https://helpfiles.keysight.com/csg/n5106a/commands_for_downloading_waveform_data.htm
                    # data_to_send = ( '#9' + '0'*(9-len(str(DEPTH))) + str(DEPTH) ).encode()
                    # data_to_send = ( '9' + '0'*(9-len(str(DEPTH))) + str(DEPTH) ).encode()
                    # data_to_send = '#9000001000'.encode()
                    data_to_send = '9000001000'.encode()
                    # data_to_send = '41000'.encode()
                    logger.debug("send: Mark + DEPTH: %s", data_to_send)
                    self.request.send(data_to_send)

                    block_position = 0

                    while block_position != DEPTH :

                        sample = signal_complex64[signal_position]
                        data_to_send = sample.tobytes()
                        logger.debug("send: %s", hex_print(data_to_send, sample))
                        self.request.send(data_to_send)
                        block_position = block_position +1

                        # Iterate along the signal and only come back to the begining
                        # when we reach the end (to avoid abrupt transitions between blocks)
                        if signal_position < signal_length -1 :
                            signal_position = signal_position +1
                        else :
                            logger.debug("send: signal_position: reset to zero at %i", signal_position)
                            signal_position = 0

                    # A block is completed (DEPTH)
                    # Time to send another waveform
                    logger.debug("send: Block BREAK at %i", block_position)

        else :
            logger.error("I don't know what to do!")
            sys.exit(1)


# def to_uint64 (integer) :
#     uint64 = np.uint64(integer)
#     return uint64


def hex_print (bytes_to_convert, source=0) :
    """ Build string printing the complex as Hex,
    as numbers and ploting real and imaginary as ASCII """

    if source.real > 32767 or source.real < -32768 :
        logger.error("out of bounds %s %s", source.real, source.imag)
        return

    result = ''
    count = 0
    bytes_to_hex = bytes_to_convert.hex()
    for symbol in bytes_to_hex :
        result = result + symbol
        count = count +1
        if (count % 2) == 0 :
            result = result + ' '

    # TODO remove condition
    if source == 0 :
        result = result + '| '
        for symbol in bytes_to_convert :
            if symbol > 32 and symbol < 127 :
                result = result + chr(symbol)
            else :
                result = result + '.'
        result = result + ' | '
    else :
        result = result + '| '+  number_padding_spaces(source.real) + number_padding_spaces(source.imag) +'|'+ from_number_to_point(source.real, source.imag) +'|'
    return result 


def number_padding_spaces (number) :
    """ Return number tabulated to the right with spaces """
    return ' '*(6-len(str(int(number.item())))) + str(int(number.item())) +' '


def from_number_to_point (source_real, source_imag) :
    """ Plot both real and imaginary signals vertically
        side by side using # symbol """
    global ZERO_UP
    global ZERO_DOWN
    SCREEN = 31

    # Find zero crossing for real part going positive
    fill = ' '
    if source_real >= 0 and ZERO_UP == False :
        fill = '-'
        ZERO_UP = True
        ZERO_DOWN = False
    # and going negative
    if source_real <= 0 and ZERO_DOWN == False :
        fill = '-'
        ZERO_DOWN = True
        ZERO_UP = False

    # Adding amplification range to bring values above 0
    # i.e. source_real: -32767 becomes 0, +32767 becomes 65534
    pos = TONE_AMP + source_real
    # Reducing to fit in the screen
    BAND = TONE_AMP / 16
    pos = int(pos.item() / BAND)
    real_string = fill* pos +'#'+ fill* (SCREEN-pos)

    pos = TONE_AMP + source_imag
    pos = int(pos.item() / BAND)
    imag_string = fill* pos +'#'+ fill* (SCREEN-pos)

    return real_string +'|'+ imag_string


if __name__ == "__main__":

    args = get_args()
    logger.info("Starting")
    if args.tone :
        TONE_FREQ = int(args.tonefreq)
        TONE_SAMPLE = int(args.sampling)
        # TONE_AMP =    32767 # Aplitude [-32768, 32767]
        TONE_DURATION = float(args.toneduration) # Duration of wave in seconds (it will repeat afterwards)
        logger.info("Generating tone of %i Hz at %i sampling rate during %f seconds", TONE_FREQ, TONE_SAMPLE, TONE_DURATION)
    # TODO --stdin shouldn't need this?
    TONE_AMP =    32767 # Aplitude [-32768, 32767]

    DEPTH = 1000
    PLAY = False
    ZERO_UP = False
    ZERO_DOWN = False

    # Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)

    socketserver.TCPServer.allow_reuse_address = True
    server_scpi = socketserver.TCPServer(("", 5025), SCPI_Handler)
    server_wave = socketserver.TCPServer(("", 5026), WAVE_Handler)

    thread1 = threading.Thread(name='SCPI', target=server_scpi.serve_forever)
    thread2 = threading.Thread(name='WAVE', target=server_wave.serve_forever)

    thread1.start()
    thread2.start()

    logger.info("Waiting for connection")

    thread1.join()
    thread2.join()
