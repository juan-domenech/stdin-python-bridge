"""
stdin Python Bridge

TCP/IP server to send waveforms coming from stdin into Twinlan UHD bridge
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
        "--sampling", default=44100, required=False, help="Sampling frequency (default 44100 samples per second )"
    )
    parser.add_argument(
        "--stdin", action="store_true", required=False, help="Expect signal coming from stdin"
    )
    parser.add_argument(
        "--encode", default="complex64", required=False, help="Input stdin signal format. Available: complex64, cu8 (AKA rtl_sdr), uint8iq and float32iq (default complex64)"
    )
    parser.add_argument(
        "--tone", action="store_true", required=False, help="Generate test tone"
    )
    parser.add_argument(
        "--tonefreq", default=2600, required=False, help="Test tone frequency in Hz (default 2600 Hz)"
    )
    parser.add_argument(
        "--toneduration", default=10, required=False, help="Test tone duration in seconds (default 10 seconds)"
    )
    parser.add_argument(
        "--showprogress", action="store_true", required=False, help="Show complex64 in Hex + complex64 in Number and Waveform in ASCII"
    )
    parser.add_argument(
        "--force", action="store_true", required=False, help="Don't wait for the START command"
    )
    parser.add_argument(
        "--loglevel", default="info", required=False, help="Log level selector (default Info)"
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
                logger.debug("sendall: %s", repr(response))
                self.request.sendall( response.encode() )
                response = ''


def scpi_responses (data) :
    global PLAY
    global TONE_AMP
    global TONE_SAMPLING
    global DEPTH
    global EXITING_PAUSE
    global TONE_FREQ
    response = ''
    logger.debug("received: %s", repr(data))
    logger.debug("items in data: %i", len(data.split('\n')))

    commands = clean_array( data.split('\n') )
    for command in commands :
        logger.info("command: %s", command)
        ignore_match = False

        """ SCPI Commands that change some of our variables """
        if command[0:9] == ':RX:RANGE' :
            TONE_AMP = float(command[10:]) /2
            logger.debug("command: :RX:RANGE updating TONE_AMP to %f", TONE_AMP)
            ignore_match = True

        elif command[0:5] == 'RATE ' :
            if args.tone :
                TONE_SAMPLING = int(command[5:])
                logger.debug("command: RATE updating TONE_SAMPLING to %i", TONE_SAMPLING)
                ignore_match = True
            elif args.stdin :
                # Check 'RATES?' command below
                proposed_sampling = int(command[5:])
                if proposed_sampling == TONE_SAMPLING :
                    logger.info("stdin: command: proposed_sampling %i coincides with TONE_SAMPLING. Nothing to do.", proposed_sampling)
                else :
                    logger.warning("stdin: command: %s can not be execute with stdin. Ignoring.", command)
                ignore_match = True

        elif command[0:6] == 'DEPTH ' :
            DEPTH = int(command[6:])
            logger.debug("command: DEPTH updating DEPTH to %i", DEPTH)
            ignore_match = True

        elif command[0:7] == 'RXFREQ ' :
            # Let's piggyback on the Center frequency intended for the SDR receiver to drive our tone generator
            RXFREQ = int(command[7:])
            logger.debug("RXFREQ %i", RXFREQ)
            # The SDR driver by default is going to place the Center frequency at 1GHz which not what we need
            # Cap to a safe level
            if RXFREQ > 10000 :
                TONE_FREQ = 2600
                logger.error("command: RXFREQ %i too high. Bringing it down to default %i", RXFREQ, TONE_FREQ)
            else :
                TONE_FREQ = RXFREQ
                logger.debug("command: RXFREQ updating TONE_FREQ to %i", TONE_FREQ)
            ignore_match = True

        """
        SCPI Commands that mostly read parts of our state.
        In some cases we care (like RATES or DEPTHS)
        but in other cases we ignore (like RXGAIN, RXFREQ, etc.)
        because they have no effect in this emulation.
        """
        match command:
            case "ping":
                response = "pong\n"
            case "*IDN?":
                # SendReply(GetMake() + "," + GetModel() + "," + GetSerial() + "," + GetFirmwareVersion());
                response = "Experimental stdin,Python3,ABCD12345678,0.1\n"
                logger.info("command: '*IDN?' answering: %s", repr(response))
            case ":RX:OFFS -0.000000":
                logger.debug("command: %s matched", command)
            # case ":RX:RANGE 2.000000":
            #     logger.debug("command: %s matched", command)
            case "REFCLK internal":
                logger.debug("command: %s matched", command)
            case "RATES?":
                """
                ngscopeclient divides Femtosecond in a second (FS_PER_SECOND) by our rates
                https://github.com/ngscopeclient/scopehal/blob/master/scopehal/UHDBridgeSDR.cpp#L306

                1000000000000000 / Desired Samples/s = Rate
                i.e:
                Rate   100000000 =    10 MS/s
                Rate  1000000000 =     1 MS/s
                Rate 10000000000 = 100000 S/s
                Rate 20000000000 =  50000 S/s
                Rate 22675736961 =  44100 S/s
                """
                # When generating a tone we can adjust to almost any sampling frequency proposed by ngscopeclient
                if args.tone :
                    response = "22675736961,20000000000,10000000000,\n" # Additional coma required
                    logger.info("command: 'RATES?' answering: %s", repr(response))
                # But when reading from stdin we have only one: the TONE_SAMPLING provided by the user
                elif args.stdin :
                    response = str( int(1000000000000000/TONE_SAMPLING) )+",\n" # Additional coma required
                    logger.info("command: 'RATES?' answering: %s", repr(response))

            case "DEPTHS?":
                response = "500,1000,2000,5000,10000,\n" # Additional coma required
                logger.info("command: 'DEPTHS?' answering: %s", repr(response))
            case "RXGAIN 35":
                logger.debug("command: %s matched", command)
            case "RXBW 10000000": # Span
                logger.debug("command: %s matched", command)
            case "RXFREQ 1000000000": # Center frequency
                logger.debug("command: %s matched", command)
            case "START":
                logger.info("command: START Playing!")
                PLAY = True
                EXITING_PAUSE = True
            case "STOP":
                logger.info("command: STOP Stoping!")
                PLAY = False
            case "SINGLE":
                logger.info("command: SINGLE Stoping!")
                PLAY = False
            case "FORCE":
                logger.info("command: FORCE Stoping!")
                PLAY = False

            # Missing
            case _:
                if not ignore_match :
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

        global EXITING_PAUSE

        # stdin section
        if args.stdin :
            logger.info("stdin active") # TODO check that we are getting something via stdin

            # Stay into the handler to keep the stablished connection always open
            while True:
                # Has command PLAY been issued?
                if PLAY == True :

                    # TODO Implement EXITING_PAUSE like in --tone

                    if ENCODE == 'complex64' :
                        """
                        complex64 : float32 + float32 IQ
                        """
                        logger.info("Sending stdin complex64")
                        send_wave_header(self, DEPTH, TONE_SAMPLING)

                        block_position = 0
                        while block_position != DEPTH :

                            data = sys.stdin.buffer.read(8) # complex64 = 8 bytes

                            # https://stackoverflow.com/questions/28995937/convert-python-byte-string-to-numpy-int
                            sample_complex64 = np.frombuffer(data, dtype=np.complex64)
                            if args.showprogress :
                                logger.info("stdin send: %s", hex_print_ascii(sample_complex64.tobytes(), sample_complex64))
                            self.request.send( sample_complex64.tobytes() )

                            block_position = block_position +1

                        # A block is completed (DEPTH)
                        # Time to send another waveform
                        logger.debug("send: Block BREAK at %i", block_position)

                    elif ENCODE == 'cu8' : # and uint8iq
                        """
                        rtl_sdr cu8 format seems to be: PCM unsigned integer 8 bits IQ (uint8iq)
                        Zero power at x80, max power positive at xFF and max power negative at x00
                        Dividing by 100 to get values to the typical -1/+1 Volts range.
                        .cu8 Tests
                        https://github.com/merbanan/rtl_433_tests/tree/master/tests/directv
                        """
                        logger.debug("Sending stdin cu8")
                        send_wave_header(self, DEPTH, TONE_SAMPLING)

                        block_position = 0
                        while block_position != DEPTH :

                            # TODO Double check that I comes first and Q second
                            data = sys.stdin.buffer.read(2) # I one byte + Q one byte

                            real32 = np.array((data[0] -127) /100, dtype='float32')
                            imag32 = np.array((data[1] -127) /100, dtype='float32')
                            # https://stackoverflow.com/questions/2598734/numpy-creating-a-complex-array-from-2-real-ones
                            sample_complex64 = np.array(real32 + 1j*imag32, dtype='complex64')
                            if args.showprogress :
                                logger.info("stdin send: %s", hex_print_ascii(sample_complex64.tobytes(), sample_complex64))
                            self.request.send( sample_complex64.tobytes() )

                            block_position = block_position +1

                        # A block is completed (DEPTH)
                        # Time to send another waveform
                        logger.debug("send: Block BREAK at %i", block_position)

                    elif ENCODE == 'float32iq' :

                        logger.debug("Sending stdin float32iq")
                        send_wave_header(self, DEPTH, TONE_SAMPLING)

                        block_position = 0
                        while block_position != DEPTH :

                            data = sys.stdin.buffer.read(8) # float32 (4 bytes) x2 channels = 8 bytes

                            # https://stackoverflow.com/questions/28995937/convert-python-byte-string-to-numpy-int
                            data_real = data[0:4]
                            data_imag = data[4:8]
                            sample_float32_real = np.frombuffer(data_real, dtype=np.float32)
                            sample_float32_imag = np.frombuffer(data_imag, dtype=np.float32)
                            sample_complex64 = np.array(sample_float32_real + 1j*sample_float32_imag, dtype='complex64')
                            if args.showprogress :
                                logger.info("stdin send: %s", hex_print_ascii(sample_complex64.tobytes(), sample_complex64))
                            self.request.send( sample_complex64.tobytes() )

                            block_position = block_position +1

                        # A block is completed (DEPTH)
                        # Time to send another waveform
                        logger.debug("send: Block BREAK at %i", block_position)

        # Test tone generator section
        elif args.tone :

            signal_position = 0

            # Stay into the handler to keep the stablished connection always open
            while True :

                # Has PLAY command been issued?
                if PLAY == True :

                    if EXITING_PAUSE == True :
                        logger.debug("PLAY: Exiting pause")
                        EXITING_PAUSE = False
                        # Rendering after pause to give a chance to pick up any change in settings (in case there are any)
                        signal_complex64 = render_tone(TONE_SAMPLING, TONE_FREQ, TONE_DURATION, TONE_AMP)
                        signal_length = len(signal_complex64)
                        logger.debug("len(signal_complex64) samples: %i", signal_length)

                    logger.info("Sending tone")
                    send_wave_header(self, DEPTH, TONE_SAMPLING)

                    block_position = 0
                    while block_position != DEPTH :

                        sample = signal_complex64[signal_position]
                        data_to_send = sample.tobytes()
                        if args.showprogress :
                            logger.info("send: %s", hex_print_ascii(data_to_send, sample))
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


def send_wave_header (self, depth, tone_sampling) :
    """
    https://github.com/ngscopeclient/scopehal-uhd-bridge/blob/main/src/uhdbridge/WaveformServerThread.cpp#L140-L148
    //Send the data out to the client
    //Just the waveform size then the sample data
    uint64_t len = nrx;
    if(!client.SendLooped((uint8_t*)&len, sizeof(len)))
        break;
    if(!client.SendLooped((uint8_t*)&rate, sizeof(len)))
        break;
    if(!client.SendLooped((uint8_t*)&buf[0], nrx * sizeof(complex<float>)))
        break;
    """
    data_to_send = np.uint64(DEPTH).tobytes()
    logger.debug("send: DEPTH:\t%s", hex_print(data_to_send))
    self.request.send(data_to_send)

    data_to_send = np.int64(TONE_SAMPLING).tobytes()
    logger.debug("send: TONE_SAMPLING:\t%s", hex_print(data_to_send))
    self.request.send(data_to_send)


def render_tone (fs, f, t, tone_amp) :
    """
    Render the test tone with the length of the specified duration
    and store the result in a Numpy array.
    """
    logger.info("Rendering tone")
    # https://stackoverflow.com/questions/48043004/how-do-i-generate-a-sine-wave-using-python
    # fs = TONE_SAMPLING
    # f  = TONE_FREQ
    # t  = TONE_DURATION
   
    # Do periods fit nicely in the signal length?
    periods_in_duration = t / (1/f)
    if periods_in_duration != int(periods_in_duration) :
        # Round up the fraction number of periods
        desired_periods = int(periods_in_duration) + bool(periods_in_duration%1)
        # Extend the duration just enough to fit full periods
        t_new = (desired_periods * t) / periods_in_duration
        logger.debug("Duration time adjusted from %f to %f seconds to fit full periods",t,t_new)
        t = t_new

    samples = np.arange(t * fs) / fs
    signal = np.sin(2 * np.pi * f * samples)

    # Amplify
    signal *= tone_amp

    # From real to complex
    # https://panoradio-sdr.de/how-to-convert-between-real-and-complex-iq-signals/
    signal_iq = hilbert(signal)
    logger.debug("len(signal_iq)) samples: %i", len(signal_iq))

    # From complex to complex 64 bits (32 bits real + 32 bits imaginary)
    signal_iq_complex64 = signal_iq.astype(np.complex64)

    # BUG?: The first real is 3.8414194e-27-1j instead of 0 and that is an outlier in the sequence
    # TODO: Forcing the first real to be 0.1 at the begining of the period
    signal_iq_complex64[0] = np.array(0.1 + -1j*1, dtype='complex64')

    return signal_iq_complex64


def hex_print (bytes_to_convert) :
    """ Build string printing the complex as Hex and as numbers """

    result = '|'
    count = 0
    bytes_to_hex = bytes_to_convert.hex()
    for symbol in bytes_to_hex :
        result = result + symbol
        count = count +1
        if (count % 2) == 0 :
            result = result + ' '
    result = result[0:24] # TODO

    result = result + '| '
    for symbol in bytes_to_convert :
        if symbol > 32 and symbol < 127 :
            result = result + chr(symbol)
        else :
            result = result + '.'
    result = result + ' | '

    return result


def hex_print_ascii (bytes_to_convert, source) :
    """
    Build string printing the complex as Hex,
    as numbers and ploting real and imaginary as ASCII
    """

    # TODO
    if source.real > 32767 or source.real < -32768 :
        logger.error("out of bounds: %s %s", source.real, source.imag)
        return

    result = '|'
    count = 0
    bytes_to_hex = bytes_to_convert.hex()
    for symbol in bytes_to_hex :
        result = result + symbol
        count = count +1
        if (count % 2) == 0 :
            result = result + ' '
    result = result[0:24] # TODO

    # Print frequency based on zero crossings
    ascii_point = from_number_to_point(source.real, source.imag)
    # Did we just cross up?
    if ascii_point[1] == '─' and ZERO_UP :
        ascii_frequency = ' '+str(PERIOD_FREQUENCY)+' Hz'
    else :
        ascii_frequency = ''

    return result + '| '+  number_padding_spaces(source.real) + number_padding_spaces(source.imag) +'│'+ ascii_point +'│'+ascii_frequency


def number_padding_spaces (number) :
    """ Return number as string limited and tabulated to the right with spaces """
    number_to_string = str(number.item())[0:20]
    return ' '*(20-len(number_to_string)) + number_to_string +' '


def from_number_to_point (source_real, source_imag) :
    """
    Plot both real and imaginary signals horizontally
    side by side using # symbol

    https://www.lookuptables.com/text/extended-ascii-table
    """
    global ZERO_UP
    global ZERO_DOWN
    global PERIOD
    global PERIOD_RUNNER
    global PERIOD_FREQUENCY

    PERIOD_RUNNER = PERIOD_RUNNER + 1

    SCREEN = 31

    # Find zero crossing for real part going positive...
    fill = ' '
    if source_real >= 0 and ZERO_UP == False :
        fill = '─'
        ZERO_UP = True
        ZERO_DOWN = False
        # Frequency meter
        PERIOD = PERIOD +1
        PERIOD_FREQUENCY = int(TONE_SAMPLING / PERIOD_RUNNER)
        PERIOD_RUNNER = 0
    # and going negative
    if source_real <= 0 and ZERO_DOWN == False :
        fill = '─'
        ZERO_DOWN = True
        ZERO_UP = False

    # Adding amplification range to bring values above 0
    # i.e. source_real: -32767 becomes 0, +32767 becomes 65534
    pos = TONE_AMP + source_real
    # Reducing to fit in the screen
    BAND = TONE_AMP / 16
    pos = int(pos.item() / BAND)
    if pos > SCREEN :
        # Saturation
        pos = SCREEN
        point = '╬'
    elif pos < 0 :
        pos = 0
        point = '╬'
    else :
        point = '■'
    real_string = fill*pos +point+ fill*(SCREEN-pos)

    pos = TONE_AMP + source_imag
    pos = int(pos.item() / BAND)
    if pos > SCREEN :
        # Saturation
        pos = SCREEN
        point = '╬'
    elif pos < 0 :
        pos = 0
        point = '╬'
    else :
        point = '■'
    imag_string = fill*pos +point+ fill*(SCREEN-pos)

    if fill == ' ' :
        return real_string +'│'+ imag_string
    else :
        return real_string +'┼'+ imag_string


if __name__ == "__main__" :

    args = get_args()

    logger = logging.getLogger(__name__)
    FORMAT = '%(levelname)s:\t%(threadName)s %(funcName)s %(message)s'
    if args.loglevel == 'debug' :
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    else :
        logging.basicConfig(level=logging.INFO, format=FORMAT)

    logger.info("Starting")

    if args.tone and args.stdin :
        logger.error("Fatal: --stdin and --tone at the same time are not allowed.")
        sys.exit(1)

    if args.encode == 'complex64' and args.stdin :
        ENCODE = 'complex64'
        logger.info("Decoding stdin in format complex64 (I real float32 + Q imaginary float32)")
    elif ( args.encode == 'cu8' or args.encode == 'uint8iq' ) and args.stdin :
        ENCODE = 'cu8'
        logger.info("Decoding stdin in format %s (8 bit unsigned integer IQ)", args.encode)
    elif args.encode == 'float32iq' and args.stdin :
        ENCODE = 'float32iq'
        logger.info("Decoding stdin in format float32iq (32 bits + 32 bits float IQ)")
    elif args.stdin :
        ENCODE = 'complex64'
        logger.info("No encode for stdin specified. Decoding stdin in format %s", ENCODE)

    if args.tone :
        TONE_FREQ = int(args.tonefreq)
        TONE_SAMPLING = int(args.sampling)
        # TONE_AMP =    32767 # Aplitude [-32768, 32767]
        TONE_DURATION = float(args.toneduration) # Duration of wave in seconds (it will repeat afterwards)
        TONE_AMP = 1
        logger.info("Generating tone of %i Hz at %i sampling rate with a render length of %f seconds", TONE_FREQ, TONE_SAMPLING, TONE_DURATION)
    elif args.stdin :
        TONE_FREQ = 1 # To prevent division by zero below
        TONE_AMP = 2
        TONE_SAMPLING = int(args.sampling)
        logger.info("Reading from stdin sampling at %i sampling rate and %i TONE_AMP in %s mode", TONE_SAMPLING, TONE_AMP, ENCODE)
    else :
        logger.error("Missing mode: --stdin or --tone")
        sys.exit(1)

    if args.force :
        PLAY = True
        EXITING_PAUSE = True
    else :
        PLAY = False
        EXITING_PAUSE = False

    # Default wave length
    # Other sizes offered via SCPI DEPTH and setting controled from UI
    DEPTH = 1000

    # Zero crossing detector signaling
    ZERO_UP = False
    ZERO_DOWN = False

    # ASCII frequency meter
    PERIOD = 0
    PERIOD_RUNNER = TONE_SAMPLING / TONE_FREQ
    PERIOD_FREQUENCY = 0

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

