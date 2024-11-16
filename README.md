# stdin-python-bridge

TCP/IP server to share waveforms coming from stdin into SDR UHD bridge to be read from ngscopeclient https://www.ngscopeclient.org/

This bridge emulates a [scopehal-uhd-bridge](https://github.com/ngscopeclient/scopehal-uhd-bridge) using SCPI instrumentation commands (port 5025) and a Twinlan TCP socket (port 5026).

## --tone mode

Intended to test our setup. The bridge generates a tone (single frequency) that is sent to ngscopeclient. The tone frequency is adjustable. The sampling frequency and amplitude are controled from the ngscopeclient UI.

First start the bridge by:
```
python3 stdin_python_bridge.py --tone
```
And then select SDR instrument in **ngscopeclient**:

Add -> SDR -> Connect... -> Transport: twinlan -> `localhost:5025:5026`

And then click on: Play ▶️

(Substitute `localhost` with your host name or IP if `stdin_python_bridge.py` is running in some other computer)

![stdin-python-bridge-sdr-uhd-ngscopeclient-1.png](/images/stdin-python-bridge-sdr-uhd-ngscopeclient-1.png)

## --stdin mode

In this mode we can pipe files or streams to **stdin-python-bridge** to be read by **ngscopeclient**.

```
cat YOUR_FILE | python3 stdin_python_bridge.py --stdin 
```

Available encode formats:

### --encode cu8 (or uint8iq)
Two channels (8 bits + 8 bits) unsigned intenger expected to be in IQ signal format.

`cu8` is common in [rtl_sdr](https://pysdr.org/content/rtlsdr.html).

This is a test file from **rtl_433** of a [DirecTV remote capture of a ~50 KHz signal](https://github.com/merbanan/rtl_433_tests/blob/master/tests/directv/01/g001_433.92M_250k.cu8).

Example:
```
cat g001_433.92M_250k.cu8 | python3 stdin_python_bridge.py --stdin --sampling 250000 --encode cu8
```

### --encode float32iq
Two channels (float32 + float32) expected to be in IQ signal format.

(this is the default)

Example to read from MacOS microphone using SoX:
```
sox -V6 -t coreaudio "MacBook Pro Microphone" -r48000 -efloating-point -c2 -t raw - | python3 stdin_python_bridge.py --stdin --encode float32iq --sampling 48000
```

### --encode complex64
Two channels (float32 + float32) expected to be in IQ [complex signal format](https://pysdr.org/content/iq_files.html).


## Optional

### --sampling
Specify the expected sampling rate of the source signal in samples/second.

### --tonefreq
Frequency of the test tone in Hz.

### --loglevel debug

### --showprogress
(Impacts performance)

Show the complex64 item of the signal in Hexadecimal (like it goes through the wire), the real and imaginary component as float32 and the wave as ASCII.
```
INFO:	WAVE handle send: |cd cc cc 3d 00 00 80 bf|  0.10000000149011612                 -1.0 |-----------------#--------------+#-------------------------------|
INFO:	WAVE handle send: |15 5b b9 3e a3 a2 6e bf|   0.3620230257511139  -0.9321691393852234 |                     #          | #                              |
INFO:	WAVE handle send: |70 c8 2c 3f 9d e5 3c bf|   0.6749334335327148  -0.7378786206245422 |                          #     |    #                           |
INFO:	WAVE handle send: |af 72 65 3f a1 10 e3 be|   0.8962811827659607  -0.4434862434864044 |                              # |        #                       |
INFO:	WAVE handle send: |57 fc 7e 3f d0 20 b6 bd|   0.9960379004478455 -0.08892977237701416 |                               #|              #                 |
INFO:	WAVE handle send: |7f ee 75 3f 86 2d 8e 3e|   0.9606704115867615   0.2776910662651062 |                               #|                    #           |
INFO:	WAVE handle send: |99 83 4b 3f c0 4c 1b 3f|   0.7949767708778381   0.6066398620605469 |                            #   |                         #      |
INFO:	WAVE handle send: |c7 7c 05 3f 45 71 5a 3f|   0.5214352011680603    0.853290855884552 |                        #       |                             #  |
INFO:	WAVE handle send: |15 68 35 3e 6b f3 7b 3f|  0.17715485394001007   0.9841830134391785 |                  #             |                               #|
INFO:	WAVE handle send: |16 bf 43 be 76 47 7b 3f| -0.19115862250328064    0.981559157371521 |------------#-------------------+-------------------------------#|
INFO:	WAVE handle send: |06 96 08 bf bb 84 58 3f|  -0.5335391759872437    0.845775306224823 |       #                        |                             #  |
INFO:	WAVE handle send: |ba b4 4d bf 73 62 18 3f|  -0.8035389184951782   0.5952522158622742 |   #                            |                         #      |
INFO:	WAVE handle send: |63 eb 76 bf e1 27 87 3e|  -0.9645292162895203    0.263976126909256 |#                               |                    #           |
INFO:	WAVE handle send: |ae a2 7e bf 12 2c d3 bd|  -0.9946697950363159  -0.1031114012002945 |#                               |              #                 |
INFO:	WAVE handle send: |a4 ce 63 bf 71 94 e9 be|  -0.8898718357086182  -0.4562106430530548 | #                              |        #                       |
INFO:	WAVE handle send: |fe 12 2a bf e4 56 3f bf|   -0.664352297782898  -0.7474195957183838 |     #                          |    #                           |
INFO:	WAVE handle send: |8a 89 b2 be 75 ee 6f bf| -0.34870558977127075  -0.9372323155403137 |          #                     | #                              |
INFO:	WAVE handle send: |af 6c 69 3c 59 f9 7f bf| 0.014247103594243526  -0.9998984932899475 |----------------#---------------+#-------------------------------|
```
