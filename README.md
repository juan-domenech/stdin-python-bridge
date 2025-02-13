# stdin-python-bridge

TCP/IP server to share waveforms coming from stdin into SDR UHD bridge to be read from ngscopeclient https://www.ngscopeclient.org/

This bridge emulates a [scopehal-uhd-bridge](https://github.com/ngscopeclient/scopehal-uhd-bridge) using SCPI instrumentation commands (port 5025) and a Twinlan TCP socket (port 5026).

▶️ [YouTube tutorial:](https://www.youtube.com/watch?v=sNBDqhYXUR8)

[![ngscopeclient: Injecting audio via stdin and SDR UHD twinlan input](http://img.youtube.com/vi/sNBDqhYXUR8/0.jpg)](http://www.youtube.com/watch?v=sNBDqhYXUR8 "ngscopeclient: Injecting audio via stdin and SDR UHD twinlan input")

# Instructions

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
cat g001_433.92M_250k.cu8 | python3 stdin_python_bridge.py --stdin --samplerate 250000 --encode cu8
```

### --encode float32iq
Two channels (float32 + float32) expected to be in IQ signal format.

(this is the default)

#### Examples with SoX
SoX manual page: https://man.cx/sox(1)

Read a WAV file to create a 2 channels pipe with I and Q components (MacOS):
```
sox -V6 "|sox 1001.float32.mono.wav -p" "|sox 1001.float32.mono.wav -p hilbert -n 511" -r48000 -c2 -efloating-point -t raw - --combine merge | python3 stdin_python_bridge.py --stdin --encode float32iq --samplerate 48000
```

Read from microphone to create a 2 channels pipe with I and Q components (MacOS):
```
sox -V6 "|sox -v10 -t coreaudio 'MacBook Pro Microphone' -r48000 -c1 -p" "|sox -v10 -t coreaudio 'MacBook Pro Microphone' -r48000 -c1 -p hilbert -n 511" -r48000 -c2 -efloating-point -t raw - --combine merge | python3 stdin_python_bridge.py --stdin --encode float32iq --samplerate 48000
```

Read from line-in to create a 2 channels pipe expecting the source to be I/Q already (PC Ubuntu):
```
sox -V6 -48000 -b32 -c2 -t alsa -D hw:CARD=PCH,DEV=0 -b32 -efloating-point -t raw - | python3 stdin_python_bridge.py --stdin --encode float32iq --samplerate 48000 
```

### --encode complex64
Two channels (float32 + float32) expected to be in IQ [complex signal format](https://pysdr.org/content/iq_files.html).

This is the internal format of this script and the way the samples are sent over the wire.

## Optional

### --samplerate
Specify the expected sampling rate of the source signal in samples/second.

### --tonefreq
Frequency of the test tone in Hz.

### --loglevel debug

### --showprogress
(Impacts performance)

Show the complex64 item of the signal in Hexadecimal, the real and imaginary component as float32 and the wave as ASCII.
```
INFO:	WAVE handle send: |cd cc cc 3d 00 00 80 bf|  0.10000000149011612                 -1.0 │─────────────────■──────────────┼■───────────────────────────────│
INFO:	WAVE handle send: |bc e8 aa 3e f4 50 71 bf|   0.3338068723678589  -0.9426414966583252 │                     ■          │■                               │
INFO:	WAVE handle send: |24 1b 21 3f 0a f3 46 bf|   0.6293203830718994  -0.7771459817886353 │                          ■     │   ■                            │
INFO:	WAVE handle send: |a0 46 5a 3f 77 c2 05 bf|   0.8526401519775391  -0.5224985480308533 │                             ■  │       ■                        │
INFO:	WAVE handle send: |e2 67 7a 3f cd e6 54 be|   0.9781476259231567  -0.2079116851091385 │                               ■│            ■                   │
INFO:	WAVE handle send: |55 cf 7d 3f a8 a8 05 3e|   0.9914448857307434  0.13052618503570557 │                               ■│                  ■             │
INFO:	WAVE handle send: |01 19 64 3f 71 71 e8 3e|   0.8910065293312073  0.45399048924446106 │                              ■ │                       ■        │
INFO:	WAVE handle send: |01 38 30 3f 23 b2 39 3f|   0.6883545517921448   0.7253744006156921 │                           ■    │                           ■    │
INFO:	WAVE handle send: |c9 3f d0 3e 1d de 69 3f|   0.4067366421222687   0.9135454297065735 │                      ■         │                              ■ │
INFO:	WAVE handle send: |2a af a0 3d f9 35 7f 3f|  0.07845909893512726   0.9969173073768616 │                 ■              │                               ■│
INFO:	WAVE handle send: |ee 83 84 be ea 46 77 3f|   -0.258819043636322   0.9659258127212524 │───────────■────────────────────┼───────────────────────────────■│
INFO:	WAVE handle send: |00 00 11 bf ef f9 52 3f|          -0.56640625   0.8241261839866638 │      ■                         │                             ■  │
INFO:	WAVE handle send: |bd 1b 4f bf 18 79 16 3f|    -0.80901700258255   0.5877852439880371 │   ■                            │                         ■      │
INFO:	WAVE handle send: |36 75 75 bf 76 6a 91 3e|  -0.9588197469711304  0.28401535749435425 │■                               │                    ■           │
INFO:	WAVE handle send: |2f a6 7f bf 3a 5e 56 bd|  -0.9986295104026794  -0.0523359552025795 │■                               │               ■                │
INFO:	WAVE handle send: |5e 83 6c bf 15 ef c3 be|  -0.9238795042037964  -0.3826834261417389 │ ■                              │         ■                      │
INFO:	WAVE handle send: |bd 3e 3e bf 25 4c 2b bf|  -0.7431448101997375  -0.6691306233406067 │    ■                           │     ■                          │
INFO:	WAVE handle send: |27 4e f4 be 29 fa 60 bf|  -0.4771587550640106  -0.8788171410560608 │        ■                       │ ■                              │
INFO:	WAVE handle send: |5b 30 20 be 25 d9 7c bf| -0.15643446147441864  -0.9876883625984192 │             ■                  │■                               │
INFO:	WAVE handle send: |f3 9b 3a 3e 98 b6 7b bf|  0.18223552405834198  -0.9832549095153809 │──────────────────■─────────────┼■───────────────────────────────│ 

```
