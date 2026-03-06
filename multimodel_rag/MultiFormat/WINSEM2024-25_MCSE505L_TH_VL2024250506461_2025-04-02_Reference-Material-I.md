Computer Networking: A Top-Down Approach  8th edition  Jim Kurose, Keith Ross Pearson, 2020

Chapter 7 Wireless and

Mobile Networks

A note on the use of these PowerPoint slides:

We’re making these slides freely available to all (faculty, students, readers). They’re in PowerPoint form so you see the animations; and can add, modify, and delete slides  (including this one) and slide content to suit your needs. They obviously represent a lot of work on our part. In return for use, we only ask the following:

- If you use these slides (e.g., in a class) that you mention their source (after all, we’d like people to use our book!)
- If you post any slides on a www site, that you note that they are adapted from (or perhaps identical to) our slides, and note our copyright of this material.

For a revision history, see the slide note for this page. 

Thanks and enjoy!  JFK/KWR

     All material copyright 1996-2020

     J.F Kurose and K.W. Ross, All Rights Reserved

<!-- image -->

# Wireless and Mobile Networks: context

Wireless and Mobile Networks: 7-2

- more wireless (mobile) phone subscribers than fixed (wired) phone subscribers (10-to-1 in 2019)!
- more mobile-broadband-connected devices than fixed-broadband-connected devices devices (5-1 in 2019)!
- 4G/5G cellular networks now embracing Internet protocol stack, including SDN
- two important (but different) challenges
- wireless: communication over wireless link
- mobility: handling the mobile user who changes point of attachment to network

# Chapter 7 outline

<!-- image -->

- Introduction

Wireless

- Wireless Links and network characteristics 
- WiFi: 802.11 wireless LANs
- Cellular networks: 4G and 5G

Mobility

- Mobility management: principles
- Mobility management: practice
- 4G/5G networks
- Mobile IP
- Mobility: impact on higher-layer protocols

Wireless and Mobile Networks: 7- 3

# Elements of a wireless network

Wireless and Mobile Networks: 7- 4

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

wired network 

infrastructure

# Elements of a wireless network

Wireless and Mobile Networks: 7- 5

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

wired network 

infrastructure

wireless hosts

- laptop, smartphone, IoT
- run applications
- may be stationary (non-mobile) or mobile
- wireless does not always mean mobility! 

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# Elements of a wireless network

Wireless and Mobile Networks: 7- 6

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

wired network 

infrastructure

 base station

- typically connected to wired network
- relay - responsible for sending packets between wired network and wireless host(s) in its “area”
- e.g., cell towers,  802.11 access points 

<!-- image -->

<!-- image -->

<!-- image -->

# Elements of a wireless network

Wireless and Mobile Networks: 7- 7

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

wired network 

infrastructure

 wireless link

- typically used to connect mobile(s) to base station, also used as backbone link 
- multiple access protocol coordinates link access 
- various transmission rates and distances, frequency bands

# Characteristics of selected wireless links

Wireless and Mobile Networks: 7- 8

Indoor

Outdoor

Midrange

outdoor

Long range

outdoor

10-30m

50-200m

200m-4Km

4Km-15Km

2 Mbps

4G LTE

802.11ac

802.11n

802.11g

802.11b

3.5 Gbps

600 Mbps

54 Mbps

11 Mbps

Bluetooth

802.11ax

14 Gbps

5G

10 Gbps

802.11 af,ah

# Elements of a wireless network

Wireless and Mobile Networks: 7- 9

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

wired network 

infrastructure

 infrastructure mode

- base station connects mobiles into wired network
- handoff: mobile changes base station providing connection into wired network

# Elements of a wireless network

Wireless and Mobile Networks: 7- 10

ad hoc mode

- no base stations
- nodes can only transmit to other nodes within link coverage
- nodes organize themselves into a network: route among themselves

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# Wireless network taxonomy

Wireless and Mobile Networks: 7- 11

single hop

multiple hops

infrastructure

(e.g., APs)

no

infrastructure

host connects to  base station (WiFi, cellular) which connects to  larger Internet

no base station, no connection to larger  Internet (Bluetooth, ad hoc nets)

host may have to relay through several wireless nodes to connect to larger 

Internet: mesh net

no base station, no connection to larger  Internet. May have to relay to reach other  a given wireless node MANET, VANET

# Chapter 7 outline

<!-- image -->

- Introduction

Link Layer: 6-12

Wireless

- Wireless links and network characteristics 
- WiFi: 802.11 wireless LANs
- Cellular networks: 4G and 5G

Mobility

- Mobility management: principles
- Mobility management: practice
- 4G/5G networks
- Mobile IP
- Mobility: impact on higher-layer protocols

# Wireless link characteristics (1)

Wireless and Mobile Networks: 7- 13

important differences from wired link ….

- decreased signal strength: radio signal attenuates as it propagates through matter (path loss)
- interference from other sources: wireless network frequencies (e.g., 2.4 GHz) shared by many devices (e.g., WiFi, cellular, motors): interference 
- multipath propagation: radio signal reflects off objects ground, arriving at destination at slightly different times

…. make communication across (even a point to point) wireless link much more “difficult” 

- 

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# Wireless link characteristics (2)

Wireless and Mobile Networks: 7- 14

- SNR: signal-to-noise ratio
- larger SNR – easier to extract signal from noise (a “good thing”)
- SNR versus BER tradeoffs
- given physical layer: increase power -&gt; increase SNR-&gt;decrease BER
- given SNR: choose physical layer that meets BER requirement, giving highest throughput
- SNR may change with mobility: dynamically adapt physical layer (modulation technique, rate) 
- 

10

20

30

40

QAM256 (8 Mbps)

QAM16 (4 Mbps)

BPSK (1 Mbps)

SNR(dB)

BER

10-1

10-2

10-3

10-5

10-6

10-7

10-4

# Wireless link characteristics (3)

Wireless and Mobile Networks: 7- 15

<!-- image -->

<!-- image -->

Multiple wireless senders, receivers create additional problems (beyond multiple access):

A

B

C

Hidden terminal problem

- B, A hear each other
- B, C hear each other
- A, C can not hear each other means A, C unaware of their interference at B
- 

A

B

C

A’s signal

strength

space

C’s signal

strength

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Signal attenuation:

- B, A hear each other
- B, C hear each other
- A, C can not hear each other interfering at B

# Code Division Multiple Access (CDMA)

Wireless and Mobile Networks: 7- 16

- unique “code” assigned to each user; i.e., code set partitioning
- all users share same frequency, but each user has own “chipping” sequence (i.e., code) to encode data
- allows multiple users to “coexist” and transmit simultaneously with minimal interference (if codes are “orthogonal”)
- encoding: inner product: (original data) X (chipping sequence)
- decoding: summed inner-product: (encoded data) X (chipping sequence)
- 
- 

# Chapter 7 outline

<!-- image -->

- Introduction

Link Layer: 6-17

Wireless

- Wireless links and network characteristics 
- WiFi: 802.11 wireless LANs
- Cellular networks: 4G and 5G

Mobility

- Mobility management: principles
- Mobility management: practice
- 4G/5G networks
- Mobile IP
- Mobility: impact on higher-layer protocols

# IEEE 802.11 Wireless LAN

Wireless and Mobile Networks: 7- 18

| IEEE 802.11 standard   | Year        | Max data rate   | Range   | Frequency                    |
|------------------------|-------------|-----------------|---------|------------------------------|
| 802.11b                | 1999        | 11 Mbps         | 30 m    | 2.4 Ghz                      |
| 802.11g                | 2003        | 54 Mbps         | 30m     | 2.4 Ghz                      |
| 802.11n  (WiFi 4)      | 2009        | 600             | 70m     | 2.4, 5 Ghz                   |
| 802.11ac (WiFi 5)      | 2013        | 3.47Gpbs        | 70m     | 5 Ghz                        |
| 802.11ax (WiFi 6)      | 2020 (exp.) | 14 Gbps         | 70m     | 2.4, 5 Ghz                   |
| 802.11af               | 2014        | 35 – 560 Mbps   | 1 Km    | unused TV bands (54-790 MHz) |
| 802.11ah               | 2017        | 347Mbps         | 1 Km    | 900 Mhz                      |

- all use CSMA/CA for multiple access, and have base-station and ad-hoc network versions

# 802.11 LAN architecture

Wireless and Mobile Networks: 7- 19

- wireless host communicates with base station
- base station = access point (AP)
- Basic Service Set (BSS) (aka “cell”) in infrastructure mode contains:
- wireless hosts
- access point (AP): base station
- ad hoc mode: hosts only

BSS 1

BSS 2

Internet

switch

 or router

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# 802.11: Channels, association

Wireless and Mobile Networks: 7- 20

- spectrum divided into channels at different frequencies
- AP admin chooses frequency for AP
- interference possible: channel can be same as that chosen by neighboring AP!

- arriving host: must associate with an AP
- scans channels, listening for beacon frames containing AP’s name (SSID) and MAC address
- selects AP to associate with
- then may perform authentication
- then typically run DHCP to get IP address in AP’s subnet
- 

BSS

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# 802.11: passive/active scanning

Wireless and Mobile Networks: 7- 21

AP 2

AP 1

H1

BBS 2

BBS 1

1

2

3

1

passive scanning: 

1. beacon frames sent from APs
2. association Request frame sent: H1 to selected AP 
3. association Response frame sent from  selected AP to H1

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

AP 2

AP 1

H1

BBS 2

BBS 1

1

2

2

3

4

active  scanning: 

1. Probe Request frame broadcast from H1
2. Probe Response frames sent from APs
3. Association Request frame sent: H1 to selected AP 
4. Association Response frame sent from selected AP to H1

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# IEEE 802.11: multiple access

Wireless and Mobile Networks: 7- 22

- avoid collisions: 2+ nodes transmitting at same time
- 802.11: CSMA - sense before transmitting
- don’t collide with detected ongoing transmission by another node
- 802.11: no collision detection!
- difficult to sense collisions: high transmitting signal, weak received signal due to fading
- can’t sense all collisions in any case: hidden terminal, fading
- goal: avoid collisions: CSMA/CollisionAvoidance

space

<!-- image -->

<!-- image -->

A

B

C

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

A

B

C

A’s signal

strength

C’s signal

strength

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

# IEEE 802.11 MAC Protocol: CSMA/CA

Wireless and Mobile Networks: 7- 23

802.11 sender

1 if sense channel idle for DIFS  then 

transmit entire frame (no CD)

sender

receiver

DIFS

data

SIFS

ACK

802.11 receiver

 if frame received OK

   return ACK after SIFS (ACK needed due to hidden terminal problem) 

2 if sense channel busy then 

start random backoff time

timer counts down while channel idle

transmit when timer expires

if no ACK, increase random backoff interval, repeat 2

# Avoiding collisions (more)

Wireless and Mobile Networks: 7- 24

idea: sender “reserves” channel use for data frames using small reservation packets

- sender first transmits small request-to-send (RTS) packet to BS using CSMA
- RTSs may still collide with each other (but they’re short)
- BS broadcasts clear-to-send CTS in response to RTS
- CTS heard by all nodes
- sender transmits data frame
- other stations defer transmissions 

# Collision Avoidance: RTS-CTS exchange

Wireless and Mobile Networks: 7- 25

AP

A

B

RTS(A)

RTS(B)

RTS(A)

CTS(A)

CTS(A)

DATA (A)

ACK(A)

ACK(A)

reservation collision

defer

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

time

# 802.11: mobility within same subnet

Wireless and Mobile Networks: 7- 26

- H1 remains in same IP subnet: IP address can remain same

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

H1

BBS 2

BBS 1

- switch: which AP is associated with H1?

- self-learning : switch will see frame from H1 and “remember” which switch port can be used to reach H1

# 802.11: advanced capabilities

Wireless and Mobile Networks: 7- 27

Rate adaptation

- base station, mobile dynamically change transmission rate (physical layer modulation technique) as mobile moves, SNR varies 

10

20

30

40

SNR(dB)

BER

10-1

10-2

10-3

10-5

10-6

10-7

10-4

QAM256 (8 Mbps)

QAM16 (4 Mbps)

BPSK (1 Mbps)

operating point

1. SNR decreases, BER increase as node moves away from base station

2. When BER becomes too high, switch to lower transmission rate but with lower BER

# 802.11: advanced capabilities

Wireless and Mobile Networks: 7- 28

power management

- node-to-AP: “I am going to sleep until next beacon frame”
- AP knows not to transmit frames to this node
- node wakes up before next beacon frame
- beacon frame: contains list of mobiles with AP-to-mobile frames waiting to be sent
- node will stay awake if AP-to-mobile frames to be sent; otherwise sleep again until next beacon frame

# Personal area networks: Bluetooth

Wireless and Mobile Networks: 7- 29

- less than 10 m diameter
- replacement for cables (mouse, keyboard, headphones)
- ad hoc: no infrastructure
- 2.4-2.5 GHz ISM radio band, up to 3 Mbps
- master controller / clients devices:
- master polls clients, grants requests for client transmissions

radius of

coverage

C

C

C

P

P

P

P

M

C

master device

client device

parked device (inactive)

P

M

# Personal area networks: Bluetooth

Wireless and Mobile Networks: 7- 30

- TDM, 625 msec sec. slot
- FDM: sender uses 79 frequency channels in known, pseudo-random order slot-to-slot (spread spectrum)
- other devices/equipment not in piconet only interfere in some slots
- parked mode: clients can “go to sleep” (park) and later wakeup (to preserve battery)
- bootstrapping: nodes self-assemble (plug and play) into piconet

radius of

coverage

C

C

C

P

P

P

P

M

C

master device

client device

parked device (inactive)

P

M