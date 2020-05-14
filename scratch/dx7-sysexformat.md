Sysex Documentation 
===================

(Message GUS:472)
Received: from mailhub.iastate.edu by po-3.iastate.edu 
	id AA06806; Sat, 25 Sep 93 16:13:53 -0500
Received: from Waisman.Wisc.EDU (don.waisman.wisc.edu) by mailhub.iastate.edu
	id AA23002; Sat, 25 Sep 1993 16:14:09 -0500
Received: from Waisman.Wisc.EDU by Waisman.Wisc.EDU (PMDF V4.2-10 #2484) id
 <01H3DDLUXLDSBMA3H1@Waisman.Wisc.EDU>; Sat, 25 Sep 1993 16:13:40 CDT
Date: Sat, 25 Sep 1993 16:13:40 -0500 (CDT)
From: "Ewan A. Macpherson" <MACPHERSON@waisman.wisc.edu>
Subject: DX7 Data Format
To: xeno@iastate.edu
Message-Id: <01H3DDLUY4O2BMA3H1@Waisman.Wisc.EDU>
Organization: Waisman Center, University of Wisconsin-Madison
X-Vms-To: IN::"xeno@iastate.edu"
Mime-Version: 1.0
Content-Type: TEXT/PLAIN; CHARSET=US-ASCII
Content-Transfer-Encoding: 7BIT

Gary:

I don't know anything about the differences between the DX7 and DX7s, but this
DX7 info may be useful.  I posted this to r.m.s. before xmas.

I've seen many requests for public domain / shareware DX editors, but I've
never seen a definitive reply.  They're usually along the lines of "I was
roaching around on CompuServe last month, and I think I remember seeing one..."

Anyway, hope this helps ... 

=========================================================================

For those interested in unpacking the uscd.edu DX7 patch data, here is
DX7 data format information.

     compiled from - the DX7 MIDI Data Format Sheet
                   - article by Steve DeFuria (Keyboard Jan 87)
                   - looking at what my DX7 spits out

I have kept the kinda weird notation used in the DX7 Data Sheet to reduce
typing errors. Where it doesn't quite make sense to me I've added comments.
(And I will not be liable for errors etc ....)

Contents: A: SYSEX Message: Bulk Data for 1 Voice
          B: SYSEX Message: Bulk Data for 32 Voices
          C: SYSEX Message: Parameter Change
          D: Data Structure: Single Voice Dump & Voice Parameter #'s
          E: Function Parameter #'s
          F: Data Structure: Bulk Dump Packed Format

////////////////////////////////////////////////////////////
A:
SYSEX Message: Bulk Data for 1 Voice
------------------------------------
       bits    hex  description

     11110000  F0   Status byte - start sysex
     0iiiiiii  43   ID # (i=67; Yamaha)
     0sssnnnn  00   Sub-status (s=0) & channel number (n=0; ch 1)
     0fffffff  00   format number (f=0; 1 voice)
     0bbbbbbb  01   byte count MS byte
     0bbbbbbb  1B   byte count LS byte (b=155; 1 voice)
     0ddddddd  **   data byte 1

        |       |       |

     0ddddddd  **   data byte 155
     0eeeeeee  **   checksum (masked 2's complement of sum of 155 bytes)
     11110111  F7   Status - end sysex



///////////////////////////////////////////////////////////
B:
SYSEX Message: Bulk Data for 32 Voices
--------------------------------------
       bits    hex  description

     11110000  F0   Status byte - start sysex
     0iiiiiii  43   ID # (i=67; Yamaha)
     0sssnnnn  00   Sub-status (s=0) & channel number (n=0; ch 1)
     0fffffff  09   format number (f=9; 32 voices)
     0bbbbbbb  20   byte count MS byte
     0bbbbbbb  00   byte count LS byte (b=4096; 32 voices)
     0ddddddd  **   data byte 1

        |       |       |

     0ddddddd  **   data byte 4096  (there are 128 bytes / voice)
     0eeeeeee  **   checksum (masked 2's comp. of sum of 4096 bytes)
     11110111  F7   Status - end sysex


/////////////////////////////////////////////////////////////
C:
SYSEX MESSAGE: Parameter Change
-------------------------------
       bits    hex  description

     11110000  F0   Status byte - start sysex
     0iiiiiii  43   ID # (i=67; Yamaha)
     0sssnnnn  10   Sub-status (s=1) & channel number (n=0; ch 1)
     0gggggpp  **   parameter group # (g=0; voice, g=2; function)
     0ppppppp  **   parameter # (these are listed in next section)
                     Note that voice parameter #'s can go over 128 so
                     the pp bits in the group byte are either 00 for
                     par# 0-127 or 01 for par# 128-155. In the latter case
                     you add 128 to the 0ppppppp byte to compute par#. 
     0ddddddd  **   data byte
     11110111  F7   Status - end sysex


//////////////////////////////////////////////////////////////

D:
Data Structure: Single Voice Dump & Parameter #'s (single voice format, g=0)
-------------------------------------------------------------------------

Parameter
 Number    Parameter                  Value Range
---------  ---------                  -----------
  0        OP6 EG rate 1              0-99
  1         "  "  rate 2               "
  2         "  "  rate 3               "
  3         "  "  rate 4               "
  4         "  " level 1               "
  5         "  " level 2               "
  6         "  " level 3               "
  7         "  " level 4               "
  8        OP6 KBD LEV SCL BRK PT      "        C3= $27
  9         "   "   "   "  LFT DEPTH   "
 10         "   "   "   "  RHT DEPTH   "
 11         "   "   "   "  LFT CURVE  0-3       0=-LIN, -EXP, +EXP, +LIN
 12         "   "   "   "  RHT CURVE   "            "    "    "    "  
 13        OP6 KBD RATE SCALING       0-7
 14        OP6 AMP MOD SENSITIVITY    0-3
 15        OP6 KEY VEL SENSITIVITY    0-7
 16        OP6 OPERATOR OUTPUT LEVEL  0-99
 17        OP6 OSC MODE (fixed/ratio) 0-1        0=ratio
 18        OP6 OSC FREQ COARSE        0-31
 19        OP6 OSC FREQ FINE          0-99
 20        OP6 OSC DETUNE             0-14       0: det=-7
 21 \
  |  > repeat above for OSC 5, OSC 4,  ... OSC 1
125 /
126        PITCH EG RATE 1            0-99
127          "    " RATE 2              "
128          "    " RATE 3              "
129          "    " RATE 4              "
130          "    " LEVEL 1             "
131          "    " LEVEL 2             "
132          "    " LEVEL 3             "
133          "    " LEVEL 4             "
134        ALGORITHM #                 0-31
135        FEEDBACK                    0-7
136        OSCILLATOR SYNC             0-1
137        LFO SPEED                   0-99
138         "  DELAY                    "
139         "  PITCH MOD DEPTH          "
140         "  AMP   MOD DEPTH          "
141        LFO SYNC                    0-1
142         "  WAVEFORM                0-5, (data sheet claims 9-4 ?!?)
                                       0:TR, 1:SD, 2:SU, 3:SQ, 4:SI, 5:SH
143        PITCH MOD SENSITIVITY       0-7
144        TRANSPOSE                   0-48   12 = C2
145        VOICE NAME CHAR 1           ASCII
146        VOICE NAME CHAR 2           ASCII
147        VOICE NAME CHAR 3           ASCII
148        VOICE NAME CHAR 4           ASCII
149        VOICE NAME CHAR 5           ASCII
150        VOICE NAME CHAR 6           ASCII
151        VOICE NAME CHAR 7           ASCII
152        VOICE NAME CHAR 8           ASCII
153        VOICE NAME CHAR 9           ASCII
154        VOICE NAME CHAR 10          ASCII
155        OPERATOR ON/OFF
              bit6 = 0 / bit 5: OP1 / ... / bit 0: OP6

Note that there are actually 156 parameters listed here, one more than in 
a single voice dump. The OPERATOR ON/OFF parameter is not stored with the
voice, and is only transmitted or received while editing a voice. So it
only shows up in parameter change SYS-EX's.


////////////////////////////////////////////////////////

E:
Function Parameters: (g=2)
-------------------------

Parameter
Number        Parameter           Range
---------     ----------          ------
64         MONO/POLY MODE CHANGE  0-1      O=POLY
65         PITCH BEND RANGE       0-12
66           "    "   STEP        0-12
67         PORTAMENTO MODE        0-1      0=RETAIN 1=FOLLOW
68              "     GLISS       0-1
69              "     TIME        0-99
70         MOD WHEEL RANGE        0-99
71          "    "   ASSIGN       0-7     b0: pitch,  b1:amp, b2: EG bias
72         FOOT CONTROL RANGE     0-99
73          "     "     ASSIGN    0-7           "
74         BREATH CONT RANGE      0-99
75           "     "   ASSIGN     0-7           "
76         AFTERTOUCH RANGE       0-99
77             "      ASSIGN      0-7           "

///////////////////////////////////////////////////////////////

F:
Data Structure: Bulk Dump Packed Format
---------------------------------------

OK, now the tricky bit. For a bulk dump the 155 voice parameters for each
 voice are packed into 32 consecutive 128 byte chunks as follows ...

byte             bit #
 #     6   5   4   3   2   1   0   param A       range  param B       range
----  --- --- --- --- --- --- ---  ------------  -----  ------------  -----
  0                R1              OP6 EG R1      0-99
  1                R2              OP6 EG R2      0-99
  2                R3              OP6 EG R3      0-99
  3                R4              OP6 EG R4      0-99
  4                L1              OP6 EG L1      0-99
  5                L2              OP6 EG L2      0-99
  6                L3              OP6 EG L3      0-99
  7                L4              OP6 EG L4      0-99
  8                BP              LEV SCL BRK PT 0-99
  9                LD              SCL LEFT DEPTH 0-99
 10                RD              SCL RGHT DEPTH 0-99
 11    0   0   0 |  RC   |   LC  | SCL LEFT CURVE 0-3   SCL RGHT CURVE 0-3
 12  |      DET      |     RS    | OSC DETUNE     0-14  OSC RATE SCALE 0-7
 13    0   0 |    KVS    |  AMS  | KEY VEL SENS   0-7   AMP MOD SENS   0-3
 14                OL              OP6 OUTPUT LEV 0-99
 15    0 |         FC        | M | FREQ COARSE    0-31  OSC MODE       0-1
 16                FF              FREQ FINE      0-99
 17 \
  |  > these 17 bytes for OSC 5
 33 /
 34 \
  |  > these 17 bytes for OSC 4
 50 /
 51 \
  |  > these 17 bytes for OSC 3
 67 /
 68 \
  |  > these 17 bytes for OSC 2
 84 /
 85 \
  |  > these 17 bytes for OSC 1
101 /

byte             bit #
 #     6   5   4   3   2   1   0   param A       range  param B       range
----  --- --- --- --- --- --- ---  ------------  -----  ------------  -----
102               PR1              PITCH EG R1   0-99
103               PR2              PITCH EG R2   0-99
104               PR3              PITCH EG R3   0-99
105               PR4              PITCH EG R4   0-99
106               PL1              PITCH EG L1   0-99
107               PL2              PITCH EG L2   0-99
108               PL3              PITCH EG L3   0-99
109               PL4              PITCH EG L4   0-99
110    0   0 |        ALG        | ALGORITHM     0-31
111    0   0   0 |OKS|    FB     | OSC KEY SYNC  0-1    FEEDBACK      0-7
112               LFS              LFO SPEED     0-99
113               LFD              LFO DELAY     0-99
114               LPMD             LF PT MOD DEP 0-99
115               LAMD             LF AM MOD DEP 0-99
116  |  LPMS |      LFW      |LKS| LF PT MOD SNS 0-7   WAVE 0-5,  SYNC 0-1
117              TRNSP             TRANSPOSE     0-48
118          NAME CHAR 1           VOICE NAME 1  ASCII
119          NAME CHAR 2           VOICE NAME 2  ASCII
120          NAME CHAR 3           VOICE NAME 3  ASCII
121          NAME CHAR 4           VOICE NAME 4  ASCII
122          NAME CHAR 5           VOICE NAME 5  ASCII
123          NAME CHAR 6           VOICE NAME 6  ASCII
124          NAME CHAR 7           VOICE NAME 7  ASCII
125          NAME CHAR 8           VOICE NAME 8  ASCII
126          NAME CHAR 9           VOICE NAME 9  ASCII
127          NAME CHAR 10          VOICE NAME 10 ASCII

/////////////////////////////////////////////////////////////////////

And that's it.

Hope this is useful.

ewan.



  0                R1              OP6 EG R1      0-99
  1                R2              OP6 EG R2      0-99
  2                R3              OP6 EG R3      0-99
  3                R4              OP6 EG R4      0-99
  4                L1              OP6 EG L1      0-99
  5                L2              OP6 EG L2      0-99
  6                L3              OP6 EG L3      0-99
  7                L4              OP6 EG L4      0-99
  8                BP              LEV SCL BRK PT 0-99
  9                LD              SCL LEFT DEPTH 0-99
 10                RD              SCL RGHT DEPTH 0-99
 11    0   0   0 |  RC   |   LC  | SCL LEFT CURVE 0-3   SCL RGHT CURVE 0-3
 12  |      DET      |     RS    | OSC DETUNE     0-14  OSC RATE SCALE 0-7
 13    0   0 |    KVS    |  AMS  | KEY VEL SENS   0-7   AMP MOD SENS   0-3
 14                OL              OP6 OUTPUT LEV 0-99
 15    0 |         FC        | M | FREQ COARSE    0-31  OSC MODE       0-1
 16                FF              FREQ FINE      0-99
 
102               PR1              PITCH EG R1   0-99
103               PR2              PITCH EG R2   0-99
104               PR3              PITCH EG R3   0-99
105               PR4              PITCH EG R4   0-99
106               PL1              PITCH EG L1   0-99
107               PL2              PITCH EG L2   0-99
108               PL3              PITCH EG L3   0-99
109               PL4              PITCH EG L4   0-99
110    0   0 |        ALG        | ALGORITHM     0-31
111    0   0   0 |OKS|    FB     | OSC KEY SYNC  0-1    FEEDBACK      0-7
112               LFS              LFO SPEED     0-99
113               LFD              LFO DELAY     0-99
114               LPMD             LF PT MOD DEP 0-99
115               LAMD             LF AM MOD DEP 0-99
116  |  LPMS |      LFW      |LKS| LF PT MOD SNS 0-7   WAVE 0-5,  SYNC 0-1
117              TRNSP             TRANSPOSE     0-48
118          NAME CHAR 1           VOICE NAME 1  ASCII
119          NAME CHAR 2           VOICE NAME 2  ASCII
120          NAME CHAR 3           VOICE NAME 3  ASCII
121          NAME CHAR 4           VOICE NAME 4  ASCII
122          NAME CHAR 5           VOICE NAME 5  ASCII
123          NAME CHAR 6           VOICE NAME 6  ASCII
124          NAME CHAR 7           VOICE NAME 7  ASCII
125          NAME CHAR 8           VOICE NAME 8  ASCII
126          NAME CHAR 9           VOICE NAME 9  ASCII
127          NAME CHAR 10          VOICE NAME 10 ASCII