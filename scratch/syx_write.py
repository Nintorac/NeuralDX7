#%%
import mido
from pathlib import Path
import json
from uuid import uuid4
from itertools import chain
from tqdm import tqdm as tqdm
import numpy as np

from syx_parser import PARAMETER_ORDER, ARTIFACTS_ROOT

# # %%
# presets = [mido.read_syx_file(patch.as_posix()) for patch in iter(dexed_presets)]

N_OSC = 6
N_VOICE = 32
# # %%

def checksum(data):
    return (128-sum(data)&127)%128

# preset = map(lambda x: x[0], presets[3][0]
# %%
def encode_head():
    """
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

    """

    header = [  '0x43',
                '0x00',
                '0x09',
                '0x20',
                '0x00',]

    return [int(i, 0) for i in header]

# consume_head(sysex_iter)
#%%
def encode_osc(params, n):

    """
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
    """

    oscillator_params = []

    oscillator_params += [params[f'{n}_R1']]
    oscillator_params += [params[f'{n}_R2']]
    oscillator_params += [params[f'{n}_R3']]
    oscillator_params += [params[f'{n}_R4']]
    oscillator_params += [params[f'{n}_L1']]
    oscillator_params += [params[f'{n}_L2']]
    oscillator_params += [params[f'{n}_L3']]
    oscillator_params += [params[f'{n}_L4']]
    oscillator_params += [params[f'{n}_BP']]
    oscillator_params += [params[f'{n}_LD']]
    oscillator_params += [params[f'{n}_RD']]

    RC = params[f'{n}_RC'] << 2
    LC = params[f'{n}_LC']
    oscillator_params += [RC | LC]


    DET = params[f'{n}_DET'] << 3
    RS = params[f'{n}_RS']
    oscillator_params += [DET | RS]


    KVS = params[f'{n}_KVS'] << 2
    AMS = params[f'{n}_AMS'] 
    oscillator_params += [KVS|AMS]

    oscillator_params += [params[f'{n}_OL']]


    FC = params[f'{n}_FC'] << 1
    M = params[f'{n}_M']
    oscillator_params += [FC|M]

    oscillator_params += [params[f'{n}_FF']]


    return oscillator_params

#%%
def encode_global(params):
    """
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
    """

    global_params = []


    global_params += [params['PR1']]
    global_params += [params['PR2']]
    global_params += [params['PR3']]
    global_params += [params['PR4']]
    global_params += [params['PL1']]
    global_params += [params['PL2']]
    global_params += [params['PL3']]
    global_params += [params['PL4']]

    global_params += [params['ALG']]

    OKS = params['OKS'] << 3
    print(OKS, '------', params['OKS'])
    FB = params['FB']

    global_params += [OKS|FB]
    print(OKS|FB)


    global_params += [params['LFS']]
    global_params += [params['LFD']]
    global_params += [params['LPMD']]
    global_params += [params['LAMD']]

    LPMS = params['LPMS'] << 4
    LFW = params['LFW'] << 1
    LKS = params['LKS']
    # if (LPMS & LFW) or (LPMS & LKS):
    #     print('que', LPMS | LFW | LKS)
    global_params += [LPMS | LFW | LKS]

    global_params += [params['TRNSP']]

    global_params += [params[f'NAME_{i}'] for i in range(10)]


    return global_params

def encode_syx(params_list):

    head = encode_head()

    data = []
    assert len(params_list) == N_VOICE

    # voices
    for params in params_list:
        
        for osc in range(N_OSC):
            data += encode_osc(params, osc)

        data += encode_global(params)


    this_checksum = checksum(data)


    return [*head, *data, this_checksum]
    
#%%



data = np.load(ARTIFACTS_ROOT.joinpath('dev-dx7.npy'))
data = np.load(ARTIFACTS_ROOT.joinpath('dev-dx7.npy'))[[1]*32]

params_list = list(map(lambda params: dict(zip(PARAMETER_ORDER, list(params))), data))

syx = encode_syx(params_list)

message = mido.Message('sysex', data=syx)
mido.write_syx_file(ARTIFACTS_ROOT.joinpath('patch.syx'), [message])

# %%


# %%
