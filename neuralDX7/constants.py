from pathlib import Path
import bitstruct
import mido


def take(take_from, n):
    for _ in range(n):
        yield next(take_from)

N_OSC = 6
N_VOICES = 32

def checksum(data):
    return (128-sum(data)&127)%128

GLOBAL_VALID_RANGES = {
    'PR1':  range(0, 99+1),
    'PR2':  range(0, 99+1),
    'PR3':  range(0, 99+1),
    'PR4':  range(0, 99+1),
    'PL1':  range(0, 99+1),
    'PL2':  range(0, 99+1),
    'PL3':  range(0, 99+1),
    'PL4':  range(0, 99+1),
    'ALG':  range(0, 31+1),
    'OKS':  range(0, 1+1),
    'FB':   range(0, 7+1),
    'LFS':  range(0, 99+1),
    'LFD':  range(0, 99+1),
    'LPMD':  range(0, 99+1),
    'LAMD':  range(0, 99+1),
    'LPMS': range(0, 7+1),
    'LFW':  range(0, 5+1),
    'LKS':  range(0, 1+1),
    'TRNSP':  range(0, 48+1),
    'NAME CHAR 1': range(128),
    'NAME CHAR 2': range(128),
    'NAME CHAR 3': range(128),
    'NAME CHAR 4': range(128),
    'NAME CHAR 5': range(128),
    'NAME CHAR 6': range(128),
    'NAME CHAR 7': range(128),
    'NAME CHAR 8': range(128),
    'NAME CHAR 9': range(128),
    'NAME CHAR 10': range(128),
 }

OSCILLATOR_VALID_RANGES = {
    'R1':  range(0, 99+1),
    'R2':  range(0, 99+1),
    'R3':  range(0, 99+1),
    'R4':  range(0, 99+1),
    'L1':  range(0, 99+1),
    'L2':  range(0, 99+1),
    'L3':  range(0, 99+1),
    'L4':  range(0, 99+1),
    'BP':  range(0, 99+1),
    'LD':  range(0, 99+1),
    'RD':  range(0, 99+1),
    'RC':  range(0, 3+1),
    'LC':  range(0, 3+1),
    'DET': range(0, 14+1),
    'RS':  range(0, 7+1),
    'KVS': range(0, 7+1),
    'AMS': range(0, 3+1),
    'OL':  range(0, 99+1),
    'FC':  range(0, 31+1),
    'M':   range(0, 1+1),
    'FF':  range(0, 99+1),
}

VOICE_PARAMETER_RANGES = {f'{i}_{key}': value for key, value in OSCILLATOR_VALID_RANGES.items() for i in range(N_OSC)}
VOICE_PARAMETER_RANGES.update(GLOBAL_VALID_RANGES)

def verify(actual, ranges):
    assert set(actual.keys())==set(ranges.keys()), 'Params dont match'
    for key in actual:
        if not actual[key] in ranges[key]:
            return False
    return True


HEADER_KEYS = [
    'ID',
    'Sub-status',
    'format number',
    'byte count',
    'byte count',
]

GENERAL_KEYS = [
    'PR1',
    'PR2',
    'PR3',
    'PR4',
    'PL1',
    'PL2',
    'PL3',
    'PL4',
    'ALG',
    'OKS',
    'FB',
    'LFS',
    'LFD',
    'LPMD',
    'LAMD',
    'LPMS',
    'LFW',
    'LKS',
    'TRNSP',
    'NAME CHAR 1',
    'NAME CHAR 2',
    'NAME CHAR 3',
    'NAME CHAR 4',
    'NAME CHAR 5',
    'NAME CHAR 6',
    'NAME CHAR 7',
    'NAME CHAR 8',
    'NAME CHAR 9',
    'NAME CHAR 10',
]

OSC_KEYS = [
    'R1',
    'R2',
    'R3',
    'R4',
    'L1',
    'L2',
    'L3',
    'L4',
    'BP',
    'LD',
    'RD',
    'RC',
    'LC',
    'DET',
    'RS',
    'KVS',
    'AMS',
    'OL',
    'FC',
    'M',
    'FF',
]

FOOTER_KEYS = ['checksum']


VOICE_KEYS = [f'{i}_{key}' for i in range(6) for key in OSC_KEYS] + \
        GENERAL_KEYS 

KEYS =  HEADER_KEYS + \
        list(VOICE_KEYS * N_VOICES) + \
        FOOTER_KEYS



header_bytes = [
    'p1u7',             # ID # (i=67; Yamaha)
    'p1u7',             # Sub-status (s=0) & channel number (n=0; ch 1)
    'p1u7',             # format number (f=9; 32 voices)
    'p1u7',             # byte count MS byte
    'p1u7',             # byte count LS byte (b=4096; 32 voices)
]




general_parameter_bytes = [ 
    'p1u7',             # PR1
    'p1u7',             # PR2
    'p1u7',             # PR3
    'p1u7',             # PR4
    'p1u7',             # PL1
    'p1u7',             # PL2
    'p1u7',             # PL3
    'p1u7',             # PL4
    'p3u5',             # ALG
    'p4u1u3',           # OKS|    FB
    'p1u7',             # LFS
    'p1u7',             # LFD
    'p1u7',             # LPMD
    'p1u7',             # LAMD
    'p1u3u3u1',         # LPMS |      LFW      |LKS
    'p1u7',             # TRNSP
    'p1u7',             # NAME CHAR 1
    'p1u7',             # NAME CHAR 2
    'p1u7',             # NAME CHAR 3
    'p1u7',             # NAME CHAR 4
    'p1u7',             # NAME CHAR 5
    'p1u7',             # NAME CHAR 6
    'p1u7',             # NAME CHAR 7
    'p1u7',             # NAME CHAR 8
    'p1u7',             # NAME CHAR 9
    'p1u7',             # NAME CHAR 10
]

osc_parameter_bytes = [
    'p1u7',         # R1
    'p1u7',         # R2
    'p1u7',         # R3
    'p1u7',         # R4
    'p1u7',         # L1
    'p1u7',         # L2
    'p1u7',         # L3
    'p1u7',         # L4
    'p1u7',         # BP
    'p1u7',         # LD
    'p1u7',         # RD
    'p4u2u2',       # RC | LC 
    'p1u4u3',       # DET | RS
    'p3u3u2',       # KVS | AMS
    'p1u7',         # OL
    'p2u5u1',       # FC | M
    'p1u7'          # FF
]

voice_bytes = (osc_parameter_bytes * N_OSC) + general_parameter_bytes

tail_bytes = [
    'p1u7',         # checksum
]


full_string = ''.join(header_bytes + osc_parameter_bytes * 6 + general_parameter_bytes)
dx7_struct = bitstruct.compile(full_string)

voice_struct = bitstruct.compile(''.join(voice_bytes), names=VOICE_KEYS)
header_struct = bitstruct.compile(''.join(header_bytes))

N_PARAMS = len(VOICE_PARAMETER_RANGES)
MAX_VALUE = max([max(i) for i in VOICE_PARAMETER_RANGES.values()]) + 1


"""
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
"""
class DX7Single():
    HEADER = int('0x43', 0), int('0x00', 0), int('0x00', 0), int('0x01', 0), int('0x1B', 0)
    
    GENERAL_KEYS = [
        'PR1',
        'PR2',
        'PR3',
        'PR4',
        'PL1',
        'PL2',
        'PL3',
        'PL4',
        'ALG',
        'FB',
        'OKS',
        'LFS',
        'LFD',
        'LPMD',
        'LAMD',
        'LKS',
        'LFW',
        'LPMS',
        'TRNSP',
        'NAME CHAR 1',
        'NAME CHAR 2',
        'NAME CHAR 3',
        'NAME CHAR 4',
        'NAME CHAR 5',
        'NAME CHAR 6',
        'NAME CHAR 7',
        'NAME CHAR 8',
        'NAME CHAR 9',
        'NAME CHAR 10',
    ]

    OSC_KEYS = [
        'R1',
        'R2',
        'R3',
        'R4',
        'L1',
        'L2',
        'L3',
        'L4',
        'BP',
        'LD',
        'RD',
        'LC',
        'RC',
        'RS',
        'AMS',
        'KVS',
        'OL',
        'M',
        'FC',
        'FF',
        'DET',
    ]

    @staticmethod
    def keys():

        osc_keys = DX7Single.OSC_KEYS
        osc_params = [f'{i}_{param}' for i in range(N_OSC) for param in osc_keys]
        # print(osc_params)
        all = osc_params + DX7Single.GENERAL_KEYS
        return all

    @staticmethod
    def struct():
        return bitstruct.compile('p1u7'*155, names=DX7Single.keys())


    @staticmethod
    def to_syx(voices):


        assert len(voices)==1
        voice = voices[0]
        voices_bytes = bytes()
        voices_bytes = DX7Single.struct().pack(dict(zip(VOICE_KEYS, voice)))    
        
        patch_checksum = [checksum(voices_bytes)]

        data = bytes(DX7Single.HEADER) \
            + voices_bytes \
            + bytes(patch_checksum)


        return mido.Message('sysex', data=data)
        

def consume_syx(path):

    path = Path(path).expanduser()
    try:
        preset = mido.read_syx_file(path.as_posix())[0]
    except IndexError as e:
        return None
    except ValueError as e:
        return None
    if len(preset.data) == 0:
        return None

    def get_voice(data):
        
        unpacked = voice_struct.unpack(data)

        if not verify(unpacked, VOICE_PARAMETER_RANGES):
            return None
        
        return unpacked

    get_header = header_struct.unpack
    sysex_iter = iter(preset.data)
    
    try:
        header = get_header(bytes(take(sysex_iter, len(header_bytes))))
        yield from (get_voice(bytes(take(sysex_iter, len(voice_bytes)))) for _ in range(N_VOICES))
    except RuntimeError:
        return None

if __name__=="__main__":
    print(VOICE_KEYS)

    # print(DX7Single.to_syx(n)