from pathlib import Path
import bitstruct

ARTIFACTS_ROOT = Path('/content/gdrive/My Drive/audio/artifacts').expanduser()

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

if __name__=="__main__":
    print(VOICE_KEYS)