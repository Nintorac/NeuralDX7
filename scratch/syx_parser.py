#%%
import mido
from pathlib import Path
import json
from uuid import uuid4
from itertools import chain
from tqdm import tqdm as tqdm
import numpy as np

PARAMETER_ORDER = ['PR1', 'PR2', 'PR3', 'PR4', 'PL1', 'PL2', 'PL3', 'PL4', 'ALG', 'OKS', 'FB', 'LFS', 'LFD', 'LPMD', 'LAMD', 'LPMS', 'LFW', 'LKS', 'TRNSP', 'NAME_0', 'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4', 'NAME_5', 'NAME_6', 'NAME_7', 'NAME_8', 'NAME_9', '0_R1', '0_R2', '0_R3', '0_R4', '0_L1', '0_L2', '0_L3', '0_L4', '0_BP', '0_LD', '0_RD', '0_RC', '0_LC', '0_DET', '0_RS', '0_KVS', '0_AMS', '0_OL', '0_FC', '0_M', '0_FF', '1_R1', '1_R2', '1_R3', '1_R4', '1_L1', '1_L2', '1_L3', '1_L4', '1_BP', '1_LD', '1_RD', '1_RC', '1_LC', '1_DET', '1_RS', '1_KVS', '1_AMS', '1_OL', '1_FC', '1_M', '1_FF', '2_R1', '2_R2', '2_R3', '2_R4', '2_L1', '2_L2', '2_L3', '2_L4', '2_BP', '2_LD', '2_RD', '2_RC', '2_LC', '2_DET', '2_RS', '2_KVS', '2_AMS', '2_OL', '2_FC', '2_M', '2_FF', '3_R1', '3_R2', '3_R3', '3_R4', '3_L1', '3_L2', '3_L3', '3_L4', '3_BP', '3_LD', '3_RD', '3_RC', '3_LC', '3_DET', '3_RS', '3_KVS', '3_AMS', '3_OL', '3_FC', '3_M', '3_FF', '4_R1', '4_R2', '4_R3', '4_R4', '4_L1', '4_L2', '4_L3', '4_L4', '4_BP', '4_LD', '4_RD', '4_RC', '4_LC', '4_DET', '4_RS', '4_KVS', '4_AMS', '4_OL', '4_FC', '4_M', '4_FF', '5_R1', '5_R2', '5_R3', '5_R4', '5_L1', '5_L2', '5_L3', '5_L4', '5_BP', '5_LD', '5_RD', '5_RC', '5_LC', '5_DET', '5_RS', '5_KVS', '5_AMS', '5_OL', '5_FC', '5_M', '5_FF']

def uuid():

    return uuid4().hex

ARTIFACTS_ROOT = Path('~/audio/artifacts').expanduser()



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
    'NAME_0': range(128),
    'NAME_1': range(128),
    'NAME_2': range(128),
    'NAME_3': range(128),
    'NAME_4': range(128),
    'NAME_5': range(128),
    'NAME_6': range(128),
    'NAME_7': range(128),
    'NAME_8': range(128),
    'NAME_9': range(128),
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

def verify(actual, ranges, prefix=None):
        
    assert set(actual.keys())==set(ranges.keys()), 'Params dont match'

    for key in actual:
        if not actual[key] in ranges[key]:
            # print(f'{key} value {actual[key]} should be in {ranges[key]}')
            return False
    return True

# # %%
# presets = [mido.read_syx_file(patch.as_posix()) for patch in iter(dexed_presets)]

N_OSC = 6
N_VOICE = 32
# # %%
# preset = map(lambda x: x[0], presets[3][0]
# %%
def consume_head(sysex_iter):
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

    expected = ['0x43',
                '0x00',
                '0x09',
                '0x20',
                '0x00',]

    for i in expected:
        assert int(i, 0) == next(sysex_iter), 'unexpected header'

# consume_head(sysex_iter)
#%%
def consume_osc(sysex_iter):

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

    def process_byte(this_byte):

        this_byte = this_byte & int('0b1111111', 0)

        return int(this_byte)

    int_sysex_iter = iter(map(process_byte, sysex_iter))

    R1 = next(int_sysex_iter)
    R2 = next(int_sysex_iter)
    R3 = next(int_sysex_iter)
    R4 = next(int_sysex_iter)
    L1 = next(int_sysex_iter)
    L2 = next(int_sysex_iter)
    L3 = next(int_sysex_iter)
    L4 = next(int_sysex_iter)
    BP = next(int_sysex_iter)
    LD = next(int_sysex_iter)
    RD = next(int_sysex_iter)

    _RC_LC = next(int_sysex_iter) & int('0b1111', 0)
    RC = _RC_LC >> 2
    LC = _RC_LC & int('0b11', 0)



    _DET_RS = next(int_sysex_iter) & int('0b11111', 0)
    DET = _DET_RS >> 4
    RS = _DET_RS & int('0b111', 0)

    _KVS_AMS = next(int_sysex_iter) & int('0b11111', 0)
    KVS = _KVS_AMS >> 2
    AMS = _KVS_AMS & int('0b11', 0)

    OL = next(int_sysex_iter)

    _FC_M = next(int_sysex_iter) & int('0b111111', 0)
    FC = _FC_M >> 1
    M = _FC_M & int('0b1', 0)

    FF = next(int_sysex_iter)


    oscilattor_config = {
        'R1': R1,
        'R2': R2,
        'R3': R3,
        'R4': R4,
        'L1': L1,
        'L2': L2,
        'L3': L3,
        'L4': L4,
        'BP': BP,
        'LD': LD,
        'RD': RD,
        'RC': RC,
        'LC': LC,
        'DET': DET,
        'RS': RS,
        'KVS': KVS,
        'AMS': AMS,
        'OL': OL,
        'FC': FC,
        'M': M,
        'FF': FF,
    }

    return oscilattor_config
#%%



def consume_global(sysex_iter):
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

    def process_byte(this_byte):

        this_byte = this_byte & int('0b111111', 0)

        return this_byte

    sysex_iter = iter(map(process_byte, sysex_iter))


    PR1 = int(next(sysex_iter))
    PR2 = int(next(sysex_iter))
    PR3 = int(next(sysex_iter))
    PR4 = int(next(sysex_iter))
    PL1 = int(next(sysex_iter))
    PL2 = int(next(sysex_iter))
    PL3 = int(next(sysex_iter))
    PL4 = int(next(sysex_iter))

    ALG = int(next(sysex_iter)) & int('0b11111', 0)

    OKS_FB = int(next(sysex_iter))
    OKS = (OKS_FB & int('0b1000', 0)) >> 3
    FB = (OKS_FB & int('0b111', 0))

    LFS = int(next(sysex_iter))
    LFD = int(next(sysex_iter))
    LPMD = int(next(sysex_iter))
    LAMD = int(next(sysex_iter))

    LPMS_LFW_LKS = int(next(sysex_iter))
    LPMS = LPMS_LFW_LKS >> 4
    LFW = (LPMS_LFW_LKS >> 1) & int('0b111', 0)
    LKS = LPMS_LFW_LKS & int('0b1', 0)

    TRNSP = int(next(sysex_iter))

    to_ascii = lambda byte: bytes.fromhex(byte.decode("ascii")).decode('ascii')
    to_ascii = lambda byte: ascii(byte.decode('ascii'))
    NAME = [(f'NAME_{i}', int(next(sysex_iter))) for i in range(10)]

    global_config = {
        'PR1': PR1,
        'PR2': PR2,
        'PR3': PR3,
        'PR4': PR4,
        'PL1': PL1,
        'PL2': PL2,
        'PL3': PL3,
        'PL4': PL4,
        'ALG': ALG,
        'OKS': OKS,
        'FB': FB,
        'LFS': LFS,
        'LFD': LFD,
        'LPMD': LPMD,
        'LAMD': LAMD,
        'LPMS': LPMS,
        'LFW': LFW,
        'LKS': LKS,
        'TRNSP': TRNSP,
    }
    global_config.update(NAME)

    return uuid(), global_config

def consume_syx(path):

    path = Path(path).expanduser()

    try:
        preset = mido.read_syx_file(path.as_posix())[0]
    except IndexError as e:
        return None
    except ValueError as e:
        return None

    sysex_iter = iter(preset.data)
    # print(len(list(preset.bytes())))
    try:
        consume_head(sysex_iter)
    except AssertionError as e:
        return None

    for i in range(N_VOICE):
        def consume_oscillator():

            oscilattor_config = consume_osc(sysex_iter)

            if not verify(oscilattor_config, OSCILLATOR_VALID_RANGES):
                raise ValueError('Oscillator has values outside range')

            return oscilattor_config.items()

        prefix_oscillator = lambda n: [(f'{n}_{key}', value) for key,value in consume_oscillator()]
        oscilattor_mapper = chain.from_iterable(map(prefix_oscillator, range(N_OSC)))
        
        # oscillator_config = {i: consume_osc(sysex_iter) for i in range(N_OSC)}
        patch_config = {}
        has_error = False
        # oscilattor
        try:
            patch_config.update(oscilattor_mapper)
        except ValueError:
            # invalid range in oscillators
            has_error = True

        name, global_config = consume_global(sysex_iter)

        if not verify(global_config, GLOBAL_VALID_RANGES) or has_error:
            # print('eror')
            yield
            continue

        patch_config.update(global_config)




        yield ((path, i), patch_config)
#%%

if __name__=='__main__':
    DEV = False

    # PRESETS_ROOT = '~/audio/artifacts/dx7-patches'
    # PRESETS_ROOT = '~/audio/artifacts/dx7-patches-dev'
    preset_root = ARTIFACTS_ROOT.joinpath('dx7-patches').expanduser()

    preset_paths = sorted(preset_root.glob('**/*.syx'))
    if DEV:
        preset_paths = preset_paths[:10]
    preset_paths = iter(filter(lambda preset_path: preset_path.is_file(), preset_paths))

    consume_chain = chain.from_iterable(map(consume_syx, preset_paths))
    consume_chain = filter(lambda x: x is not None, consume_chain)

    outputs = []
    for (path, i), params in tqdm(consume_chain):
        # print(params)
        data = map(params.get, PARAMETER_ORDER)
        # _, data = zip(*params.items())
        # print(data)
        # break
        outputs += [(path, i, tuple(data))]

    # for i in list(map(list, outputs)):
    #     print(len(i))
    paths, ns, data = zip(*outputs)
    data = np.array(list(set(data)))
    paths = np.array([path.as_posix() for path in paths])
    ns = np.array(ns)
    np.savez(ARTIFACTS_ROOT.joinpath('dx7.npy'), outputs, paths, ns)
    # patch_map = dict(tqdm((iter(consume_chain))))
    # random_key, *_ = patch_map.keys()
    # # print(global_config['NAME'])

    # name = items.pop('')
    # print(global_config)
    # print(json.dumps(oscillator_configs, indent=4))
    # print(list(sysex_iter))





# %%


# %%
