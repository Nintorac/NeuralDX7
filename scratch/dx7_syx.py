#%%
import bitstruct
import mido
from pathlib import Path
from itertools import chain

from dx7_constants import voice_struct, verify, VOICE_PARAMETER_RANGES, header_struct,\
    header_bytes, voice_bytes, take, VOICE_KEYS, ARTIFACTS_ROOT, N_VOICES, N_OSC

# %%


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
#%%
from tqdm import tqdm as tqdm
from functools import reduce
import numpy as np
if __name__=='__main__':
    DEV = False
    dataset_file = 'dx7.npy'
# PRESETS_ROOT = '~/audio/artifacts/dx7-patches'
# PRESETS_ROOT = '~/audio/artifacts/dx7-patches-dev'
    preset_root = ARTIFACTS_ROOT.joinpath('dx7-patches').expanduser()

    preset_paths = iter(tqdm(sorted(preset_root.glob('**/*.syx'))))
    if DEV:
        dataset_file = f'dev-{dataset_file}'
        preset_paths = take(preset_paths, 10)
    preset_paths = iter(filter(lambda preset_path: preset_path.is_file(), preset_paths))

    consume_chain = chain.from_iterable(map(consume_syx, preset_paths))
    consume_chain = filter(lambda x: x is not None, consume_chain)

    arr_dtype = list(zip(VOICE_KEYS, ['u8']*len(VOICE_KEYS)))
    to_arr = lambda voice_dict: np.array([tuple(voice_dict.values())], dtype=arr_dtype)
    arr = np.concatenate(list(map(to_arr, consume_chain)))
    arr = np.unique(arr)
    np.save(ARTIFACTS_ROOT.joinpath(dataset_file).as_posix(), arr)
    # print(to_arr(next(iter(consume_chain))))
    # print(type(consume_chain))
    # 1/0
    # # for i in list(map(list, outputs)):
    # #     print(len(i))
    # paths, ns, data = zip(*outputs)
    # data = np.array(list(set(data)))
    # paths = np.array([path.as_posix() for path in paths])
    # ns = np.array(ns)
    # np.savez(ARTIFACTS_ROOT.joinpath('dx7.npy'), outputs, paths, ns)
    # # patch_map = dict(tqdm((iter(consume_chain))))
    # # random_key, *_ = patch_map.keys()
    # # # print(global_config['NAME'])

    # # name = items.pop('')
    # # print(global_config)
    # # print(json.dumps(oscillator_configs, indent=4))
    # # print(list(sysex_iter))





# %%


# %%
