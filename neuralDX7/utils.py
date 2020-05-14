
import mido
import torch
import numpy as np
from pathlib import Path
from itertools import chain
from neuralDX7.constants import VOICE_KEYS, VOICE_PARAMETER_RANGES, MAX_VALUE, checksum
from neuralDX7.constants import voice_struct, VOICE_KEYS, checksum
import bitstruct


def mask_parameters(x, voice_keys=VOICE_KEYS, inf=1e9):
    device = x.device
    mask_item_f = lambda x: torch.arange(MAX_VALUE).to(device) > max(x) 
    mapper = map(mask_item_f, map(VOICE_PARAMETER_RANGES.get, voice_keys))

    mask = torch.stack(list(mapper))
    
    return torch.masked_fill(x, mask, -inf)


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

def dx7_bulk_pack(voices):

    HEADER = int('0x43', 0), int('0x00', 0), int('0x09', 0), int('0x20', 0), int('0x00', 0)
    assert len(voices)==32
    voices_bytes = bytes()
    for voice in voices:
        voice_bytes = voice_struct.pack(dict(zip(VOICE_KEYS, voice)))
        voices_bytes += voice_bytes
    
    
    patch_checksum = [checksum(voices_bytes)]

    data = bytes(HEADER) + voices_bytes + bytes(patch_checksum)

    return mido.Message('sysex', data=data)



def generate_syx(patch_list):

    dx7_struct