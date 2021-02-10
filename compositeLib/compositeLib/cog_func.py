import os

def _run_command(cmd, verbose=False):
    if verbose:
        print(cmd)
    result = os.system(cmd)
    if result != 0:
        raise Exception('command "%s" failed with code %d.' % (cmd, result))


def cog_create_from_tif(src_tif,dst_cog):
    command = f'rio cogeo create {src_tif} {dst_cog}'
    _run_command(command)


