import json
from training.data_evlk import get_dataloader
import cvar_pyutils.debugging_tools
import os
from training.params_evlk import parse_args

def main():
    args = parse_args()
    if args.debug:
        if args.debug_ip is None:
            import pydevd_pycharm
            pydevd_pycharm.settrace(os.environ['SSH_CONNECTION'].split()[0], port=args.debug_port,
                                    stdoutToServer=True,
                                    stderrToServer=True, suspend=False)
        else:
            cvar_pyutils.debugging_tools.set_remote_debugger(args.debug_ip, args.debug_port)


    dataloader = get_dataloader(args)

    for batch in dataloader:
        continue


if __name__ == '__main__':
    main()
