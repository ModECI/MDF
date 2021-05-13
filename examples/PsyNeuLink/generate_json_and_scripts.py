import glob
import os
import runpy
import subprocess
import psyneulink as pnl


def main():
    reconstructed_identifer = 'reconstructed'

    for example in glob.glob(os.path.join(os.path.dirname(__file__), '*.py')):
        if reconstructed_identifer in example or example == __file__:
            continue

        pnl.clear_registry()
        border = '=' * len(example)
        print(f'{border}\n{example}\n{border}')
        base_fname = example.replace('.py', '')
        script_globals = runpy.run_path(example)


        compositions = list(filter(lambda v: isinstance(v, pnl.Composition), script_globals.values()))
        nonnested_comps = []

        for x in compositions:
            for y in compositions:
                if x in y.nodes:
                    break
            else:
                nonnested_comps.append(x)

        try:
            comp = nonnested_comps[0]
        except IndexError:
            continue

        json_summary = pnl.generate_json(comp)

        with open(f'{base_fname}.json', 'w') as outfi:
            outfi.write(json_summary)

        reconstructed_fname = f'{base_fname}.{reconstructed_identifer}.py'
        with open(reconstructed_fname, 'w') as outfi:
            outfi.write(pnl.generate_script_from_json(json_summary))
        subprocess.run(['black', reconstructed_fname])
        subprocess.run(['python', reconstructed_fname])

if __name__ == "__main__":
    main()
