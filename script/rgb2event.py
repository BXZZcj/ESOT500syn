import subprocess, argparse, os
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--v2ce", default="libs/V2CE-Toolbox/v2ce_cli.py")
    args = ap.parse_args()
    cmd = ["python", args.v2ce, "--input", args.rgb_dir, "--output", args.out_file, "--fps", str(args.fps)]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()