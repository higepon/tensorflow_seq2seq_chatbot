import os.path
import argparse
import config
import glob
import subprocess

# Script to deploy latest model to server


def create_checkpoint_file(path):
    # rewrite path
    path = path.replace('Users', 'home')
    f = open('{}/checkpoint'.format(config.GENERATED_DIR), 'w')
    f.write('model_checkpoint_path: "{}"\n'.format(path))
    f.write('all_model_checkpoint_paths: "{}"\n'.format(path))
    f.close()


def deploy(host):
    files_to_deploy = get_files_to_deploy()
    subprocess.call("scp {} {}:chatbot_generated/".format(' '.join(files_to_deploy), host), shell=True)
    # double check
    if files_to_deploy != get_files_to_deploy():
        raise "inconsistent state"


def get_files_to_deploy():
    source_dir = config.GENERATED_DIR
    all_files = sorted(glob.glob('{}/*ckpt*'.format(source_dir)), reverse=True)
    # first is latest
    target_checkpoint = all_files[0]
    path, _ = os.path.splitext(target_checkpoint)
    create_checkpoint_file(path)
    files_to_deploy = glob.glob('{}.*'.format(path))
    files_to_deploy.append('{}/checkpoint'.format(config.GENERATED_DIR))
    return files_to_deploy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-host', type=str, required=True, help='deploy target host')
    args = parser.parse_args()
    host = args.host
    deploy(host)

if __name__ == '__main__':
    main()
