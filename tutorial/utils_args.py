import argparse

def read_args1(job_dir, data_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', default=job_dir, help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data-dir', default=data_dir, help='GCS or local path where to look for data files')
    args, other = parser.parse_known_args()
    return args.job_dir, args.data_dir