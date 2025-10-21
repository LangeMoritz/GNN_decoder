import argparse
import sys
import os

sys.path.append("..")
from src.decoder import Decoder

os.environ['JOB_NAME'] = 'my_name'
os.environ['NOTES'] = 'my_note'

def main():
    job_name = os.getenv('JOB_NAME')
    notes = os.getenv('NOTES')
    script_name = notes + '_' + job_name
    # command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", required=True)
    args = parser.parse_args()
    # create decoder object
    config_file = args.configuration
    decoder = Decoder(config_file, script_name)
    decoder.run()
    
if __name__ == "__main__":
    main()