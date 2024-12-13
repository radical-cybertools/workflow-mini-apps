import argparse

def extract_times(filename):
    bootstrap_start_time = None
    bootstrap_stop_time = None
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue

            content = line.strip().split(',')
            event = content[1]
            
            if event == 'bootstrap_0_start' and bootstrap_start_time is None:
                bootstrap_start_time = float(content[0])
            if event == 'bootstrap_0_stop' and bootstrap_stop_time is None:
                bootstrap_stop_time = float(content[0])
            if bootstrap_start_time and bootstrap_stop_time:
                break

    return bootstrap_start_time, bootstrap_stop_time


def main():
    parser = argparse.ArgumentParser(description="Get workflow execution time")
    parser.add_argument('-f', '--file', type=str, required=True, help='filename with path')

    args = parser.parse_args()

    bootstrap_start, bootstrap_stop = extract_times(args.file)

    print(f"First bootstrap_0_start time: {bootstrap_start}")
    print(f"First bootstrap_0_stop time: {bootstrap_stop}")
    print("Workflow execution time: {}".format(bootstrap_stop-bootstrap_start))


if __name__ == '__main__':
    main()
