import argparse

def extract_times(filename):
    timetable_start = {}
    timetable_stop = {}

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue

            content = line.strip().split(',')
            event = content[1]
            
            if event == 'exec_start':
                task_id = content[4]
                if task_id in timetable_start:
                    continue
                else:
                    timetable_start[task_id] = float(content[0])
            if event == 'exec_stop':
                task_id = content[4]
                timetable_stop[task_id] = float(content[0])

    return timetable_start, timetable_stop 


def main():
    parser = argparse.ArgumentParser(description="Get task execution time")
    parser.add_argument('-f', '--file', type=str, required=True, help='filename with path')

    args = parser.parse_args()

    timetable_start, timetable_stop = extract_times(args.file)

    print(f"Task exec_start timetable: {timetable_start}")
    print(f"Task exec_stop timetable: {timetable_stop}")

    assert(len(timetable_start) == len(timetable_stop))
    task_execution_table = {}

    for task_id, start_time in timetable_start.items():
        stop_time = timetable_stop[task_id]
        task_execution_table[task_id] = stop_time - start_time

    print(f"Task duration: {task_execution_table}")



if __name__ == '__main__':
    main()
