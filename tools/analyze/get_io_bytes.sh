#!/bin/bash

main() {
    if [ "$#" -ne 5 ]
    then
	    echo "Error! The format is: ./this_script.sh rct_sandbox_name darshan_dir task_type (sim/train) task_id r/w (for read or write)"
        return 1
    fi

#    rct_sandbox="re.session.polaris-login-01.twang3.019559.0000"
#    darshan_dir="/eagle/RECUP/twang/miniapp/exalearn-original/real_work_polaris_cpu/darshan_record/ex3_p1"

    rct_sandbox=$1
    darshan_dir=$2
    task_type=$3
    task_id=$4
    out_mode=$5

    rct_input_file="/home/twang3/radical.pilot.sandbox/${rct_sandbox}/pilot.0000/${task_type}.000${task_id}/${task_type}.000${task_id}.out"

    if [ "$out_mode" = "r" ]
    then
        grep "Temp for Darshan, " ${rct_input_file} | awk '{print $7 $10}' | while IFS="," read -r pid hostname; do
            if [ $(ls ${darshan_dir}/twang3_python_id*-${pid}_*-$hostname-* | wc -l) -ne 1 ]; then
                echo "Error with ${pid} and ${hostname}"
                return 1
            fi
            filename=$(ls ${darshan_dir}/twang3_python_id*-${pid}_*-$hostname-*); echo -n "$filename    ";
            darshan-parser $filename | grep -E "POSIX_BYTES_READ" | awk '{sum += $5} END {print sum}';
        done | sort -k2 | awk '{print; sum += $2} END {print "Total read = ", sum}'
    elif [ "$out_mode" = "w" ]
    then
        grep "Temp for Darshan, " ${rct_input_file} | awk '{print $7 $10}' | while IFS="," read -r pid hostname; do
            if [ $(ls ${darshan_dir}/twang3_python_id*-${pid}_*-$hostname-* | wc -l) -ne 1 ]; then
                echo "Error with ${pid} and ${hostname}"
                return 1
            fi
            filename=$(ls ${darshan_dir}/twang3_python_id*-${pid}_*-$hostname-*); echo -n "$filename    ";
            darshan-parser $filename | grep -E "POSIX_BYTES_WRITTEN" | awk '{sum += $5} END {print sum}';
        done | sort -k2 | awk '{print; sum += $2} END {print "Total write = ", sum}'
    else
        echo "Error! Mode has to be r(read) or w(write)"
        return 1
    fi
}

main "$@"
