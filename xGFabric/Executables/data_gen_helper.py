import os
import time
import sys

def print_help():
    print("""
Usage: python data_gen_helper.py <root_dir> <cubic_start> <cubic_end> <cubic_step_size>
       <trigonal_c1_part1_start> <trigonal_c1_part1_end> <trigonal_c1_part1_step_size>
       <trigonal_c2_part1_start> <trigonal_c2_part1_end> <trigonal_c2_part1_step_size>
       <trigonal_c1_part2_start> <trigonal_c1_part2_end> <trigonal_c1_part2_step_size>
       <trigonal_c2_part2_start> <trigonal_c2_part2_end> <trigonal_c2_part2_step_size>
       <tetragonal_c1_start> <tetragonal_c1_end> <tetragonal_c1_step_size>
       <tetragonal_c2_start> <tetragonal_c2_end> <tetragonal_c2_step_size>
       <num_step> <cif_path_in> <root_path_out>

Arguments:
    <root_dir>                  Root directory for configuration files.
    <cubic_start>, <cubic_end>, <cubic_step_size>
                                Start, end, and step size for cubic symmetry.
    <trigonal_c1_part1_start>, <trigonal_c1_part1_end>, <trigonal_c1_part1_step_size>
                                Start, end, and step size for trigonal_c1_part1 symmetry.
    <trigonal_c2_part1_start>, <trigonal_c2_part1_end>, <trigonal_c2_part1_step_size>
                                Start, end, and step size for trigonal_c2_part1 symmetry.
    <trigonal_c1_part2_start>, <trigonal_c1_part2_end>, <trigonal_c1_part2_step_size>
                                Start, end, and step size for trigonal_c1_part2 symmetry.
    <trigonal_c2_part2_start>, <trigonal_c2_part2_end>, <trigonal_c2_part2_step_size>
                                Start, end, and step size for trigonal_c2_part2 symmetry.
    <tetragonal_c1_start>, <tetragonal_c1_end>, <tetragonal_c1_step_size>
                                Start, end, and step size for tetragonal_c1 symmetry.
    <tetragonal_c2_start>, <tetragonal_c2_end>, <tetragonal_c2_step_size>
                                Start, end, and step size for tetragonal_c2 symmetry.
    <num_step>                  Number of steps.
    <cif_path_in>               Path to the input .cif file.
    <root_path_out>             Path to the output directory.
""")

if __name__ == '__main__':
    if len(sys.argv) < 26 or '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        sys.exit(1)

    start = time.time()
    print("Arguments are: ", sys.argv)
    root_dir = sys.argv[1]
    cubic_start, cubic_end, cubic_step_size = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    trigonal_c1_part1_start, trigonal_c1_part1_end, trigonal_c1_part1_step_size = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    trigonal_c2_part1_start, trigonal_c2_part1_end, trigonal_c2_part1_step_size = float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10])
    trigonal_c1_part2_start, trigonal_c1_part2_end, trigonal_c1_part2_step_size = float(sys.argv[11]), float(sys.argv[12]), float(sys.argv[13])
    trigonal_c2_part2_start, trigonal_c2_part2_end, trigonal_c2_part2_step_size = float(sys.argv[14]), float(sys.argv[15]), float(sys.argv[16])
    tetragonal_c1_start, tetragonal_c1_end, tetragonal_c1_step_size = float(sys.argv[17]), float(sys.argv[18]), float(sys.argv[19])
    tetragonal_c2_start, tetragonal_c2_end, tetragonal_c2_step_size = float(sys.argv[20]), float(sys.argv[21]), float(sys.argv[22])
    num_step = int(sys.argv[23])
    cif_path_in = sys.argv[24]
    root_path_out = sys.argv[25]

    for pi in range(num_step):

        cubic_start_in = cubic_start + pi * cubic_step_size
        cubic_end_in = cubic_end - (num_step - 1 - pi) * cubic_step_size
        cubic_step_size_in = cubic_step_size * num_step

        cubic_file_name = root_dir + '/configs_phase{}/config_1001460_cubic.txt'.format(pi) #OOK: Why we have hardcoded numbers?
        os.makedirs(os.path.dirname(cubic_file_name), exist_ok = True)
        with open(cubic_file_name, "w") as f:
            f.write("[Global_Params]\n")
            f.write("path_in = \'" + cif_path_in + "\'\n")
            f.write("symmetry = 'cubic'\n")
            f.write("name = 'Ba2BiO5'\n")
            f.write("cif = '1001460.cif'\n")
            f.write("instprm = 'NOMAD-Bank4-ExperimentMatch.instprm'\n")
            f.write("path_out = \'" + root_path_out + "/phase{}/test_cubic/\'\n".format(pi))
            f.write("name_out = 'cubic_1001460'\n")
            f.write("sweep_cell_1 = [{}, {}, {}]\n".format(cubic_start_in, cubic_end_in, cubic_step_size_in))
            f.write("tmin = 1.36\n")
            f.write("tmax = 18.919\n")
            f.write("tstep = 0.0009381\n")
        os.makedirs(root_path_out + "/phase{}/test_cubic/".format(pi), exist_ok=True)

        trigonal_c1_part1_start_in = trigonal_c1_part1_start + pi * trigonal_c1_part1_step_size
        trigonal_c1_part1_end_in = trigonal_c1_part1_end - (num_step - 1 - pi) * trigonal_c1_part1_step_size
        trigonal_c1_part1_step_size_in = trigonal_c1_part1_step_size * num_step

        trigonal_part1_file_name = root_dir + 'configs_phase{}/config_1522004_trigonal_part1.txt'.format(pi)
        os.makedirs(os.path.dirname(trigonal_part1_file_name), exist_ok = True)
        with open(trigonal_part1_file_name, "w") as f:
            f.write("[Global_Params]\n")
            f.write("path_in = \'" + cif_path_in + "\'\n")
            f.write("symmetry = 'trigonal'\n")
            f.write("name = 'LaMnO3'\n")
            f.write("cif = '1522004.cif'\n")
            f.write("instprm = 'NOMAD-Bank4-ExperimentMatch.instprm'\n")
            f.write("path_out = \'" + root_path_out + "/phase{}/test_trigonal_part1/\'\n".format(pi))
            f.write("name_out = 'trigonal_1522004'\n")
            f.write("sweep_cell_1 = [{}, {}, {}]\n".format(trigonal_c1_part1_start_in, trigonal_c1_part1_end_in, trigonal_c1_part1_step_size_in))
            f.write("sweep_cell_4 = [{}, {}, {}]\n".format(trigonal_c2_part1_start, trigonal_c2_part1_end, trigonal_c2_part1_step_size))
            f.write("tmin = 1.36\n")
            f.write("tmax = 18.919\n")
            f.write("tstep = 0.0009381\n")
        os.makedirs(root_path_out + "/phase{}/test_trigonal_part1/".format(pi), exist_ok=True)

        trigonal_c1_part2_start_in = trigonal_c1_part2_start + pi * trigonal_c1_part2_step_size
        trigonal_c1_part2_end_in = trigonal_c1_part2_end - (num_step - 1 - pi) * trigonal_c1_part2_step_size
        trigonal_c1_part2_step_size_in = trigonal_c1_part2_step_size * num_step

        trigonal_part2_file_name = root_dir + 'configs_phase{}/config_1522004_trigonal_part2.txt'.format(pi)
        os.makedirs(os.path.dirname(trigonal_part2_file_name), exist_ok = True)
        with open(trigonal_part2_file_name, "w") as f:
            f.write("[Global_Params]\n")
            f.write("path_in = \'" + cif_path_in + "\'\n")
            f.write("symmetry = 'trigonal'\n")
            f.write("name = 'LaMnO3'\n")
            f.write("cif = '1522004.cif'\n")
            f.write("instprm = 'NOMAD-Bank4-ExperimentMatch.instprm'\n")
            f.write("path_out = \'" + root_path_out + "/phase{}/test_trigonal_part2/\'\n".format(pi))
            f.write("name_out = 'trigonal_1522004'\n")
            f.write("sweep_cell_1 = [{}, {}, {}]\n".format(trigonal_c1_part2_start_in, trigonal_c1_part2_end_in, trigonal_c1_part2_step_size_in))
            f.write("sweep_cell_4 = [{}, {}, {}]\n".format(trigonal_c2_part2_start, trigonal_c2_part2_end, trigonal_c2_part2_step_size))
            f.write("tmin = 1.36\n")
            f.write("tmax = 18.919\n")
            f.write("tstep = 0.0009381\n")
        os.makedirs(root_path_out + "/phase{}/test_trigonal_part2/".format(pi), exist_ok=True)

        tetragonal_c1_start_in = tetragonal_c1_start + pi * tetragonal_c1_step_size
        tetragonal_c1_end_in = tetragonal_c1_end - (num_step - 1 - pi) * tetragonal_c1_step_size
        tetragonal_c1_step_size_in = tetragonal_c1_step_size * num_step

        tetragonal_file_name = root_dir + 'configs_phase{}/config_1531431_tetragonal.txt'.format(pi)
        os.makedirs(os.path.dirname(tetragonal_file_name), exist_ok = True)
        with open(tetragonal_file_name, "w") as f:
            f.write("[Global_Params]\n")
            f.write("path_in = \'" + cif_path_in + "\'\n")
            f.write("symmetry = 'tetragonal'\n")
            f.write("name = 'KNbO3'\n")
            f.write("cif = '1531431.cif'\n")
            f.write("instprm = 'NOMAD-Bank4-ExperimentMatch.instprm'\n")
            f.write("path_out = \'" + root_path_out + "/phase{}/test_tetragonal/\'\n".format(pi))
            f.write("name_out = 'tetragonal_1531431'\n")
            f.write("sweep_cell_1 = [{}, {}, {}]\n".format(tetragonal_c1_start_in, tetragonal_c1_end_in, tetragonal_c1_step_size_in))
            f.write("sweep_cell_3 = [{}, {}, {}]\n".format(tetragonal_c2_start, tetragonal_c2_end, tetragonal_c2_step_size))
            f.write("tmin = 1.36\n")
            f.write("tmax = 18.919\n")
            f.write("tstep = 0.0009381\n")
        os.makedirs(root_path_out + "/phase{}/test_tetragonal/".format(pi), exist_ok=True)

    end = time.time()
    print("Time for generating config file for gsas is ", end - start)
