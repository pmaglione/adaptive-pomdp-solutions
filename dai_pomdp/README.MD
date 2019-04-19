Dependencies
-----------------------------
- boto
- gsl
- zmdp

Other Changes to Make
-----------------------------
- EM/MakeFile, path for GSLInclude
- Run MakeFile again

How to Run
-----------------------------
1. Make 'em' (from inside 'EM' folder)
2. Edit the path to your current working directory.
    - Ballots.py, line 43
    - Ballots.py, line 55
    - main.py, line 48
    - main.py, line 51
3. New running method
    - ```/bin/bash run_experiment.sh <zmdp_learning_seconds> <items_num> <data_balance>```