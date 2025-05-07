# Reproducibile activity monitoring
Here, we make publicly available: (1) a sensor dataset collected from a real smart home, and (2) a set of rules
to analyzing sensor data for activity monitoring. These rules are written with a domain-specific language named Allen (that you can find in this repository).
Because of the dedicated nature of this DSL, our monitoring rules are concise and high-level, facilitating their comprehension and evolution.

**Repository content:**

- Makefile
- Allen/src: the Allen DSL used to execute our rules. Further information about Allen are found at: https://github.com/NicVolanschi/Allen
- dataset.csv: a set of sensor data, collected over one year from a real home
- log-analyses.aln: a set of rules to detect activities as well as sensor failures 
- out: an empty directory that will contain the output of the rules (the detected activities and failures)
- dayspermonth.pl: Perl script to compute the number of days of sensor failures for each month.
- visualization_*.py:  Python scripts to visualize in a synoptic way the output of analyses rules for sensor failures and activity detection.

# Installation

You need Perl with its core modules installed. The Allen code has been tested on Perl versions 5.18 and MacOS X,
but should work on other configurations as well.

The visualization part needs to run the python scripts on Anaconda Jupyter. For your convenience, Anaconda Jupyter is provided as a
Docker container (you need to have Docker installed to use it). Alternatively, you might already have Anaconda Jupyter installed.

# Getting started

1. Download the files of reproducibilitymonitoring repository: `git clone https://gitlab.inria.fr/rbelloum/reproducibilitymonitoring.git`.
2. Go to the folder of the downloaded files: `cd reproducibilitymonitoring`.
3. Execute activity monitoring rules (log-analyses.aln) over the public dataset (dataset.csv) that we provided: `make detect`.
Various files have been generated under out/ and work/ subdirectories. They contain the output of monitoring rules detecting activities and sensor failures. Finally, a docker server has been invoked for visualizing the results.
 **NB: (1) This step may last a few minutes. (2) If this step fails, it means that Allen or make do not work well in your native environment. You can instead run this step in Docker: `make detect-indock` (you must have Docker installed).**
4. Run the docker container with the Jupyter Anaconda visualization server: `make server`. Alternatively, if you have Anaconda Jupyter already installed, you can directly run it from the browser, and proceed to step 6.
5. Copy/paste the indicated URL into your browser. This opens a Jupyter notebook web page.
6. Click on the 'work' subdirectory within the notebook.
7. Click on the 'New' button in the upper right corner of the notebook, and select 'Python 3' from the drop-down menu. This will open a new tab in the browser with an interactive Jupyter envrionment.
8. You can visualize the activity information using the scripts examples for toilet, outing, and sleeping activities, by typing into the command field: `%run visualize_toilet`, `%run visualize_outing`, or `%run visualize_sleep_quiet`, respectively. Once the command is typed in the field, click on the 'Run' button in the top part of the notebook. The corresponding diagram should appear in the notebook.

To delete all the generated files and start anew: `make clean`.

# Documentation

[1] Nic Volanschi, Bernard Serpette, and Charles Consel. "Implementing a Semi-causal Domain-Specific Language for
Context Detection over Binary Sensors". In GPCE 2018 - 17th ACM SIGPLAN International Conference on
Generative Programming: Concepts & Experiences.

[2] Nic Volanschi, Bernard Serpette, Adrien Carteron, and Charles Consel.
"A Language for Online State Processing of Binary Sensors, Applied to Ambient Assisted Living".
In Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2:4, 2018.

[3] Charles Consel, Lucile Dupuy, and Hélène Sauzéon. 2017. HomeAssist: An assisted living platform
for aging in place based on an interdisciplinary approach.
In International Conference on Applied Human Factors and Ergonomics. Springer, 129–140.
