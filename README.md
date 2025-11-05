# cav-class

----

This repository contains the python files and Jupyter notebooks for the main project of the Connected and Autonomous Vehicle class.

- The notebooks provide an implementation of the topics taught in class to be completed by the students, as well as questions to test their understanding of their code and the concepts behind it. For each module, the notebook must be finished before starting the Python files.
- The Python files must be completed by the students in order to implement the previously introduced concept within the CARLA simulator.

# Set-up

----

To use this code, you will need to download the CARLA open-source simulator at https://github.com/carla-simulator/carla/releases/tag/0.9.16

After installing conda creating your environment, activate it and install the requirements with: conda install --file requirements.txt

Make sure your terminal is in the same directory as the requirements.txt file.

# Dataset

----

The perception module rely on 2 different datasets.

- the first one is a traffic sign classification dataset that can be downloaded from this repository directly.
- the second one is a CARLA detection dataset containing 8 classes, from cars to traffic lights. It is located at https://app.roboflow.com/carla-test/modified-carla/3

