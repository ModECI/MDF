set -ex

# Temporary file for helping installation of MDF on OSBv2

# To use this:
#  1) Go to https://www.v2.opensourcebrain.org
#  2) Log in (register first if you're not an OSBv2 user)
#  3) Go to https://www.v2.opensourcebrain.org/repositories/12
#  4) Click on Create new workspace. Dialog opens
#  5) Give it a unique name (e.g. with your username). Description and image are optional
#  6) Click on Create a new workspace.
#        DO accept the offer to copy all files from the GitHub repo to the new workspace (Click OK)
#        DON'T accept the offer to open workspace straight away; opens in wrong application.. (Click CANCEL instead)
#  7) Back at https://www.v2.opensourcebrain.org, find your new workspace, click the ... (3 dots, top right), open with JupyterLab
#  8) When JupyterLab opens, click on Terminal in the launcher
#  9) Type:
#        cd Mod*/expo_demo/
#        ./install_on_osbv2.sh
#        cd examples/MDF/
#        python simple.py -run
#  10) Hey presto, you've just run your first MDF model!

pip install dask==2.30.0 distributed==2.30.1 GPy==1.10.0 protobuf==3.17.0 torch==1.8.0
python setup.py install


