# rocket-landing-demo
Basic mock-up of a fully automated rocket landing on a droneship using Kerbal Space Program, the kRPC add-on, and Python. 
  
## Files ######

  * README.md
  * xxxx.py
  * xxxx.ship
  
## Software ######

  * [Kerbal Space Program](https://www.kerbalspaceprogram.com/en/)
  * [Python 2.7](https://www.python.org/download/releases/2.7/)
  * [CKAN](https://github.com/KSP-CKAN/CKAN/releases)
  
## KSP Add-ons ######

  * [kRPC](https://krpc.github.io/krpc/)
  * [Kerbal Reusability Expansion](https://forum.kerbalspaceprogram.com/index.php?/topic/138871-145-kre-kerbal-reusability-expansion/)
  * [CKAN](https://github.com/KSP-CKAN/CKAN/releases)
  
## Python Libraries #####

  * [kRPC](https://krpc.github.io/krpc/)
  * [numpy](http://www.numpy.org/)
  * [pandas](https://pandas.pydata.org/)

## Installation Instructions #####

  1. Install Kerbal Space Program
  2. Install CKAN
  3. Put the xxxx.ship into the ship folder in the game directory ".../Game/..."
  4. Use CKAN to install kRPC and Kerbal Reusability Expansion
  5. Install Python 2.7
  6. Install kRPC, numpy, and pandas libraries
  7. Copy xxxx.py into the python directory

## Running the Simulation #####

  1. Launch KSP from CKAN
  2. Create a new sandbox game
  3. Click on launchpad and select the "xxxx" ship
  4. Open the in-game kRPC display and select create/start server
  5. Go to python and run the xxxx.py file
  6. Watch the simulation in KSP
  7. Telemetry data is saved as a pandas dataframe in python
  8. Plot telemetry by loading the dataframe and plotting with matplotlib
