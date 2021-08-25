This directory contains examples to run with the translation from MDF to ONNX 
(code will be in [interfaces](/src/modeci_mdf/interfaces/mdf_to_onnx)).

#Contents of this directory
".json" files: MDF representations

"-m2o.onnx" files: Generated ONNX files. 

".png" files: Netron visualization of the ONNX files.

#Translating from MDF to ONNX

<mdf_to_onnx> -i <.json> -od <output directory where the .onnx file is saved> 

##Example 1 : ab.json

The MDf file (ab.json) is generated from a Pytorch module that is exported to ONNX and which is converted to the MDF 
representation.

The application has two nodes connected serially.
