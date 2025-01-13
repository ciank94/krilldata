# krilldata
package for training and predicting krill density using multivariable regression methods

## Source code (krilldata/)
- readKrillBase.py handles reading and preprocessing data from krillbase
- dataFusion.py handles preprocessing of bathymetry and SST data, and fusion with krill observations
- krillTrain.py handles training of regression models
- krillPredict.py handles prediction of krill density using trained models

## Usage (usage/)
- train.py contains example workflow for training model to predict krill density
- mapKrill.py contains example workflow for mapping predicted krill density on input features
