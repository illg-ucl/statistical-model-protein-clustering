# statistical-model-protein-clustering

Statistical model of random protein clustering on a cell surface, based on Poisson distributions. 

Protein molecules can be fluorescently labelled and imaged in experiments using fluorescence microscopy techniques. Analysis of the images yields the experimental cluster size (diameter in nm) and the number of molecules for each measured cluster. The probability distribution for the number of molecules in a cluster of a given size can be modelled as a Poisson distribution considering a relevant average cell-surface density of molecules (using this code). The modelled distributions can be compared to the experimental data.

This theoretical model is based on Poisson probability distributions that are specific to given cluster sizes. The model can be used to determine whether measured numbers of molecules in clusters are consistent with a random distribution of receptors on the cell surface (as in the paper below). The model can be applied to find out if the clustering of receptor proteins observed on cell surfaces is due to random chance colocalisation, consistent with the relevant Poisson distributions of receptors, or whether the clustering observed is due to other reasons, for example, due to the binding of virus particles to the cell surface which causes receptors to cluster around them.

**Related academic publication**: 

Single-Molecule Super-Resolution Imaging of T-Cell Plasma Membrane CD4 Redistribution upon HIV-1 Binding, Y. Yuan, C. A. Jacobs, I. Llorente Garcia, P. M. Pereira, S. P. Lawrence, R. F. Laine, M. Marsh, R. Henriques, Viruses 13, 142 (2021). https://www.mdpi.com/1999-4915/13/1/142. 

Statistical modelling of the surface distribution (number of molecules per cluster) for CD4 cell-surface receptor proteins in the absence of HIV viruses (left below) and in the presence of HIV viruses (right below): 

<p align="center">
  <img src="https://github.com/illg-ucl/statistical-model-protein-clustering/blob/main/statistical_model_receptor_clustering_1.png" width=60% height=60%>
  
###### (A) Model schematic. Top of panel A: counting CD4 molecules (blue) in membrane areas of different sizes (small (pink), medium (green) or large (yellow)) compared to a theoretical model based on an averaged Poisson distribution that corresponds to a random distribution of receptors on the cell surface and to multiple measurements in membrane areas (clusters) of different sizes (the black line is the average of Poisson distributions for different cluster sizes [pink, green, yellow]). Bottom of panel A: experimentally determined distribution: HIV binding alters the organisation of receptors, the occurrence of clusters of a certain number of molecules per cluster is altered (solid line) and the distribution differs from the expected averaged Poisson distribution (dotted line). (B,C) Comparison of modelled (‘Random’) and measured distributions of the numbers of molecules per cluster for untreated cells (B) and HIV-treated (bound) cells (C). The discrepancy between the predicted model and the experimental data in HIV-treated cells is highlighted in green shading in A and C. 
</p> 


# Relevant files

* **requirements.txt**: file containing all python package and version requirements.
* **Jupyter notebook (.ipynb)**: "cluster_stoichiometry_statistics_n_100.ipynb" contains the code with all outputs.
* **Python script (.py)**: "cluster_stoichiometry_statistics_n_100.py" same code as the Jupyter notebook but without outputs, saved as a Python script.
* **Input data**: (not provided) the code takes as input a spreadsheet with two columns: cluster diameter (nm) and number of molecules per cluster.


# Copyright and License

Copyright (c) 2020. Isabel Llorente-Garcia, Dept. of Physics and Astronomy, University College London, United Kingdom.

Released and licensed under a BSD 2-Clause License:

https://github.com/illg-ucl/statistical-model-protein-clustering/blob/main/LICENSE

This program is free software: you can redistribute it and/or modify
it under the terms of the BSD 2-Clause License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 2-Clause License for more details. You should have received 
a copy of the BSD 2-Clause License along with this program. 

Citation: If you use this software for your data analysis please acknowledge 
          it in your publications and cite as follows.
          
              -Citation example 1: 
               statistical-model-protein-clustering. (Version). 2020. Isabel Llorente-Garcia, 
               Dept. of Physics and Astronomy, University College London, United Kingdom.
               https://github.com/illg-ucl/statistical-model-protein-clustering. (Download date).
               
              -Citation example 2:
               @Manual{... ,
                title  = {statistical-model-protein-clustering. (Version).},
                author       = {{Isabel Llorente-Garcia}},
                organization = { Dept. of Physics and Astronomy, University College London, United Kingdom.},
                year         = 2020,
                url          = {https://github.com/illg-ucl/statistical-model-protein-clustering}
                }

