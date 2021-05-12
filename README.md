MFA (Molecular Field Analysis) in asymmetric catalysis for Mac OS. As an example, the MFA of the asymmetric fluorination reactions (Ref 2) is described.

# Dependency

Python3, numpy, R, glmnet

Test environment : Mac OS 10.14, Python 3.7.4, numpy 1.16.4, R 3.6.0, glmet 2.0-18.

# Instruction

#### 1. Download

Download the folders and scripts. 

  - data
  - lib
  - output
  - descriptor.py
  - regression.R
  - visualization.py

#### 2. Calculations of molecular fields

```
python descriptor.py
```

Run the script as shown above, which generates descriptors (molecular fields) calculated from the intermediate structures (descriptor.txt).

#### 3. LASSO and Elastic Net regressions

```
Rscript regression.R
```

Run the script as shown above. This script implements Elastic Net regressions between the target variables and descriptors calculated above. For running the script, R package, [glmnet](https://cran.r-project.org/web/packages/glmnet/) is required. This script generates the file for the regression coefficient values to the "output" folder (coefficient.txt). 
#### 4. Visualization

```
python visualization.py
```

Run the script as shown above. This script visualizes the important structural information on the intermediate structures based on the regression coefficients. Running the script generates xyz files of the intermediate structures with the important structural information in the "output" folder. A few lines from the bottom of the xyz files correspond to the important structural information visualized. N/C atoms are the structural information corresponding to the positive coefficients that overlap(N)/do not overlap(C) with the intermediate structure. O/F atoms are the structural information corresponding to the negative coefficients that overlap(O)/do not overlap(F) with the intermediate structure. 

Opening the xyz files using [mercury](https://www.ccdc.cam.ac.uk/solutions/csd-system/components/mercury/) and editing the graphics afford the figures in the text of Ref 2. 

#  References

1. S. Yamaguchi*, T. Nishimura, Y. Hibe, M. Nagai, H. Sato, I. Johnston *J. Comp. Chem.* **2017**, *37*, 1825.
2. S. Yamaguchi*, M. Sodeoka *Bull. Chem. Soc. Jpn.* **2019**, *92*, 1701.





