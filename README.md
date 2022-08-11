# Active inference, selective attention, and the cocktail party problem
Code for the following paper: Holmes, E., Parr, T., Griffiths, T. D., &amp; Friston, K. J. (2021). Active inference, selective attention, and the cocktail party problem. Neuroscience &amp; Biobehavioral Reviews, 131, 1288-1304. https://doi.org/10.1016/j.neubiorev.2021.09.038

## Background and purposes
This code accompanies a paper on "Active inference, selective attention, and the cocktail party problem" (full citation above). Please refer to the paper for an explanation of the methodology and goals. The purpose of distributing this code on GitHub is so that interested readers can run the code and recreate the simulations. The code generate figures demonstrating key principles from the paper (e.g., effects of changing model parameters on error rates and types; effects of the length of the preparatory interval on reaction times and simulated EEG responses).

## License
This project is licensed under the GNU General Public License v3.0; see the [LICENSE](LICENSE) file for details. The code was written by Emma Holmes, although note that some code has been adapted from existing code in the [SPM toolbox](https://www.fil.ion.ucl.ac.uk/spm/).

This project can be cited using the following DOI: placeholder
  
## Getting started
### Prerequisites
The code was written using MATLAB. Users are required to have [SPM12](https://www.fil.ion.ucl.ac.uk/spm/) in their MATLAB path. The code was written for use with SPM12 version 8157. It was tested using MATLAB 2017b, although it should theoretically be compatible with any version of MATLAB that is capable of running SPM12: please refer to the [SPM website](https://www.fil.ion.ucl.ac.uk/spm/) for details.

### Running the code
Different scripts within this repository recreate different aspects of the paper and can be run individually, as desired.

## Scripts
The following functions are called from other scripts in the directory:
- CRM_MDP_SpecifyModel.m: This routine uses active inference for Markov decision processes to illustrate attending to one phrase in a mixture during cocktail party listening. It is used in Sections 2 and 3 of the paper to simulate the optimal model and changes to parameters that impair performance. Higher-level scripts in this repository call this routine with specific (combinations of) parameters.
- CRM_MDP_SpecifyModelPrep.m: This routine introduces a cue-target interval between the visual cue and the phrases. It is used in Sections 4 and 5 of the paper to simulate preparatory attention and generate reaction times and EEG responses under different temporal models. Higher-level scripts in this repository call this routine with specific (combinations of) parameters.
- spm_MDP_VB_LFP_edit.m: This routine is edited from the SPM12 script 'spm_MDP_VB_LFP.m' and adds additional capabilities for plotting simulated electrophysiological responses. It is called by CRM_MDP_SpecifyModelPrep.m when disp_plots is set to true (i.e., for the simulations in Section 5 of the paper).

The following are higher-level scripts, which can be called to reproduce elements of the paper:
- CRM_MDP_OptimalModel.m: This script runs the optimal model that produces high performance (Section 2 of the paper).
- CRM_MDP_ManipulateParametersIndividually.m: This script changes particular parameters individually, to examine their impact on performance (Section 3 of the paper).
- CRM_MDP_ManipulateParametersCombined.m: This script changes precision parameters for attended and unattended words in combination, to examine their relative contributions to performance (Section 3 of the paper).
- CRM_MDP_TestTemporalFunctions.m: This script generates models with five different temporal functions, to examine their effects on reaction times (Section 4 of the paper) and simulated EEG responses (Section 5 of the paper).

## Output files
Example output files are included in the 'outputs' folder.
