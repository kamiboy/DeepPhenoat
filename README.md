# DeepPhenoat
The python script DeepPhenoat.py should be run using python version 3.
It requires several modules, such as numpy and torch.
It uses a module called BEDBUG for loading genotypes from bed/bim/fam files. This module is developed by me and included for you. The module file bedbug.py should be placed in the same folder as DeepPhenoat.py

It is best to run DeepPhenoat.py using the following command: 
python -u DeepPhenoat.py | tee AI.log
This way it will print out progress to the terminal, as well as save all printed output into the log file AI.log. In case anything goes wrong or there is need for any tweaks I it is crucial that I receive this file as it contains crucial diagnostics information to help me track down any potential issues. 

There are a number of variables in the start of DeepPhenoat.py file that needs to be set according to your local environment.

These include the work folder "workdir" which is where the program expects to find all necessary data, except for genotypes.
The genotype folder "snpdir" is where the program expects to find all bed/bim/fam genotype files.

The program is written to handle two differen kinds of bed/bim/fam file names. One where all the genotypes for all chromosomes are in one big file.
If that is the case set "snpdir" to point the location of the file, and "chr_file" to the name of the file without the file type (the stuff after the last .).
So if your bed file is named /genotypes/snps.bed, set "chr_file" equal to "snps", set "chr_file_postfix" euqal to None and snpdir to "/genotypes/", ending all folder locations with / is important, so don't forget.
Example:
snpdir= "/genotypes/"
chr_file = "snps"
chr_file_postfix = None

If, on the other hand, all your genotypes are in a separate file according to chromosome, do as follows.
If the files are calles something akin to '/genotypes/snps_chr1_rev4.bed', set "chr_file" equal to "snps_chr", set "chr_file_postfix" euqal to  "_rev4" and snpdir to "/genotypes/", ending all folder locations with / is important, so don't forget.
Example:
snpdir = "/genotypes/"
chr_file = "snps_chr"
chr_file_postfix = "_rev4"

If you have no postfix in your chromosomal bed files, e.g. '/genotypes/snps_chr1.bed' then set set "chr_file" equal to "snps_chr", set "chr_file_postfix" euqal to ""
Example:
snpdir = "/genotypes/"
chr_file = "snps_chr"
chr_file_postfix = ""

If you are using a completely different naming scheme for your genotypes, please let me know and I'll try recoding the program accordingly.

The variable "phenotypes_file" denotes the name of the file with all your phenotypes. A dummy example file is provided for reference.

The file is expected to have a header row, whose contents are not important, as the first rows is ignored during loading.
The remaining rows are expected to contain 3 tab separated columns.
The first column is for the ID of each case.
The second column is a string denoting the antigen in question, e.g. Fya
The third column is an integer that denotes the antigen status for the case, either 0 for negative, or 1 for positive. No other values are expected in this column, therefore please remove any NAs or the like prior to running the program.

The variable "phenotypes_info_file" denotes the file with necessary information on the phenotypes to be tested and or trained on.
This file is provided for you named: phenotypes_info.tsv

Due to you having unexpectedly very few phenotypes the AI training might not perform very well as it is in need of large quantities of data. If that is the case we might have to judge training based the select number of snps you have identified as most important for each phenotype.
In that case you can provide a variants file for training. These are to be called AI.training.variants.?.aec for training the autoencoder AI model, or AI.training.variants.?.mlp for the multiplayer perceptron AI model. The ? should be replaced with the phenotype in question, so for Fya the files would be called:
AI.training.variants.Fya.mlp
AI.training.variants.Fya.aec

In your case the contents of both files can just be identical, so make one file and make a copy for the other.

The files are expected to have a header row, whose content does not matter as it is ignored.
The remaining rows are expected to have four columns.
The first column denotes the chromosome of the variants, e.g. 1 for chromosme 1, X for chromosome X
The second column is an integer denoting the position of the variant within the chromosome, e.g. 15920497
The third column denotes the reference allele of the variant e.g G
The third column denotes the alternate allele of the variant e.g C
Example:
1	15920497	G	C

The program can be run with two different modes enabled. Training (perform_trining) or prediction (perform_training). The variables should be set to True or False to enable or disable them. To begin with we should focus on getting prediction to work.

In Prediction mode the program will load phenotypes from the phenotype file and will attempt to generate phenotypes based on existing AI models for all phenotypes in the provided phenotypes_info file. It will calculate the accuracy of the prediction in comparison to the provided phenotypes and save the results into AI.evaluation.prediction.?.aec and/or AI.evaluation.prediction.?.mlp files

In Training mode the program will load phenotypes from the phenotype file and will attempt to train AI models for all phenotypes in the provided phenotypes_info file. If the phenotypes file is not None, then all provided phenotypes will be loaded, and after prediction they will be compared to the predicted phenotypes and the results will be written to the AI.evaluation.training.?.aec and/or AI.evaluation.training.?.mlp files

There are a number of other parameters that can be set, but they should not be necessary to mess with unless we run into problems.

Finally, let it be said that this program is over 1500 lines of code, and while it has been tested to the best of my abilities, since python is a runtime interpreted language there is a good chance that there are edge cases that I have not been able to catch. If the program crashes please provide me a copy of the log file and I'll do my best to fix any bugs or unexpected issues in a speedy fashion. Thank you for your patience.
