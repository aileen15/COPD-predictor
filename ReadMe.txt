1. Python package dependence:
   To run predication.py and Sentiment.py, the following package must be installed
   sklearn
   matplotlib
   numpy

2. Instruction for running Preprocessing.py:    
  (i)  By default, no command line arguments need be provided
       the default values:       
       year = '2012'
       merge_all = False              
  (ii)For input non-default value, please follow the command line instruction.
       For example:
       Preprocessing.py:  --year="2013" 
  Note: You need generate each year feature data set first, then merge all the indvidual year data into copd_single_year.csv by providing command argument
       --mergeall = True
        

3. Instruction for running predication.py:
   (i) By default, no command line arguments need be provided
       the default values: 
       binary_classify = True
       single_year = True
       year = "" (for all years from 2012 to 2014)       
   (ii)Assume "copd_single_year.csv" or "copd_multi_year.csv" exists 
  (iii)For input non-default value, please follow the command line instruction.
       For example:
       predication.py --binary=False --single_year=False 
  (iv) While running the prediction program, a few graphic windows will be displayed. The graphic window must be manully closed before the program   
       can continue.

