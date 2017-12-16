#--------------------------------------------------------------------
# prediction.py:  Predicting County Level Cost Differences for Treating
#                 Chronic Obstructive Pulmonary Disease
# Jonathan Lin, Michael Smith, Aileen Wang
# Stanford University
# jolituba@stanford.edu,  msmith11@stanford.edu,  aileen15@stanford.edu
# December 15, 2017
#---------------------------------------------------------------------
import matplotlib.pyplot as plt
import csv
from util import *
import datetime
from optparse import OptionParser

#---------------------------------------------------------------------------------
# Helper function to handle 'NaN' and 'Infinity' Input values
#----------------------------------------------------------------------------------
def getvalue(val):
    if val == 'NaN' or val == 'Nan':
        return '0'
    elif val == 'Infinity':
        return '100'
    else:
        return val
#----------------------------------------------------------------------------------
# Extracting 'analysis_value' column values from the input csv files and merging
# into one csv file
#----------------------------------------------------------------------------------
def processCSVfiles(files, csv_outputfile):      
    with open(files[0], 'rb') as input1:
        reader1 = csv.reader(input1)
        records1 = list(reader1)
    data = []
    output_keys = ['year', 'fips', 'county', 'state', 'urban']
    output_keys.append(files[0][0:8])
    
    keys1 = records1[0]
    data.append(output_keys);
    urban_index = output_keys.index('urban')
    for i in range(1, len(records1)):
        row = []       
        for j in range(0, len(keys1)):            
            if keys1[j] in output_keys or keys1[j] == 'analysis_value':
               row.append(getvalue(records1[i][j]))
        data.append(row)
        if data[i][urban_index] == 'Urban':
            data[i][urban_index] = 1
        else:
            data[i][urban_index] = 0
            
    for count in range(1, len(files)):
        with open(files[count], 'rb') as input2:
            reader2 = csv.reader(input2)
            records2 = list(reader2)
        fips_index1 = records1[0].index('fips')
        fips_index2 = records2[0].index('fips')
        output_keys.append(files[count][0:8])       
        keys2 = records2[0]                   
        col = []
        for i in range(1, len(records1)):
            col.append(records1[i][fips_index1])
        for i in range(1, len(records2)):
            if i < len(records2) and records2[i][fips_index2] in col:
                idx1 = col.index(records2[i][fips_index2])+1
                idx2 = len(records2[i])-1
                if idx1 < len(data):
                    data[idx1].append(getvalue(records2[i][idx2]))
        i = 1
       
        while i < len(data):
            if len(data[i]) <= len(output_keys)-1:
                del data[i]
            else:
                i = i + 1
      
    with open(csv_outputfile, 'wb') as output:
        writer = csv.writer(output)
        writer.writerows(data)

#----------------------------------------------------------------------------------------
# Calculating the percentage increase/decrease by comparing the copd_apc column values of the
# next year with that of current year and adding 'apc_percent_diff' column into the current
# year csv file
#-----------------------------------------------------------------------------------------        
def calculatePercent(files, csv_outputfile):
    with open(files[0], 'rb') as input1:
        reader1 = csv.reader(input1)
        records1 = list(reader1)
    data = []
        
    keys1 = records1[0]
    keys1.append("apc_percent_diff")
    data.append(keys1)
    for i in range(1, len(records1)):
        row = []       
        for j in range(0, len(keys1)-1):                        
            row.append(records1[i][j])
        data.append(row)
               
    with open(files[1], 'rb') as input2:
        reader2 = csv.reader(input2)
        records2 = list(reader2)
        fips_index2 = records2[0].index('fips')
        fips_index1 = records1[0].index('fips')
        apc_index1 = records1[0].index('copd_apc')
       
        keys2 = records2[0]            
        col = []
        for i in range(1, len(records1)):
            col.append(records1[i][fips_index1])
        for i in range(1, len(records2)):
            if i < len(records2) and i < len(records1) and records2[i][fips_index2] in col:
                idx1 = col.index(records2[i][fips_index2])+1
                idx2 = len(records2[i])-1               
                diff_perc = 100*(float(records2[i][idx2]) - float(records1[i][apc_index1])) / float(records1[i][apc_index1])
                data[idx1].append(str(diff_perc))
        i = 1       
        while i < len(data):
            if len(data[i]) <= len(keys1)-1:
                del data[i]
            else:
                i = i + 1
      
    with open(csv_outputfile, 'wb') as output:
        writer = csv.writer(output)
        writer.writerows(data)

#----------------------------------------------------------------------------------------
# Replacing the 'copd_apc' column of current multi-year period with the 'apc_percent_diff'
# column of next multi-year period csv file
#-----------------------------------------------------------------------------------------     
def replace_copd_apc(files, csv_outputfile):
    with open(files[0], 'rb') as input1:
        reader1 = csv.reader(input1)
        records1 = list(reader1)
    data = []
           
    keys1 = records1[0]    
    data.append(keys1)    
    for i in range(1, len(records1)):
        row = []       
        for j in range(0, len(keys1)):                        
            row.append(records1[i][j])       
        data.append(row)

    with open(files[1], 'rb') as input2:
        reader2 = csv.reader(input2)
        records2 = list(reader2)
   

    fips_index1 = records1[0].index('fips')
    fips_index2 = records2[0].index('fips')    
    apc_index1 = records1[0].index('apc_percent_diff')
    apc_index2 = records2[0].index('copd_apc')
             
    col = []
    for i in range(1, len(records1)):
        col.append(records1[i][fips_index1])
    for i in range(1, len(records2)):
        if i < len(records1) and i < len(records2) and records2[i][fips_index2] in col:
            idx1 = col.index(records2[i][fips_index2])+1                                            
            data[idx1][apc_index1] = records2[i][apc_index2]            
        
    with open(csv_outputfile, 'wb') as output:
        writer = csv.writer(output)
        writer.writerows(data)

#-----------------------------------------------------------------------------------------------------
# Extracting 'Unemployment Rate (5y Avg.)' and 'Percent Below Federal Poverty Level (5y Avg.)' column
# values from "MMD County Profiles Data Download.csv" file and adding to the "copd_multi_year.csv" file
#-----------------------------------------------------------------------------------------------------    
def addProfiledata(files, csv_outputfile):
    with open(files[0], 'rb') as input1:
        reader1 = csv.reader(input1)
        records1 = list(reader1)
    data = []
        
    keys1 = records1[0]   
    keys1[len(keys1) - 1] = "apc_percent_diff"
    keys1.insert(len(keys1) -1, "below_poverty")
    keys1.insert(len(keys1) - 1, "unemp_rate")
    data.append(keys1)
    for i in range(1, len(records1)):
        row = []       
        for j in range(0, len(keys1)-2):                        
            row.append(records1[i][j])
        data.append(row)
               
    with open(files[1], 'rb') as input2:
        reader2 = csv.reader(input2)
        records2 = list(reader2)
        fips_index2 = records2[0].index('fips')
        fips_index1 = records1[0].index('fips')
        value_index2 = records2[0].index('value')        
        col = []
        for i in range(1, len(records1)):
            col.append(records1[i][fips_index1])
            
        for i in range(1, len(records2)):
            if i < len(records2) and records2[i][1] == '2015' and records2[i][0] == 'Unemployment Rate (5y Avg.)' \
               and records2[i][fips_index2] in col:           
                idx1 = col.index(records2[i][fips_index2])+1                                        
                data[idx1].insert(len(data[idx1]) -1 , str(float(records2[i][value_index2])*100))

        for i in range(1, len(records2)):
            if i < len(records2) and records2[i][1] == '2015' and records2[i][0] == 'Percent Below Federal Poverty Level (5y Avg.)' \
               and records2[i][fips_index2] in col:           
                idx1 = col.index(records2[i][fips_index2])+1                                        
                data[idx1].insert(len(data[idx1]) -1 , str(float(records2[i][value_index2])*100))   
        i = 1       
        while i < len(data):
            if len(data[i]) <= len(keys1)-2:
                del data[i]
            else:
                i = i + 1     
    with open(csv_outputfile, 'wb') as output:
        writer = csv.writer(output)
        writer.writerows(data)
#----------------------------------------------------------------------------------------
# Merging all single year csv files into a combined csv file 'copd_single_year.csv'
#-----------------------------------------------------------------------------------------  
def MergeAllfiles(files, csv_outputfile):      
    with open(files[0], 'rb') as input1:
        reader1 = csv.reader(input1)
        records1 = list(reader1)
    data = []
       
    for i in range(0, len(records1)):
        row = []       
        for j in range(0, len(records1[0])):                       
            row.append(getvalue(records1[i][j]))
        data.append(row)       
            
    for count in range(1, len(files)):
        with open(files[count], 'rb') as input2:
            reader2 = csv.reader(input2)
            records2 = list(reader2)       
        for i in range(1, len(records2)):
            row = [] 
            for j in range(0, len(records2[0])): 
                row.append(records2[i][j])
            data.append(row) 
       
    with open(csv_outputfile, 'wb') as output:
        writer = csv.writer(output)
        writer.writerows(data)

def main(argv):
    op = OptionParser()   
    op.add_option("-Y", "--year", dest="year",
                  default="2012", type="string",
                  help="Year: 2012, 2013, 2014, 2012-2014" )
    op.add_option("-A", "--mergeall", dest="merge_all",
                  default=False,    
                  help="merge mode: True for merging all single csv files and False for processing other csv files.")
    op.print_help()
    (opts, args) = op.parse_args()    
    year = opts.year
    if opts.merge_all == True:
        files =["copd_perc_" + year + ".csv", "copd_perc_" + str(int(year)+1) + ".csv", "copd_perc_" + str(int(year)+2) + ".csv"]
        MergeAllfiles(files, "copd_single_year.csv")
        return
 
    if len(year)== 4:       
        files = ["copd_apc_" + year + ".csv", "copd_edv_" + year + ".csv", "copd_hos_" + year + ".csv", "copd_pre_" + year + ".csv",\
                 "copd_pqi_" + year+ ".csv", "asth_pre_" + year + ".csv", "toba_pre_" + year + ".csv"]
        processCSVfiles(files, "copd_merge_" + year + ".csv")    
        files = ["copd_merge_" + year + ".csv", "copd_apc_"+ str(int(year) + 1) + ".csv"]
        calculatePercent(files, "copd_perc_" + year + ".csv")        
    else:
        files = ["copd_edv_" + year + ".csv", "copd_hos_" + year + ".csv", "copd_pre_" + year + ".csv", "copd_prs_" + year + ".csv", \
                 "copd_pqi_" + year + ".csv", "asth_pre_" + year + ".csv", "toba_pre_" + year + ".csv", "copd_apc_" + year + ".csv"]
        processCSVfiles(files, "copd_merge_" + year + ".csv")
        files = ["copd_merge_" + year + ".csv", "MMD County Profiles Data Download.csv"]        
        addProfiledata(files, "temp.csv")       
        files = ["temp.csv", "copd_merge_" + "2012-2015" + ".csv"]
        replace_copd_apc(files, "copd_multi_year.csv")

if __name__ == '__main__':
    main(sys.argv)



