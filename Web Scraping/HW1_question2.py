##Question 2: Pubmed Publications

import numpy as np
import urllib
from bs4 import BeautifulSoup
import pandas as pd
import re


#print(aca_raw.columns)
aca_raw = pd.read_csv(r'''C:\Users\Bernard\Desktop\big data\HW1\NIHHarvard.csv''')

#extract columns
aca_data = aca_raw.loc[:, ['Activity', 'Contact PI / Project Leader']]  

#indexing specific rows as requrired
idx_f = aca_data.loc[:,'Activity'].str[0] == 'F'    
idx_t = aca_data.loc[:,'Activity'].str[0] == 'T'
idx = idx_f | idx_t
#extract data 
author = aca_data.loc[-idx,'Contact PI / Project Leader']    
#remove duplicate names. strip delete leading/trailing spaces
author_unique = np.unique(author.str.strip())      

def chk_abnormal_lst_name(name):    
#function to check names with abnormal last names
    
    #split string by ","
    name_comma_spt = name.split(sep = ",")    
    #get string before "," as last names. solit last names by space and count words
    lst_name_len = len( name_comma_spt[0].split() )   
    if lst_name_len >=2:   
        return True
    else:    #if more than 1 words in last name, then something is wrong, mark it
        return False


#get names with abnormal last names
ab_lst_name_bln =[]
for author in author_unique:   
    ab_lst_name_bln.append(chk_abnormal_lst_name(author))
    
#name list of abnormal last names
ab_name = author_unique[ab_lst_name_bln]    

#name list of normal last names
normal_name = author_unique[np.logical_not(ab_lst_name_bln)]    

def rm_initial_mid(name) :    
#function to remove middle name and initials
    
    name_spt = name.split()
    #name_wd_cnt = len(name_spt)
    name_no_mid_list = name_spt[0:2]
    name_no_mid = " ".join(name_no_mid_list)
    name_out = re.sub('\s[A-Z]{1}.?\s|\s[A-Z]{1}.?$', ' ', name_no_mid)
    return name_out.strip()

#delete midlle names and intitials
author_output = []
for au in normal_name:    
    m = len(au.split())
    if m >=3:
        au = rm_initial_mid(au)
    author_output.append(au)

author_name_lf = np.unique(author_output)

#check names that only differs by middle name
print((author_name_lf == author_output).all())    
#make sure all affliation values are the same
print(np.unique(aca_raw.iloc[:,0]))    


PI = "LIN, XIHONG"
aff = "Harvard"
pub_url = 'https://www.ncbi.nlm.nih.gov/pubmed/?term=' \
+ PI.replace(', ', '%2C+') + '%5BAuthor%5D+' + 'AND+' \
+ aff + '%5BAffiliation%5D'

PI = 'VAN VACTOR, DAVID L.'
pub_url = 'https://www.ncbi.nlm.nih.gov/pubmed/?term=' \
+ PI.replace(', ', '%2C+').replace(' ', '+') + '%5BAuthor%5D+' + 'AND+' \
+ aff + '%5BAffiliation%5D'

pub_html = urllib.request.urlopen(pub_url).read()
soup = BeautifulSoup(pub_html, 'lxml') # Parse the HTML as a string

part = soup.find('div', {'class': 'title_and_pager'})
info = part.find('h3', {'class': "result_count left"})
num_results = info.text

aff = "Harvard"
nums_pub = []


def cnt_pub(name, lst_normal = True):
#function that input names and output nmber of publications
    
    #if names are abnormal
    if lst_normal == True:    
        #establish link by last and first name
         name_srch = 'https://www.ncbi.nlm.nih.gov/pubmed/?term=' \
         + name.replace(', ', '%2C+') + '%5BAuthor%5D+' + 'AND+' \
         + aff + '%5BAffiliation%5D'    
    
    #if names are abnormal
    else:    
        #establish link by full name *including middle and initial)
        name_srch = 'https://www.ncbi.nlm.nih.gov/pubmed/?term=' \
        + name.replace(', ', '%2C+').replace(' ', '+') + '%5BAuthor%5D+' + 'AND+' \
        + aff + '%5BAffiliation%5D'  
    try:
        author_html = urllib.request.urlopen(name_srch).read()
        soup = BeautifulSoup(author_html, 'lxml') 
        part = soup.find('div', {'class': 'title_and_pager'})
        info = part.find('h3', {'class': "result_count left"})
        num_results = info.text    
        #print(num_results)
        num = num_results.split()[-1]    #locate the number only
    
    #if error, publication set to 1 (automatically entered paper page)
    except:   
        num = 1
        
    return num

#loop through normal names and get number of publications
normal_pub_num = []    
for aut_it in author_output:     
    normal_pub_num.append(cnt_pub(aut_it))

#loop through abnormal names and get number of publications
ab_pub_num = []
for aut_it in ab_name:     
    ab_pub_num.append(cnt_pub(aut_it, False))

normal_pub_data = pd.DataFrame({ 'Author' : author_output , 
                                'Number of Publications' : normal_pub_num})
normal_pub_data['Affliation'] = aff 

ab_pub_data = pd.DataFrame({ 'Author' : ab_name , 
                                'Number of Publications' : ab_pub_num})
ab_pub_data['Affliation'] = aff 

normal_pub_data.to_csv(r'''C:\Users\Bernard\Desktop\big data\HW1\Pubmed_normal.csv''')
ab_pub_data.to_csv(r'''C:\Users\Bernard\Desktop\big data\HW1\Pubmed_abnormal.csv''')






