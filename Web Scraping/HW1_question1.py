##Question 1: Yahoo Stock Price 100

#import urllib.request
import numpy as np
import urllib
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep
import sys
#import requests
import random


### Scrape symbols from wiki page###
col_list = []

#specify the url
url = "https://en.wikipedia.org/wiki/S%26P_100"
html = urllib.request.urlopen(url).read()
    
soup = BeautifulSoup(html, 'lxml') #parse the HTML as a string
table = soup.find_all('table')[2]  


rows = table.findAll('tr')  #locate each symbol and company name and add to list
for row in rows:
    for col in row.findAll("td"):
        col_list.append(col.text)
        
col_np = np.array(col_list)
sym_idx = list( range(0, len(col_list)-1,2) )   #symbols are indexed as even number in list
#com_idx = list( range(1, len(col_list),2) )
symbol = np.empty([ len(sym_idx), 1])
company =  np.empty([ len(sym_idx), 1])
symbol = col_np[sym_idx]
company = np.delete(col_np, sym_idx)    #get company names
     
cols ={'Sym':symbol, 'Com':company}
df=pd.DataFrame(cols)
df=df[['Sym', 'Com']]
print(df.head())

corr = df['Sym'] == 'BRK.B'   #correct symbol
df.loc[corr, 'Sym'] = 'BRK-B'




###Selenium download modify and output data###

#for test use
'''
path_to_chromedriver = 'C:\Programing\Python3\chromedriver.exe' # change path as needed
browser = webdriver.Chrome(executable_path=path_to_chromedriver)
url = 'https://finance.yahoo.com/quote/XOM/history?p=XOM'
browser.get(url)
'''

def srch_element(driver, x_path, wait_time):  #function to locate elements by xpath
    WebDriverWait(driver, wait_time).until(
            lambda s: s.find_element_by_xpath(x_path).is_displayed())
    res = driver.find_element_by_xpath(x_path)
    return res


#test function srch_element
'''
xom_path ='//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[2]/span[2]/a/span'
xom_dl = srch_element(browser, xom_path, 15)
xom_dl.click()
'''


path_to_chromedriver = 'C:\Programing\Python3\chromedriver.exe' # change path as needed
chrome_options = webdriver.ChromeOptions()
prefs = {'download.default_directory': 'C:\sp100data'}    #set download dir for chrome
chrome_options.add_experimental_option('prefs', prefs)    
browser = webdriver.Chrome(executable_path=path_to_chromedriver,
                           chrome_options=chrome_options)
dl_path ='//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[2]/span[2]/a/span'   #xpath for download element

#for test use
'''
url = 'https://finance.yahoo.com/quote/XOM/history?p=XOM'
browser.get(url)
data = srch_element(browser, dl_path, 15)
data.click()    
raw_data = pd.read_csv('C:\sp100data\XOM'+'.csv')
#n = raw_data.shape[0]
raw_data['Symbol'] = ['XOM'] 
cols = raw_data.columns.tolist()
cols = cols[-1:] + cols[:-1]
md_data = raw_data[cols]
'''


test = ['SO','SPG','T']  #a small test set
out_data = pd.DataFrame()  #initiate the output with an empty data frame




for symbol in df['Sym']:   
#loop through symbols, dowload data, read into enviroment, add column and output
    
    url = 'https://finance.yahoo.com/quote/' + symbol + '/history?p=' + symbol
    browser.get(url)
#    frequency = srch_element(browser, fre_path, 15)
#    frequency.click()
    data = srch_element(browser, dl_path, 20)
    data.click()
    sys.stdout.flush()
    #sleep(5)
    sleep(2+random.uniform(1,3))
    print(symbol + " data downloaded" )
    rd_dir = 'C:\sp100data\\' + symbol + '.csv'
    raw_data = pd.read_csv(rd_dir)
   # n = raw_data.shape[0]
    raw_data['Symbol'] = symbol 
   # cols = []
    cols = raw_data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    md_data = raw_data[cols]
    out_data = out_data.append(md_data)
    print(md_data.head())

out_data.to_csv(r'''C:\Users\Bernard\Desktop\big data\HW1\sp100.csv''')  
