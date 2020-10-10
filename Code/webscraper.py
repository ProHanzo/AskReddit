from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import datetime
import pandas as pd
  

#%% FUNCTIONS
# FILTER COMMENTS
def filter_comments(text):
    num=""
    #22.2k
    if text[2] == ".":
        num = int(text[0]+text[1]+text[3]) * 100
        comments.append(num)
        return
    #2.2k
    if text[1] == ".":
        num = int(text[0] + text[2]) * 100
        comments.append(num)
        return
    #742 comments
    else:
        i = 0
        while text[i] != " ":
            num += text[i]
            i += 1
        
        comments.append(int(num))
        return
# FILTER POST TIME
def filter_datetime(convert, period):
    if(period == "year"):
        current_time = 0
    if(period == "hours"):
        a= datetime.datetime.now()
        b = a - datetime.timedelta(hours=convert)
        current_time = b.hour
    if(period == "months"):
        a = datetime.datetime.today()
        b = a - datetime.timedelta(days=convert)
        current_time = b.weekday()
    return current_time

# FILTER POST TIME
def filter_posttime(timetext, period):
    if timetext[1] == " ":
        convert = int(timetext[0])
        
        current_time = filter_datetime(convert, period)
        
        timestamp.append(current_time)
        return 
    else:
        timecalc = timetext[0] + timetext[1]
        convert = int(timecalc)
        
        current_time = filter_datetime(convert, period)
        
        timestamp.append(current_time)
        return

# FILTER UP VOTES
def filterupvotes(text,upvotes):
    if(text == "•"):
        upvotes.append(1)
        return
    length = len(text)-1
    if(text[length] == "k"):
        temp = ""
        for i in range(length):
            temp += text[i]
        temp = float(temp)
        temp = int(temp) * 1000
        upvotes.append(temp)
    else:
        text = int(text)
        upvotes.append(text)
#%%

# location of the chrome driver
driver = webdriver.Chrome("#####")

title=[]
upvotes=[] 
timestamp=[]
comments=[]
driver.get("https://www.reddit.com/r/AskReddit/top/?t=year")

elem = driver.find_element_by_tag_name("body")

def scrolldown(num):
    no_of_pagedowns = num

    while no_of_pagedowns:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        no_of_pagedowns-=1
        print(no_of_pagedowns)
        
scrolldown(400)
"""
TODO: Load this project into git 
"""
content = driver.page_source
soup = BeautifulSoup(content, 'html.parser')

# Set reddit to Top of the WEEK
for a in soup.findAll('div', attrs={'tabindex':'-1'} ):     
    # skips promoted threads
    if (a.findAll('span', attrs={'class':'_2oEYZXchPfHwcf9mTMGMg8'})): 
        print("stop1")         
        continue            
    # checks if both title & upvotes exist
    elif(a.findAll('a',href=True, attrs={'class':'SQnoC3ObvgnGjWt90zD9Z _2INHSNB8V5eaWp4P0rY_mE'})
       and a.findAll('div', attrs={'class':'_1rZYMD_4xY3gRcSS3p8ODO'})
       and a.findAll('a', attrs={'class':'_3jOxDPIQ0KaOWpzvSQo-1s'})
       and a.findAll('span', attrs={'class':'FHCV02u6Cp2zYL0fhQPsO'})):
        
        _title=a.find('h3', attrs={'class':'_eYtD2XCVieq6emjKBH3m'})
        title.append(_title.text)
        
        _upvotes=a.find('div', attrs={'class':'_1rZYMD_4xY3gRcSS3p8ODO'})
        filterupvotes(_upvotes.text,upvotes)
        
        _timestamp=a.find('a', attrs={'class':'_3jOxDPIQ0KaOWpzvSQo-1s'})
        filter_posttime(_timestamp.text, "year")
        
        _comments=a.find('span', attrs={'class':'FHCV02u6Cp2zYL0fhQPsO'})
        filter_comments(_comments.text)
          


df_add = pd.DataFrame({'Title':title, 'Upvotes':upvotes, 'Time(military)':timestamp, '# of Comments':comments}) 
df_add.to_csv('./askreddit_data_month.csv',mode='a', index=False, encoding='utf-8',header=False)

