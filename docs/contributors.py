
import requests
import pandas as pd
import json
import textwrap
from datetime import date


url = "https://api.github.com/orgs/ModECI/repos"

response= requests.get(url=url, auth= ('<github_username>', '<github_token>') )

json_data = response.json()

df= pd.DataFrame(json_data)

repo_name= df['name']
html_url= df['html_url']
repo_url= dict(zip(repo_name, html_url))

list_url= list(df['contributors_url'])
list_range = len(list_url)

empty_list=[]
for i in range(list_range):
    url= list_url[i]
    data= requests.get(url=url)
    empty_list.append(data.json())
    
con_json= []
for item in empty_list:
    for i in item:
        con_json.append(i)
        
df1= pd.DataFrame(con_json)

<<<<<<< HEAD
per_info= list(df1['url'].unique())
len_per_info= len(per_info)

empty_list=[]
for i in range(len_per_info):
    url= per_info[i]
    data= requests.get(url=url)
    empty_list.append(data.json())

df2= pd.DataFrame(empty_list)
df2['name']= df2['name'].fillna('Name Not Available')
name= df2['name']
login= df2['login']
url_html= df2['html_url']
url_id= df2['id']

login_html= list(zip(name, login,url_html))
zip_dict= dict(zip(url_id,login_html))



if 49699333 in zip_dict:
    del zip_dict[49699333]
      
=======
login= list(df1['login'].unique())
url= list(df1['html_url'].unique())
login_url= dict(zip(login, url))

if 'dependabot[bot]' in login_url:
    del login_url['dependabot[bot]']
    
>>>>>>> f0fb71abfcaf53536d19096909804243f52fa974
file= 'sphinx/source/api/Contributors.md'
with open(file, 'w') as f:
    print(textwrap.dedent(
        """\
        (ModECI:contributors)=
        
        # ModECI contributors

<<<<<<< HEAD
        This page list names and github profiles of contributors to the various ModECI repositories, listed in no particular order.
        This page is generated periodically and the most recent was on {}.""".format(date.today())), file=f)
    
    print("", file=f)

    for key, val in zip_dict.items():
        print("- {} ([@{}]({}))".format(val[0], val[1], val[2]), file= f)
        
=======
            This page lists contributors to the various ModECI repositories, listed in no particular order.
            This file is generated periodically, the most recent was on {}.""".format(date.today())), file=f)
    
    print("", file=f)

    for key, val in login_url.items():
        print("- [@{}]({})".format(key, val), file=f)

>>>>>>> f0fb71abfcaf53536d19096909804243f52fa974
    print(textwrap.dedent(
        """
        ## Repositories

        """
    ), file=f)

    for key, val in repo_url.items():
        print("- [{}]({})".format(key, val), file=f)


