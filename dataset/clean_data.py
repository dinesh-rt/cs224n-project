import pandas as pd
import re
from urllib.parse import urlsplit
from github import Github
import os
import logging
import traceback


def parse_url(url):
    text_url, line = url.split("#")
    start_str,end_str = re.split("-", line)
    start = re.split("\D+",start_str)[1]
    end = re.split("\D+",end_str)[1]
    url=urlsplit(text_url).path
    user,repo,sha,file_path = re.findall('([\w-]+)/([-/.\w]+)/blob/(\w+)/([\w/.-]+)', url)[0]
    return user,repo,sha,file_path,start,end 

def get_token_from_file():
    with open('token.txt' ,'r') as f:
        return f.read().strip()

def check_url(url):
    user,repo_name,sha,file_path,start,end = parse_url(url)
    print(f"user: {user}, repo: {repo_name}, file_path: {file_path}, start: {start}, end: {end}")
    token = get_token_from_file()
    g = Github(token)
    print(token)
    print(f'{user}/{repo_name}/{file_path}')
    repo = g.get_repo(f'{user}/{repo_name}')
    contents = repo.get_contents(file_path)

def run_clean_data(bVanilla):
    #load the annotationStore.csv
    ann_df = pd.read_csv("annotationStore.csv")

    # filter python
    python_df = ann_df[ann_df["Language"] == "Python"]

    # remove duplicates
    python_df = python_df.drop_duplicates(subset=['GitHubUrl'])

    if not bVanilla:
        for row in python_df.itertuples():
            print(row, row.Index)
            url = row.GitHubUrl
            try:
                check_url(url)
            except:
                #logging.error(traceback.format_exc())
                ## delete from data frame
                print(row.Index)
                python_df.drop(index=row.Index, inplace=True)
    ## write data to file
    python_df.to_csv('annotations.csv', index=False)

if __name__ == "__main__":
    run_clean_data()



