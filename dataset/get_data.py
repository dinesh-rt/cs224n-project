from fast_ml.model_development import train_valid_test_split
import pandas as pd
import re
from urllib.parse import urlsplit
from github import Github
import os

def parse_url(url):
    text_url, line = url.split("#")
    start_str,end_str = re.split("-", line)
    start = re.split("\D+",start_str)[1]
    end = re.split("\D+",end_str)[1]
    url=urlsplit(text_url).path
    user,repo,sha,file_path = re.findall('(\w+)/(\w+)/blob/(\w+)/([\w/.]+)', url)[0]
    return user,repo,sha,file_path,start,end 

def get_token_from_file():
    with open('token.txt' ,'r') as f:
        return f.read().strip()

def download_file(url):
    user,repo_name,sha,file_path,start,end = parse_url(url)
    print(f"user: {user}, repo: {repo_name}, file_path: {file_path}, start: {start}, end: {end}")
    token = get_token_from_file()
    g = Github(token)
    print(token)
    print(f'{user}/{repo_name}/{file_path}')
    repo = g.get_repo(f'{user}/{repo_name}')
    contents = repo.get_contents(file_path)
    decoded = contents.decoded_content
    with open("temp.txt", 'wb') as f:
	    f.write(decoded)
    #read lines in a range
    with open("temp.txt", 'r') as f:
	    func = f.readlines()[int(start):int(end)]
	    print(func)
    os.remove("temp.txt")

def run_preprocess():
    df=pd.read_csv('queries.csv')

    #split the data into train, valid, test
    x_train, y_train, x_valid, y_valid, x_test, y_test = train_valid_test_split(df, target='query', train_size=0.70, valid_size = 0.15, test_size =0.15)

    #reset the index
    for data in [x_train, y_train, x_valid, y_valid, x_test, y_test]:
        data.reset_index(drop=True, inplace=True)

    #split dataset size
    print(f"train: {x_train.shape}, {y_train.shape}")
    print(f"valid: {x_valid.shape}, {y_valid.shape}")
    print(f"test: {x_test.shape}, {y_test.shape}")


    #iterate through the training queries
    for index,row in enumerate(y_train):
        if index == 0:
            print(f"query : {row}")

    #load the annotationStore.csv
    ann_df = pd.read_csv("annotationStore.csv")
    query = "priority queue"
    url_df = ann_df[(ann_df["Language"] == "Python") & (ann_df["Query"] == query)]
    #url_df is a dataframe
    #drop duplicate URLs
    url_df = url_df.drop_duplicates(subset=['GitHubUrl'])

    #now get the first index in the url_df and print
    for index, row in enumerate(url_df.itertuples()):
        if index == 0:
            url = row.GitHubUrl
            print(f"url: {url}")
            download_file(url)

if __name__ == "__main__":
    run_preprocess()
