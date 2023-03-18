#!/usr/bin/env python
"""
Generate CodeSearchNet compatible Data from open source repos.

Usage:
    get_data.py [-d] [-q QUERY_CSV] [-a ANNOTATIONS_CSV] [-o DATA_CSV] [-f FOLDER_PATH] [-m PROCESSING_METHOD]
    get_data.py -h | --help

Options:
    -h --help                                      Show command line options
    -d --dry_run                                   Run only for one entry in annotations.csv
                                                   [default: False]
    -q --query_csv FILENAME                        CSV filename for NL queries
                                                   [default: queries.csv]
    -a --annotation_csv FILENAME                   CSV filename for annotations
                                                   [default: annotations.csv]
    -o --output_csv FILENAME                       CSV filename for CodeSearchNet dataset
                                                   [default: output.csv]
    -f --folder_path FOLDERPATH                    Folder path for the generate data
                                                   [default: ./]
    -m --preprocessing_method PROCESSINGMETHOD     Select method to process data, Supported - 0(code as code token), 1 (code+docstring as code token)
                                                   [default: 0]

Examples:
    ./get_data.py -d
    ./get_data.py -q examples.csv
"""

"""
This script prepares the dataset compatible for CodeSearchNet Challenge, given
- queries.csv file which has the NL queries
- annotations.csv file which has the annotations
Run with "-h" to see full command line options
"""
from fast_ml.model_development import train_valid_test_split
import pandas as pd
import re
from urllib.parse import urlsplit
from github import Github
import os
import ast
import astor
from nltk.tokenize import RegexpTokenizer
import json
from docopt import docopt
from os.path import exists

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

def tokenize_docstring(text):
    """Gets filetered docstring tokens which help describe the function"""

    # Remove decorators and other parameter signatures in the docstring
    before_keyword, keyword, after_keyword = text.partition(':')
    before_keyword, keyword, after_keyword = before_keyword.partition('@param')
    before_keyword, keyword, after_keyword = before_keyword.partition('param')
    before_keyword, keyword, after_keyword = before_keyword.partition('@brief')

    if(after_keyword):
        words = RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(after_keyword)
    else:
        before_keyword, keyword, after_keyword = before_keyword.partition('@')
        words = RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(before_keyword)

    # Convert all docstrings to lowercase
    new_words= [word.lower() for word in words if word.isalnum()]
    return new_words


def tokenize_code(text):
    """Gets filetered fucntion tokens"""

    # Remove decorators and function signatures till the def token
    keyword = 'def '
    before_keyword, keyword, after_keyword = text.partition(keyword)
    words = RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(after_keyword)

    # Convert function tokens to lowercase and remove single alphabet variables
    new_words= [word.lower() for word in words if (word.isalpha() and len(word)>1) or (word.isnumeric())]
    return new_words

def get_function_details_from_string(input_str):
    return_functions = []
    input_ast = ast.parse(input_str)
    input_classes = [seg for seg in input_ast.body if isinstance(seg, ast.ClassDef)]
    input_functions = [seg for seg in input_ast.body if isinstance(seg, ast.FunctionDef)]
    for class1 in input_classes:
        input_functions.extend([seg for seg in class1.body if isinstance(seg, ast.FunctionDef)])
    for func in input_functions:
        docstring = ast.get_docstring(func) if ast.get_docstring(func) else ''
        function_full = astor.to_source(func)
        function_code = function_full.replace(ast.get_docstring(func, clean=False), "") if docstring else function_full
        function_token = tokenize_code(function_code)
        docstring_token = tokenize_docstring(docstring.split('\n\n')[0])
        return_functions.append((func.name, function_code, function_token, docstring, docstring_token))
    return return_functions

idx=0
def prepare_dataset(url, query, filename, preprocessing_method):
    global idx
    result_dict = {}
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
            func = f.readlines()[int(start)-1:int(end)]
            print(func)
            try:
                function_details = get_function_details_from_string("".join(func).lstrip())
                result_dict["url"] = url
                result_dict["repo"] = repo_name
                result_dict["func_name"] = function_details[0][0]
                result_dict["original_string"] = func
                result_dict["language"] = "python"
                if ((preprocessing_method == "1")):
                    result_dict["code"] = function_details[0][1] + function_details[0][3]
                    result_dict["code_tokens"] = function_details[0][2] + function_details[0][4]
                else:
                    result_dict["code"] = function_details[0][1]
                    result_dict["code_tokens"] = function_details[0][2]
                result_dict["docstring"] = query
                result_dict["docstring_tokens"] = tokenize_docstring(query.split('\n\n')[0])
                result_dict["idx"] = idx
                idx += 1
                json_object = json.dumps(result_dict)
                with open(filename, "a") as outfile:
                    outfile.write(json_object)
                    outfile.write("\n")
            except:
                pass
    os.remove("temp.txt")

def fetch_annotations(ann_df, query, dry_run, filename, preprocessing_method):
    url_df = ann_df[(ann_df["Language"] == "Python") & (ann_df["Query"] == query)]
    #url_df is a dataframe
    #drop duplicate URLs
    url_df = url_df.drop_duplicates(subset=['GitHubUrl'])

    #now get the first index in the url_df and print
    for index, row in enumerate(url_df.itertuples()):
        if dry_run and index > 0:
            break
        url = row.GitHubUrl
        print(f"url: {url}")
        prepare_dataset(url, query, filename, preprocessing_method)


def run_preprocess():
    #load the queries
    queries_file = args.get('--query_csv')
    df=pd.read_csv(queries_file)
    #load the annotations.csv, filtered files
    annotations_file = args.get('--annotation_csv')
    ann_df = pd.read_csv(annotations_file)

    #check for dry run
    dry_run = args.get('--dry_run')
    #split the data into train, valid, test
    x_train, y_train, x_valid, y_valid, x_test, y_test = train_valid_test_split(df,
                                                                                target='query',
                                                                                train_size=0.7,
                                                                                valid_size = 0.15,
                                                                                test_size =0.15)
    #folder where files would be dumped.
    folder_path = args.get("--folder_path")
    if not exists(folder_path):
        os.mkdir(folder_path)
    train_path = folder_path+"/train.jsonl"
    valid_path = folder_path+"/valid.jsonl"
    test_path = folder_path+"/test.jsonl"

    #Choose Preprocessing method
    preprocessing_method = args.get("--preprocessing_method")

    #reset the index
    for data in [x_train, y_train, x_valid, y_valid, x_test, y_test]:
        data.reset_index(drop=True, inplace=True)

    #split dataset size
    print(f"train: {x_train.shape}, {y_train.shape}")
    print(f"valid: {x_valid.shape}, {y_valid.shape}")
    print(f"test: {x_test.shape}, {y_test.shape}")


    #iterate through the training queries
    file_exists = exists(train_path)
    if file_exists:
        os.remove(train_path)
    for index,row in enumerate(y_train):
        if dry_run and index > 0:
            break
        print(f"query : {row}")
        fetch_annotations(ann_df, row, dry_run, train_path, preprocessing_method)
    file_exists = exists(valid_path)
    if file_exists:
        os.remove(valid_path)
    for index,row in enumerate(y_valid):
        if dry_run and index > 0:
            break
        print(f"query : {row}")
        fetch_annotations(ann_df, row, dry_run, valid_path, preprocessing_method)
    file_exists = exists(test_path)
    if file_exists:
        os.remove(test_path)
    for index,row in enumerate(y_test):
        if dry_run and index > 0:
            break
        print(f"query : {row}")
        fetch_annotations(ann_df, row, dry_run, test_path, preprocessing_method)
    

if __name__ == "__main__":
    args = docopt(__doc__)
    run_preprocess()
