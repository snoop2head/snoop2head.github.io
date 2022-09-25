import os, glob, re, shutil


PATH_PREVIOUS_IMGS = "/Users/noopy/Documents/_Archive/snoop2head.github.io/assets/images"
PATH_PREVIOUS_POSTS = "/Users/noopy/Documents/_Archive/snoop2head.github.io/_posts"

list_md_files = glob.glob(PATH_PREVIOUS_POSTS + "/*.md")

for md_file in list_md_files:
    basefile_name = os.path.basename(md_file)
    str_datetime = re.search(r'\d{4}-\d{2}-\d{2}', basefile_name).group(0)
    str_query = basefile_name.replace(str_datetime, "")
    str_query = str_query[1:] # remove the first dash
    str_query = str_query[:-3] # remove the .md extension
    print(basefile_name, str_datetime, str_query)
    
    # with open(md_file, "r") as f:
    #     content = f.read()
    #     list_imgs = re.findall(r"!\[.*\]\((.*)\)", content)
    #     print(list_imgs)
    #     for img in list_imgs:
    #         img_name = img.split("/")[-1]
    #         print(img_name)
    #         shutil.copy(PATH_PREVIOUS_IMGS + "/" + img_name, "./assets/images/" + img_name)

    # new_content = re.sub(r"!\[.*\]\((.*)\)", r"![\1](/assets/images/\1)", content)
    # with open(md_file, "w") as f:
    #     f.write(new_content)

    # print(new_content)

assert 1 == 2
str_metadata = f"""
---
title: {title}
date: {date} {time}
tags: {tags}
draft: false
---
"""

