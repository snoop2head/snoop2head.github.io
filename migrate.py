import random
import os, glob, re, shutil


PATH_PREVIOUS_IMGS = "/Users/noopy/Documents/_Archive/snoop2head.github.io/assets/images"
PATH_PREVIOUS_POSTS = "/Users/noopy/Documents/_Archive/snoop2head.github.io/_posts"

list_md_files = glob.glob(PATH_PREVIOUS_POSTS + "/*.md")

for md_file in list_md_files[1:]:
    basefile_name = os.path.basename(md_file)
    str_datetime = re.search(r'\d{4}-\d{2}-\d{2}', basefile_name).group(0)
    str_query = basefile_name.replace(str_datetime, "")
    str_query = str_query[1:] # remove the first dash
    str_query = str_query[:-3] # remove the .md extension
    print(basefile_name, str_datetime, str_query)

    # read markdown file
    with open(md_file, "r") as f:
        str_content = f.read()

    # find second ---
    splitted_content = str_content.split("---")
    for line in splitted_content[1].splitlines():
        if "title:" in line:
            str_title = line.replace("title:", "").strip()
            break
    str_content = "---".join(splitted_content[2:])
  
    tags = '["DL&ML"]'
    str_metadata = f"""---\ntitle: {str_title}\ndate: {str_datetime}\ntags: {tags}\ndraft: false\n---"""
    str_content = str_metadata + str_content
    # print(str_content)

    list_img_names = re.findall(r"!\[.*\]\((.*)\)", str_content)
    print(list_img_names)
    for img_name in list_img_names:
      if "assets/images" in img_name:
        origin_path = img_name.replace("../assets/images", PATH_PREVIOUS_IMGS)
        img_basename = os.path.basename(img_name)
        new_path_prefix = f"./image/{str_query}"
        abs_path = new_path_prefix.replace(".", "")
        abs_path = f"/Users/noopy/Documents/snoop2head.github.io/content/posts/DL&ML/{abs_path}"
        if not os.path.exists(abs_path):
          os.makedirs(abs_path)
        # copy and move from origin_path to abs_path
        shutil.copy(origin_path, abs_path)
        # replace the image path on the str_content
        str_content = str_content.replace(img_name, f"{new_path_prefix}/{img_basename}")
      else:
        continue
      
    
    # write to new file
    new_file_name = f"./content/posts/DL&ML/{str_query}.md"
    with open(new_file_name, "w") as f:
        f.write(str_content)
    
    