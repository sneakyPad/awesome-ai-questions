import markdown

from bs4 import BeautifulSoup
from bs4 import NavigableString
import json
import urllib.request
import os


question_count = 1

def notionmd2githubmd(md_file, idx):
    global question_count

    export_dir_name = md_file.split('/')[1]
    f = open(md_file, 'r')
    htmlmarkdown = markdown.markdown(f.read())

    i = 0

    soup = BeautifulSoup(htmlmarkdown, 'html.parser')
    new_tag = soup.new_tag("b")
    # a_tag.i.replace_with(new_tag)
    tag = soup.new_tag("b")
    tag.string = "Answer"
    # tmp = soup.ul.contents
    # soup.ul.replaceWith(soup.new_tag("ol"))
    # soup.ol.contents = tmp


    # soup.ul.decompose()

    for topic in soup.contents:
        if (type(topic) == NavigableString or topic.name == 'h1' or topic.name == 'h2' or topic.name=='h3'):
            continue
        # tmp = topic.ul.contents
        topic.name = "ol start="+str(question_count)
        # tmp = topic.contents
        # topic.replaceWith(soup.new_tag("ol", start=question_count))
        # topic.ol.contents = tmp
        # topic.select('ul:first-child')
        # topic.replaceWith(soup.new_tag("ol", start=question_count))
        i=0

        # < details > < summary > < b > Answer < / b > < / summary >
        for item in topic.contents:
            if (type(item) == NavigableString):
                i+=1
                continue

            try:

                tmpTag = soup.new_tag("p")
                removeTag = soup.new_tag("p")
                if(item.p != None):
                    node = item.p.next_sibling
                    while True:
                        # node = node.next_sibling

                        if (node == None):
                            break
                        next_node = node.next_sibling
                        if (node == '\n'):
                            removeTag.append(node)
                        contains_images = node.find('img')
                        if (contains_images!=-1 and contains_images!=None):
                            childtags = node.findAll('img')

                            if (len(childtags) > 0):
                                for childtag in childtags:
                                    childtag.attrs['src'] = 'data/' + childtag.attrs['src']
                                    childtag.attrs['alt'] = 'data/' + childtag.attrs['alt']
                                    childtag.attrs['width'] = "50%"
                                    childtag.attrs['height'] = "50%"

                        tmpTag.append(node)  # Insert original answer (ul) at the end of details
                        node = next_node

                # childtag = item.find('img')
                # if (childtag):
                #     tmpTag.img['src'] = 'data/' + tmpTag.img['src']
                #     tmpTag.img['alt'] = 'data/' + tmpTag.img['alt']
                # Create tags to make the answer expandable
                bTag = soup.new_tag("b")
                bTag.string = "Answer"
                if(item.p == None):
                    item.ul.insert_before(bTag)
                else:
                    item.p.insert_after(bTag)



                sumTag = soup.new_tag("summary")
                item.b.wrap(sumTag)

                detailsTag = soup.new_tag("details")
                item.summary.wrap(detailsTag)


                # Until here:
                # < p > question? < / p >
                # < details > < summary > < b > Answer < / b > < / summary > < / details >
                item.details.append(item.summary)  # Insert summary tag into details
                if(item.p == None):

                    item.details.append(item.ul)  # Insert original answer (ul) at the end of details
                else:
                    item.details.append(tmpTag)  # Insert original answer before the end tag of details

                # if (item.img != None):
                #     item.img['src'] = 'data/' + item.img['src']
                #     # item.img['src'] = 'data/' + export_dir_name+ '/'+ item.img['src']
                #     item.img['alt'] = 'data/' + item.img['alt']
                #     # item.img['alt'] = 'data/' + export_dir_name+ '/'+  item.img['alt']
                #     item.details.append(item.img)
                # for s in ["summary", "b", "details"]:
                #     aTag = soup.new_tag(s)
                #     if(s == 'b'):
                #         aTag.string = "Answer"
                #     if(s == "details"):
                #         item.ul.wrap(aTag)
                #     else:
                #         item.ul.insert_before(aTag)

                # item.ul.insert_before(tag)
                # print(item)

                # Remove li tag
                # item.li.decompose()
            except AttributeError:
                print('Question does not have an answer: ', item)

            # soup.ol.contents[i] = item
            topic.contents[i] = item
            i += 1
            question_count += 1

            # print(soup)
    with open("output/section_"+str(idx)+".md", "w") as file:
        file.write(str(soup))

def iterate_export_directories(base_path):
    idx = 1
    for subdir, dirs, files in os.walk(base_path):
        files.sort()
        for file in files:
            name, extension = os.path.splitext(file)
            if(extension=='.md'):
                md_file = subdir + '/' + file
                notionmd2githubmd(md_file, idx)
                idx+=1
            print(os.path.join(subdir, file))

def combine_mds(output_path):
    os.remove("README.md")
    with open("README.md", "a") as combined_readme:

        with open("introduction.md", "r") as md_description:
            combined_readme.write(md_description.read())

        for subdir, dirs, files in os.walk(output_path):
            files.sort()
            for file in files:
                name, extension = os.path.splitext(file)
                if(extension=='.md'):
                    md_file = subdir  + file

                    with open(md_file, "r") as output_md:
                        combined_readme.write('\n')
                        combined_readme.write(output_md.read())
                    # f2 = open(md_file, 'r')
                    # # with open("README.md", "a") as myfile:
                    # combined_readme.write(f2.read())
                    # f2.close()

def main():
    print("Start transforming...")
    base_path = "data/"
    output_path = "output/"
    iterate_export_directories(base_path)
    combine_mds(output_path)





if __name__ == "__main__":
    main()
