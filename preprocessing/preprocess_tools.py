from string import punctuation
import re

def clean_poem(text):

    cleaned_text = text

    #Remove html character
    cleaned_text = cleaned_text.replace("&nbsp", " ")
    #Remove links
    cleaned_text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', cleaned_text, flags=re.MULTILINE)


    #Remove stuff within square brackets
    cleaned_text = re.sub("\[.*?\]", "", cleaned_text)
    #Remove another html/markdown character
    cleaned_text = cleaned_text.replace("x200B", "")
    #Replace linebreaks and tabs
    cleaned_text = cleaned_text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\r", " ")
    cleaned_text = cleaned_text.replace("\t", " ")
    cleaned_text = cleaned_text.replace("\x14", " ")

     # Replace quotation marks
    cleaned_text = cleaned_text.replace('”', "")
    cleaned_text = cleaned_text.replace("“", "")

    #Replace dash
    cleaned_text = cleaned_text.replace("—", " ")

    # Remove punctuations
    cleaned_text = "".join(c for c in cleaned_text if not c in punctuation)

    #Remove extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())

    #Remove everything after "Support My Poetry"

    cleaned_text = cleaned_text[:cleaned_text.find("Support My Poetry")]




    return cleaned_text

def clean_lyrics(text):

    cleaned_text = text

    #Remove html character
    cleaned_text = cleaned_text.replace("&nbsp", " ")
    #Remove links
    cleaned_text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', cleaned_text, flags=re.MULTILINE)


    #Remove stuff within square brackets
    cleaned_text = re.sub("\[.*?\]", "", cleaned_text)
    #Remove another html/markdown character
    cleaned_text = cleaned_text.replace("x200B", "")
    #Replace linebreaks and tabs
    cleaned_text = cleaned_text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\r", " ")
    cleaned_text = cleaned_text.replace("\t", " ")
    cleaned_text = cleaned_text.replace("\x14", " ")

     # Replace quotation marks
    cleaned_text = cleaned_text.replace('”', "")
    cleaned_text = cleaned_text.replace("“", "")

    #Replace dash
    cleaned_text = cleaned_text.replace("—", " ")

    # Remove punctuations
    cleaned_text = "".join(c for c in cleaned_text if not c in punctuation)

    #Remove extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())




    return cleaned_text