words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"]
maxwidth=16
maxw=maxwidth
cache=[]
temp=[]
def classifier(word,i):
    global maxw
    global temp
    global cache
    global words
    maxw=maxw-len(word+' ')
    if maxw >=0:
        temp.append(word)
        if(i==len(words)-1):
            cache.append(temp)
        return(1)
    if maxw <0:
        maxw=maxwidth
        cache.append(temp)
        temp=[]
        return(0)
for i in range(len(words)):
    word=words[i]
    r=classifier(word,i)
    if r == 0:
        classifier(word,i)
print(cache)
    


    